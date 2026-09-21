from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:  # cuda-python < 13
    from cuda import cudart  # type: ignore


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def cuda_check(result):
    err, *values = result
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    if not values:
        return None
    return values[0] if len(values) == 1 else values


def parse_shape(text: str) -> tuple[int, ...]:
    shape = tuple(int(x) for x in text.split(","))
    if not shape or any(x <= 0 for x in shape):
        raise argparse.ArgumentTypeError("shape must contain positive integers, e.g. 1,3,224,224")
    return shape


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    input_shape: tuple[int, ...],
    fp16: bool,
    workspace_gib: float,
) -> None:
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    model = onnx_path.read_bytes()
    if not parser.parse(model):
        messages = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed:\n{messages}")

    if network.num_inputs != 1:
        raise ValueError(f"This compact runner expects one input tensor; model has {network.num_inputs}")

    input_tensor = network.get_input(0)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gib * (1 << 30)),
    )

    if fp16:
        if not builder.platform_has_fast_fp16:
            print("warning: platform does not advertise fast FP16; TensorRT may select FP32 tactics")
        config.set_flag(trt.BuilderFlag.FP16)

    if any(dim == -1 for dim in input_tensor.shape):
        if len(input_tensor.shape) != len(input_shape):
            raise ValueError(
                f"input rank mismatch: network={tuple(input_tensor.shape)}, requested={input_shape}"
            )
        min_shape = tuple(1 if d == -1 else d for d in input_tensor.shape)
        opt_shape = input_shape
        max_shape = tuple(
            max(o, 4 if i == 0 else o)
            if d == -1
            else d
            for i, (d, o) in enumerate(zip(input_tensor.shape, input_shape))
        )
        profile = builder.create_optimization_profile()
        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)
    elif tuple(input_tensor.shape) != input_shape:
        raise ValueError(
            f"static model expects {tuple(input_tensor.shape)}, not requested {input_shape}"
        )

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build a serialized engine")

    engine_path.write_bytes(bytes(serialized))
    print(f"wrote TensorRT engine: {engine_path}")


class TensorRTRunner:
    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda_check(cudart.cudaStreamCreate())

        self.input_names = []
        self.output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)

        if len(self.input_names) != 1:
            raise ValueError(f"runner expects one input; found {self.input_names}")

    def close(self):
        if self.stream is not None:
            cuda_check(cudart.cudaStreamDestroy(self.stream))
            self.stream = None

    def infer(self, array: np.ndarray) -> list[np.ndarray]:
        name = self.input_names[0]
        expected_dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
        x = np.ascontiguousarray(array, dtype=expected_dtype)

        if not self.context.set_input_shape(name, tuple(x.shape)):
            raise ValueError(f"TensorRT rejected input shape {x.shape} for {name}")

        allocations: list[int] = []
        outputs: list[tuple[np.ndarray, int]] = []

        try:
            d_input = int(cuda_check(cudart.cudaMalloc(x.nbytes)))
            allocations.append(d_input)
            self.context.set_tensor_address(name, d_input)

            cuda_check(
                cudart.cudaMemcpyAsync(
                    d_input,
                    int(x.ctypes.data),
                    x.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )

            for out_name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(out_name))
                if any(d < 0 for d in shape):
                    raise RuntimeError(f"unresolved output shape for {out_name}: {shape}")
                dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(out_name)))
                host = np.empty(shape, dtype=dtype)
                d_output = int(cuda_check(cudart.cudaMalloc(host.nbytes)))
                allocations.append(d_output)
                self.context.set_tensor_address(out_name, d_output)
                outputs.append((host, d_output))

            if not self.context.execute_async_v3(stream_handle=self.stream):
                raise RuntimeError("TensorRT execution failed")

            for host, d_output in outputs:
                cuda_check(
                    cudart.cudaMemcpyAsync(
                        int(host.ctypes.data),
                        d_output,
                        host.nbytes,
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self.stream,
                    )
                )

            cuda_check(cudart.cudaStreamSynchronize(self.stream))
            return [host for host, _ in outputs]
        finally:
            for ptr in allocations:
                cuda_check(cudart.cudaFree(ptr))


def benchmark(
    engine_path: Path,
    shape: tuple[int, ...],
    warmup: int,
    iterations: int,
) -> dict[str, float | int | list[int]]:
    runner = TensorRTRunner(engine_path)
    rng = np.random.default_rng(7)
    x = rng.standard_normal(shape, dtype=np.float32)

    try:
        for _ in range(warmup):
            runner.infer(x)

        samples = []
        for _ in range(iterations):
            start = time.perf_counter()
            runner.infer(x)
            samples.append((time.perf_counter() - start) * 1000.0)
    finally:
        runner.close()

    a = np.asarray(samples)
    return {
        "input_shape": list(shape),
        "iterations": iterations,
        "mean_ms": round(float(a.mean()), 4),
        "p50_ms": round(float(np.percentile(a, 50)), 4),
        "p95_ms": round(float(np.percentile(a, 95)), 4),
        "throughput_per_s": round(float(1000.0 / a.mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and benchmark TensorRT engines")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("onnx", type=Path)
    build.add_argument("engine", type=Path)
    build.add_argument("--input-shape", type=parse_shape, required=True)
    build.add_argument("--fp16", action="store_true")
    build.add_argument("--workspace-gib", type=float, default=2.0)

    bench = sub.add_parser("bench")
    bench.add_argument("engine", type=Path)
    bench.add_argument("--input-shape", type=parse_shape, required=True)
    bench.add_argument("--warmup", type=int, default=10)
    bench.add_argument("--iterations", type=int, default=100)
    bench.add_argument("--json-out", type=Path)

    args = parser.parse_args()
    if args.command == "build":
        build_engine(args.onnx, args.engine, args.input_shape, args.fp16, args.workspace_gib)
        return

    result = benchmark(args.engine, args.input_shape, args.warmup, args.iterations)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.json_out:
        args.json_out.write_text(payload + "\n")


if __name__ == "__main__":
    main()
