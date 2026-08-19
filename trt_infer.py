try:
    import tensorrt as trt
except Exception:
    print("TensorRT not installed. This script demonstrates the API usage for systems with TensorRT.")
    raise

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
def main():
    print("TensorRT scaffold - build engine from ONNX here")

if __name__ == '__main__':
    main()
