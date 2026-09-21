FROM nvcr.io/nvidia/tensorrt:25.08-py3
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY trt_infer.py .
ENTRYPOINT ["python", "trt_infer.py"]
