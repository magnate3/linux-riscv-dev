#!/bin/bash

# Start Triton server with Docker
docker run --gpus=1 --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models \
    nvcr.io/nvidia/tritonserver:23.12-py3 \
    tritonserver --model-repository=/models \
    --log-verbose=1 \
    --log-info=1 \
    --model-control-mode=explicit

# Notes:
# Port 8000: HTTP inference API
# Port 8001: gRPC inference API
# Port 8002: HTTP metrics API
# --model-control-mode=explicit means you need to explicitly load model
