#!/bin/bash

# Start Triton server with Docker
#nvcr.io/nvidia/tritonserver:20.12-py3 \
#nvcr.io/nvidia/tritonserver:20.12-py3-proto3.20 \
docker run  --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:20.12-py3 \
    tritonserver --model-repository=/models \
    --log-verbose=1 \
    --log-info=1 \
    --model-control-mode=explicit

# Notes:
# Port 8000: HTTP inference API
# Port 8001: gRPC inference API
# Port 8002: HTTP metrics API
# --model-control-mode=explicit means you need to explicitly load model
