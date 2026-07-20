#!/bin/bash

# Create model repository structure
mkdir -p model_repository/resnet18/1

# Copy your ONNX model to the repository
cp ./resnet18_fp32.onnx model_repository/resnet18/1/model.onnx

# Create config.pbtxt for model configuration
cat > model_repository/resnet18/config.pbtxt << EOF
name: "resnet18"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input.1"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
instance_group [
  {
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 4, 8, 16, 32 ]
  max_queue_delay_microseconds: 100
}

