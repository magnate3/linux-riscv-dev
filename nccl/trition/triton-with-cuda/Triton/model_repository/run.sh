#!/bin/bash
set -e

export CUDA_ARCH_LIST=native
cmake -S detection_postprocessing_cuda -B /tmp/postprocess_build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build /tmp/postprocess_build --parallel
cp /tmp/postprocess_build/libtriton_postprocess.so detection_postprocessing_cuda/1

tritonserver --model-repository=/models --log-verbose=0