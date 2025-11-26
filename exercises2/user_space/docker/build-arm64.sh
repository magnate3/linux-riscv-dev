#!/bin/bash
#export ARCH=arm64
#export OS=ubuntu20.04
#export DISTRO=ubuntu2004
#export PYTHON_VERSION=3.8
#export PYTORCH_VERSION=1.12.1
#export NCCL_VERSION=2.9.6
#export CUDA_VERSION=11.3
ARCH=arm64
OS=ubuntu20.04
DISTRO=ubuntu2004
PYTHON_VERSION=3.8
PYTORCH_VERSION=1.12.1
NCCL_VERSION=2.9.6
CUDA_VERSION=11.3

DOCKER_FILE_PATH=./arm64v8-test/Dockerfile
PYTORCH_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu116
PYTORCH_GEOMETRIC_URL=https://data.pyg.org/whl/torch-1.13.1+cu116.html
LIB_NCCL="null"

OUTPUT_IMAGE=arm64:fedmlFull
#./build-docker.sh $ARCH $OS $DISTRO $PYTHON_VERSION $PYTORCH_VERSION $NCCL_VERSION $CUDA_VERSION
docker build -f $DOCKER_FILE_PATH \
    --build-arg OS=$OS \
    --build-arg DISTRO=$DISTRO \
    --build-arg PYTHON_VERSION=$PYTHON_VERSION \
    --build-arg PYTORCH_VERSION=$PYTORCH_VERSION \
    --build-arg NCCL_VERSION=$NCCL_VERSION \
    --build-arg CUDA_VERSION=$CUDA_VERSION \
    --build-arg PYTORCH_EXTRA_INDEX_URL=$PYTORCH_EXTRA_INDEX_URL \
    --build-arg PYTORCH_GEOMETRIC_URL=$PYTORCH_GEOMETRIC_URL \
    --build-arg LIB_NCCL=$LIB_NCCL \
    --network=host \
    -t $OUTPUT_IMAGE .
