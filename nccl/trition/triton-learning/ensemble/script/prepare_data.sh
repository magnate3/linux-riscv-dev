#!/bin/bash
# 必须要在 conceptual-guide 目录下执行
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890
if [ ! -d ../data ] ; then
    echo "Executing in incorrect folder"
    exit 1
fi

current_dir="$PWD"
parent_dir="${current_dir%/*}"
WORKSPACE="$parent_dir"
echo "workspace is ${WORKSPACE}"
cd ../data

# 下载 text_detection 模型文件
if [ ! -f frozen_east_text_detection.tar.gz ] ; then
    wget https://www.dropbox.com/s/r2ingd0l3zt8hxs/frozen_east_text_detection.tar.gz
fi
if [ ! -f frozen_east_text_detection.pb ] ; then
    tar -xvf frozen_east_text_detection.tar.gz
fi

if [ ! -f detection.onnx ] ; then
    sudo docker run --rm --gpus all -v ${WORKSPACE}:/workspace nvcr.io/nvidia/tensorflow:24.07-tf2-py3 /bin/bash -c \
    "cd /workspace/data && sh /workspace/conceptual-guide/tf2onnx.sh"
fi

if [ ! -f None-ResNet-None-CTC.pth ] ; then
    wget https://www.dropbox.com/sh/j3xmli4di1zuv3s/AABzCC1KGbIRe2wRwa3diWKwa/None-ResNet-None-CTC.pth
fi

if [ ! -f str.onnx ] ; then
    sudo docker run --rm --gpus all -v ${WORKSPACE}:/workspace nvcr.io/nvidia/pytorch:24.07-py3 /bin/bash -c \
    "cd /workspace/data && python /workspace/conceptual-guide/utils/pth2onnx.py --input None-ResNet-None-CTC.pth --output str.onnx"
fi

if [ ! -f batch_str.onnx ] ; then
    sudo docker run --rm --gpus all -v ${WORKSPACE}:/workspace nvcr.io/nvidia/pytorch:24.07-py3 /bin/bash -c \
    "cd /workspace/data && python /workspace/conceptual-guide/utils/pth2onnx.py --input None-ResNet-None-CTC.pth --output batch_str.onnx --batch"
fi

if [ ! -f detection.onnx ] ; then
    echo "Failed without detection.onnx model file"
    exit 1
fi

if [ ! -f str.onnx ] ; then
    echo "Failed without str.onnx model file"
    exit 1
fi

if [ ! -f batch_str.onnx ] ; then
    echo "Failed without batch_str.onnx model file"
    exit 1
fi

echo "Succcess"
