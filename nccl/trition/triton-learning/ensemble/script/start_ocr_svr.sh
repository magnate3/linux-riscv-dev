#!/bin/bash
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

if [ -d $WORKSPACE/data/model_repository ] ; then
 sudo rm -rf $WORKSPACE/data/model_repository
fi

mkdir -p $WORKSPACE/data/model_repository/text_detection/1
cp $WORKSPACE/data/detection.onnx $WORKSPACE/data/model_repository/text_detection/1/model.onnx

mkdir -p $WORKSPACE/data/model_repository/text_recognition/1
cp $WORKSPACE/data/str.onnx $WORKSPACE/data/model_repository/text_recognition/1/model.onnx

cat <<EOF > $WORKSPACE/data/model_repository/text_detection/config.pbtxt
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 0
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1, 5 ]
  }
]
EOF

cat <<EOF > $WORKSPACE/data/model_repository/text_recognition/config.pbtxt
name: "text_recognition"
backend: "onnxruntime"
max_batch_size : 0
input [
  {
    name: "input.1"
    data_type: TYPE_FP32
    dims: [ 1, 1, 32, 100 ]
  }
]
output [
  {
    name: "308"
    data_type: TYPE_FP32
    dims: [ 1, 26, 37 ]
  }
]
EOF

if [ ! -f ${WORKSPACE}/data/img1.jpg ] ; then
cd $WORKSPACE/data
wget 'https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Conceptual_Guide/Part_1-model_deployment/img1.jpg'
fi

sudo docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${WORKSPACE}/data/model_repository:/models nvcr.io/nvidia/tritonserver:24.07-py3 \
tritonserver --model-repository=/models