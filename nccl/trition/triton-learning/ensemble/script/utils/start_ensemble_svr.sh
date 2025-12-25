#!/bin/bash
export http_proxy=127.0.0.1:7890
export https_proxy=127.0.0.1:7890

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
cp $WORKSPACE/data/batch_str.onnx $WORKSPACE/data/model_repository/text_recognition/1/model.onnx

mkdir -p $WORKSPACE/data/model_repository/detection_preprocessing/1
cp $WORKSPACE/conceptual-guide/model-backend/detection_preprocessing_model.py $WORKSPACE/data/model_repository/detection_preprocessing/1/model.py
mkdir -p $WORKSPACE/data/model_repository/detection_postprocessing/1
cp $WORKSPACE/conceptual-guide/model-backend/detection_postprocessing_model.py $WORKSPACE/data/model_repository/detection_postprocessing/1/model.py
mkdir -p $WORKSPACE/data/model_repository/recognition_postprocessing/1
cp $WORKSPACE/conceptual-guide/model-backend/recognition_postprocessing_model.py $WORKSPACE/data/model_repository/recognition_postprocessing/1/model.py
mkdir -p $WORKSPACE/data/model_repository/ensemble_model/1

cat <<EOF > $WORKSPACE/data/model_repository/text_detection/config.pbtxt
name: "text_detection"
backend: "onnxruntime"
max_batch_size : 2
input [
  {
    name: "input_images:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, 3 ]
  }
]
output [
  {
    name: "feature_fusion/Conv_7/Sigmoid:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, 1 ]
  }
]
output [
  {
    name: "feature_fusion/concat_3:0"
    data_type: TYPE_FP32
    dims: [ -1, -1, 5 ]
  }
]
EOF

cat <<EOF > $WORKSPACE/data/model_repository/text_recognition/config.pbtxt
name: "text_recognition"
backend: "onnxruntime"
max_batch_size : 2
input [
  {
    name: "input.1"
    data_type: TYPE_FP32
    dims: [ 1, 32, 100 ]
  }
]
output [
  {
    name: "308"
    data_type: TYPE_FP32
    dims: [ 26, 37 ]
  }
]
EOF

cat <<EOF > $WORKSPACE/data/model_repository/detection_postprocessing/config.pbtxt
name: "detection_postprocessing"
backend: "python"
max_batch_size: 2
input [
{
    name: "detection_postprocessing_input_1"
    data_type: TYPE_FP32
    dims: [ -1, -1, 1 ]
},
{
    name: "detection_postprocessing_input_2"
    data_type: TYPE_FP32
    dims: [ -1, -1, 5 ]
},
{
    name: "detection_postprocessing_input_3"
    data_type: TYPE_FP32
    dims: [ -1, -1, 3 ]
}
]

output [
{
    name: "detection_postprocessing_output"
    data_type: TYPE_FP32
    dims: [ -1, -1, -1 ]
}
]
instance_group [{ kind: KIND_CPU }]
EOF


cat <<EOF > $WORKSPACE/data/model_repository/detection_preprocessing/config.pbtxt
name: "detection_preprocessing"
backend: "python"
max_batch_size: 2
input [
{
    name: "detection_preprocessing_input"
    data_type: TYPE_UINT8
    dims: [ -1 ]
}
]

output [
{
    name: "detection_preprocessing_output"
    data_type: TYPE_FP32
    dims: [ -1, -1, 3 ]
}
]

instance_group [{ kind: KIND_CPU }]
EOF


cat <<EOF > $WORKSPACE/data/model_repository/recognition_postprocessing/config.pbtxt
name: "recognition_postprocessing"
backend: "python"
max_batch_size: 2
input [
    {
        name: "recognition_postprocessing_input"
        data_type: TYPE_FP32
        dims: [ 26, 37]
    }
]
output [
    {
        name: "recognition_postprocessing_output"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

instance_group [{ kind: KIND_CPU }]
EOF

cat <<EOF > $WORKSPACE/data/model_repository/recognition_postprocessing/config.pbtxt
name: "recognition_postprocessing"
backend: "python"
max_batch_size: 256
input [
    {
        name: "recognition_postprocessing_input"
        data_type: TYPE_FP32
        dims: [ 26, 37]
    }
]
output [
    {
        name: "recognition_postprocessing_output"
        data_type: TYPE_STRING
        dims: [ -1 ]
    }
]

instance_group [{ kind: KIND_CPU }]
EOF

cat <<EOF > $WORKSPACE/data/model_repository/ensemble_model/config.pbtxt
name: "ensemble_model"
platform: "ensemble"
max_batch_size: 2
input [
  {
    name: "input_image"
    data_type: TYPE_UINT8
    dims: [ -1 ]
  }
]
output [
  {
    name: "recognized_text"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "detection_preprocessing"
      model_version: -1
      input_map {
        key: "detection_preprocessing_input"
        value: "input_image"
      }
      output_map {
        key: "detection_preprocessing_output"
        value: "preprocessed_image"
      }
    },
    {
      model_name: "text_detection"
      model_version: -1
      input_map {
        key: "input_images:0"
        value: "preprocessed_image"
      }
      output_map {
        key: "feature_fusion/Conv_7/Sigmoid:0"
        value: "Sigmoid:0"
      },
      output_map {
        key: "feature_fusion/concat_3:0"
        value: "concat_3:0"
      }
    },
    {
      model_name: "detection_postprocessing"
      model_version: -1
      input_map {
        key: "detection_postprocessing_input_1"
        value: "Sigmoid:0"
      }
      input_map {
        key: "detection_postprocessing_input_2"
        value: "concat_3:0"
      }
      input_map {
        key: "detection_postprocessing_input_3"
        value: "preprocessed_image"
      }
      output_map {
        key: "detection_postprocessing_output"
        value: "cropped_images"
      }
    },
    {
      model_name: "text_recognition"
      model_version: -1
      input_map {
        key: "input.1"
        value: "cropped_images"
      }
      output_map {
        key: "308"
        value: "recognition_output"
      }
    },
    {
      model_name: "recognition_postprocessing"
      model_version: -1
      input_map {
        key: "recognition_postprocessing_input"
        value: "recognition_output"
      }
      output_map {
        key: "recognition_postprocessing_output"
        value: "recognized_text"
      }
    }
  ]
}
EOF

if [ ! -f ${WORKSPACE}/data/img1.jpg ] ; then
cd $WORKSPACE/data
wget 'https://raw.githubusercontent.com/triton-inference-server/tutorials/main/Conceptual_Guide/Part_1-model_deployment/img1.jpg'
fi

CONTAINER_NAME=triton-ensemble-model

if sudo docker ps -a | grep -q "$CONTAINER_NAME"; then
    sudo docker start -ai "$CONTAINER_NAME"
else
sudo docker run --gpus=all -it --shm-size=1024m -p8000:8000 -p8001:8001 -p8002:8002 -v ${WORKSPACE}/data/model_repository:/models --name $CONTAINER_NAME nvcr.io/nvidia/tritonserver:24.07-py3 /bin/bash -c \
"pip install torchvision opencv-python-headless -i https://pypi.tuna.tsinghua.edu.cn/simple ; tritonserver --model-repository=/models"
fi
