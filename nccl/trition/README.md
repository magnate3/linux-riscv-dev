

# <ins>System Setup</ins>

_NOTE: For this demo, all the following should happen directly on the Jetson Orin Nano Dev Kit._

## Setup Python Environment.
```
> mkdir ~/git
> cd ~/git
> sudo apt install python3.10-venv
> python -m venv infer_env_jetson
> source infer_env_jetson/bin/activate
> cd infer_env_jetson/
```
## Setup this demo's repo
```
> sudo apt-get install git-lfs
> git clone git@github.com:dsdickinson/engineering.git
> cd engineering/python/ai/computer_vision/demo-01/
> git lfs fetch --all
> git lfs pull
> sudo apt-get install libhdf5-dev (for hdf5 Python package)
> pip3 install --upgrade pip setuptools wheel # (will help w/ requirements.txt installs)
> pip3 install -r ./requirements.txt --no-cache-dir > requirements_install.txt
```

## Get Triton Client/Server bits.
```
> cd ~/git/
> mkdir triton-inference-server
> cd triton-inference-server/
> git clone -b r24.12 https://github.com/triton-inference-server/server.git
> git clone -b r24.12 https://github.com/triton-inference-server/client.git
```

## Setup Base Triton Models.
```
> cd server/docs/examples/
> ./fetch_models.sh
> sudo cp -rf model_repository /models
```

## Setup object detection model.
### Get the model and compile the .proto files.
```
> cd ~/git/
> git clone git@github.com:tensorflow/models.git
> cd models/research/
> sudo apt install protobuf-compile
> protoc object_detection/protos/*.proto --python_out=.
> cp object_detection/packages/tf2/setup.py .
```




## Triton Server
### Start server
```
> sudo docker run -d --gpus=1 --runtime=nvidia --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/models:/models nvcr.io/nvidia/tritonserver:24.01-py3-igpu tritonserver --model-repository=/models --strict-model-config=false
```

```
root@ubuntux86:# ls model_repository
densenet_onnx
root@ubuntux86:# bash my_triton_server_launch.sh 
```

```
root@ubuntux86:# tree  -L 3 model_repository-v1/
model_repository-v1/
└── densenet_onnx
    ├── 1
    │   └── model.onnx
    ├── config.pbtxt
    └── densenet_labels.txt

2 directories, 3 files
root@ubuntux86:# 
```

### Validations
### Check Triton server is ready.
```
> curl -v http://localhost:8000/v2/health/ready
```

### Check model config.
```
> curl http://localhost:8000/v2/models/object_detection/config | jq
```

### Run a test inference request against an image.


+ 20.12-py3-sdk docker中

```
docker run -it --rm --net=host -v /work/fedAgg/:/workspace2 nvcr.io/nvidia/tritonserver:20.12-py3-sdk bash
```

```
root@ubuntux86:/workspace2# python3 image_client.py   -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
Request 1, batch size 1
    13.900130 (504) = COFFEE MUG
    11.999283 (968) = CUP
    9.821717 (967) = ESPRESSO
PASS
```

+ 非20.12-py3-sdk dockerdocker中

```
root@ubuntux86:/workspace# python3 image_client.py   -m densenet_onnx -c 3 -s INCEPTION images/mug.jpg
Request 1, batch size 1
    13.915171 (504) = COFFEE MUG
    12.018230 (968) = CUP
    9.839117 (967) = ESPRESSO
PASS
```

