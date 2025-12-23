
```shell
docker pull nvcr.io/nvidia/tritonserver:20.12-py3
MODEL_PATH=$HOME/code/ML/triton-server/docs/examples/model_repository
docker run --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $MODEL_PATH:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models  --model-control-mode=explicit
# Verify Triton Is Running Correctly
curl -v localhost:8000/v2/health/ready

# Client
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:20.12-py3-sdk
./install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```


# onnx下载

[resnet/model(下载raw格式)](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet/model)


```
root@ubuntux86:# tar zxvf resnet101-v1-7.tar.gz 
resnet101-v1-7/
resnet101-v1-7/resnet101-v1-7.onnx
resnet101-v1-7/test_data_set_0/
resnet101-v1-7/test_data_set_0/input_0.pb
resnet101-v1-7/test_data_set_0/output_0.pb
root@ubuntux86:# du resnet101-v1-7.tar.gz 
162328  resnet101-v1-7.tar.gz
root@ubuntux86:# du -sh resnet101-v1-7.tar.gz 
159M    resnet101-v1-7.tar.gz
root@ubuntux86:# 
```

```
root@ubuntux86:/workspace/models/model_serving_scripts/triton# cat check_onxx_load.py 
import onnx
# Load the ONNX model
model = onnx.load("resnet101-v1-7/resnet101-v1-7.onnx")
#model = onnx.load("resnet101-v1-7.onnx")
#model = onnx.load("model_repository/resnet18/1/model.onnx")
# Check the model (raises an error if invalid)
onnx.checker.check_model(model)
root@ubuntux86:/workspace/models/model_serving_scripts/triton# python3 check_onxx_load.py 
root@ubuntux86:/workspace/models/model_serving_scripts/triton# 
```
#   check_inference_server_up

```
root@ubuntux86:/workspace/models/model_serving_scripts/triton# python3 check_inference_server_up.py 
Traceback (most recent call last):
  File "check_inference_server_up.py", line 4, in <module>
    model_metadata = client.get_model_metadata("yolo_v4")
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/http/_client.py", line 519, in get_model_metadata
    _raise_if_error(response)
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/http/_utils.py", line 69, in _raise_if_error
    raise error
tritonclient.utils.InferenceServerException: [400] Request for unknown model: 'yolo_v4' is not found
```


# bug 1 error: creating server: Internal - failed to load all models

没有config.pbtxt  densenet_labels.txt
```
root@ubuntux86:# tree -D model_repository
model_repository
└── [Dec 23 15:08]  densenet_onnx
    ├── [Dec 23 15:08]  1
    │   └── [Dec 23 15:08]  model.onnx
    ├── [Dec 23 15:08]  config.pbtxt
    └── [Dec 23 15:08]  densenet_labels.txt

2 directories, 3 files
root@ubuntux86:# 
```

即使采用--strict-model-config=false启动，客户端也会报错   
```
root@ubuntux86:/workspace2#  python3 image_client.py   -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
Traceback (most recent call last):
  File "image_client.py", line 387, in <module>
    max_batch_size, input_name, output_name, c, h, w, format, dtype = parse_model(
  File "image_client.py", line 111, in parse_model
    raise Exception(
Exception: expecting input to have 3 dimensions, model 'densenet_onnx' input has 4
root@ubuntux86:/workspace2# 
```


#  bug2 Request for unknown model: 'densenet_onnx' is not found

删掉--model-control-mode=explicit    
```
--model-control-mode=explicit
```
 

#  bug3  Protobuf parsing failed
```
python3 check_model_list.py 
[{'name': 'resnet18', 'version': '1', 'state': 'UNAVAILABLE', 'reason': 'Internal: onnx runtime error 7: Load model from /models/resnet18/1/model.onnx failed:Protobuf parsing failed.'}]
```
+ 原因是protobuf不一致(先进行python3 check_onxx_load.py ，检查有如下错误)

```
root@ubuntux86:/workspace/models/model_serving_scripts/triton# python3 check_onxx_load.py 
Traceback (most recent call last):
  File "check_onxx_load.py", line 5, in <module>
    model = onnx.load("model_repository/resnet18/1/model.onnx")
  File "/usr/local/lib/python3.8/dist-packages/onnx/__init__.py", line 212, in load_model
    model = _get_serializer(format, f).deserialize_proto(_load_bytes(f), ModelProto())
  File "/usr/local/lib/python3.8/dist-packages/onnx/serialization.py", line 118, in deserialize_proto
    decoded = typing.cast(Optional[int], proto.ParseFromString(serialized))
google.protobuf.message.DecodeError: Error parsing message with type 'onnx.ModelProto'
```

+ client
```
root@ubuntux86:/workspace/models/model_serving_scripts/triton# pip3 list | grep proto
protobuf            3.20.3
```

```
root@ubuntux86:/workspace/models/model_serving_scripts/triton#  pip list | grep onnx
onnx                1.17.0
root@ubuntux86:/workspace/models/model_serving_scripts/triton#  pip3 list | grep onnx
onnx                1.17.0
root@ubuntux86:/workspace/models/model_serving_scripts/triton#  pip3 list | grep grpcio 
grpcio              1.70.0
root@ubuntux86:/workspace/models/model_serving_scripts/triton#  pip3 list | grep grpc   
grpcio              1.70.0
root@ubuntux86:/workspace/models/model_serving_scripts/triton# 
```

+ tritonserver
```
docker run   nvcr.io/nvidia/tritonserver:20.12-py3 pip3 list | grep proto
protobuf        3.14.0
```

+  更新tritonserver:20.12-py3的protobuf

```
root@ubuntux86:# cat Dockerfile 
FROM nvcr.io/nvidia/tritonserver:20.12-py3


# setup pytorch build dependencies
RUN pip3 uninstall protobuf -y
RUN pip3 install protobuf==3.20.3 
root@ubuntux86:# 
```

` docker build -t  nvcr.io/nvidia/tritonserver:20.12-py3-proto3.20 .`

```
root@ubuntux86:/workspace/models/model_serving_scripts/triton# python3 check_model_list.py 
[{'name': 'resnet18'}]
root@ubuntux86:/workspace/models/model_serving_scripts/triton# 
```

# curl


```
curl -v localhost:8000/v2/health/ready
*   Trying 127.0.0.1:8000...
* TCP_NODELAY set
* Connected to localhost (127.0.0.1) port 8000 (#0)
> GET /v2/health/ready HTTP/1.1
> Host: localhost:8000
> User-Agent: curl/7.68.0
> Accept: */*
> 
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
< 
* Connection #0 to host localhost left intact
```
###### Check Triton server is ready.
```
> curl -v http://localhost:8000/v2/health/ready
```

###### Check model config.
```
> curl http://localhost:8000/v2/models/object_detection/config | jq
```


```
root@ubuntux86:/workspace/models# curl --location --request GET 'http://127.0.0.1:8000/v2/models/resnet18/stats'  
{"error":"requested model 'resnet18' is not available"}root@ubuntux86:/workspace/models# 
```


```
curl --location --request POST 'http://IP-Address:8000/v2/models/zst/infer' --header 'Content-Type: application/octet-stream' --header 'Accept: */*' --header 'Inference-Header-Content-Length: 275'  --data-binary '@inputv1.json' -v
```
Create an input file in json format:
```
  {
    "data" :
     [
        {
          "input__0" : [0, 145255, 175737, 525, 55583, 7, 13794, 297, 237, 8081, 53, 6524, 5425, 111, 4368, 2, 2, 3293, 7986, 83, 1672, 32628, 619, 191725, 7, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
          "input__1" : [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }
     ]
  }
 ```