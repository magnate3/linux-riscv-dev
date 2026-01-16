

```
pip install torch-tensorrt。
```

```
docker pull nvcr.io/nvidia/pytorch:24.05-py3
docker pull nvcr.io/nvidia/tensorrt:21.11-py3
docker pull nvcr.io/nvidia/tensorrt:24.05-py3
docker pull nvcr.io/nvidia/tritonserver:24.05-py3
```

测试 tritonserver 模型服务的 QPS 通常有两种方法，一种是使用 perf_analyzer 来测试，另一种是通过 model-analyzer 来获取更为详细的模型服务启动的参数，使得模型的 QPS 达到最大。     
[使用perf_analyzer和model-analyzer测试tritonserver的模型性能超详细完整版](https://blog.csdn.net/sinat_29957455/article/details/132583942)    

#  tensorrt

```
dpkg -l | grep -i tensorrt-dev
ii  tensorrt-dev                    10.0.1.6-1+cuda12.4                         amd64        Meta package for TensorRT development libraries
```

```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0  --name=tensorrt \
-v $(pwd)/models:/models \
nvcr.io/nvidia/tensorrt:24.05-py3  bash
```
```
/workspace/tensorrt/bin# ./trtexec --loadEngine=/models/resnet50_trt/1/model.plan    
```

# resnet18


```
root@ubuntu:/pytorch/triton# python3 export_model.py 
Downloading: "https://download.pytorch.org/models/resnet18-f37072fd.pth" to /root/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 44.7M/44.7M [00:01<00:00, 41.4MB/s]
models/resnet18_pytorch/1/model.pt
root@ubuntu:/pytorch/triton# python3 export_model.py 
models/resnet18_pytorch/1/model.pt
root@ubuntu:/pytorch/triton# 
```

# server

```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0  -p 8000:8000 -p 8001:8001 -p 8002:8002 --name=triton \
-v $(pwd)/models:/models \
nvcr.io/nvidia/tritonserver:24.05-py3 tritonserver --model-repository=/models
```


```
I0115 02:14:51.992766 1 grpc_server.cc:2463] "Started GRPCInferenceService at 0.0.0.0:8001"
I0115 02:14:51.993006 1 http_server.cc:4692] "Started HTTPService at 0.0.0.0:8000"
I0115 02:14:52.034027 1 http_server.cc:362] "Started Metrics Service at 0.0.0.0:8002"
```


# client



```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/tritonserver:24.05-py3-sdk bash
```

采用grpc报错   
```
perf_analyzer -m resnet18_pytorch --concurrency-range 1:16 -u localhost:8001
error: failed to get model metadata: HTTP client failed: Unsupported protocol
```
+ 换成http    

```
perf_analyzer -m resnet18_pytorch --concurrency-range 1:16 -u localhost:8000
```

```
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 879.896 infer/sec, latency 1108 usec
Concurrency: 2, throughput: 1333.64 infer/sec, latency 1470 usec
Concurrency: 3, throughput: 1325.84 infer/sec, latency 2231 usec
Concurrency: 4, throughput: 1320.5 infer/sec, latency 2995 usec
Concurrency: 5, throughput: 1320.94 infer/sec, latency 3751 usec
Concurrency: 6, throughput: 1320.32 infer/sec, latency 4509 usec
Concurrency: 7, throughput: 1318.32 infer/sec, latency 5272 usec
Concurrency: 8, throughput: 1316.55 infer/sec, latency 6039 usec
Concurrency: 9, throughput: 1319.22 infer/sec, latency 6784 usec
Concurrency: 10, throughput: 1317.85 infer/sec, latency 7549 usec
Concurrency: 11, throughput: 1315.51 infer/sec, latency 8320 usec
Concurrency: 12, throughput: 1313.88 infer/sec, latency 9090 usec
Concurrency: 13, throughput: 1314.99 infer/sec, latency 9844 usec
Concurrency: 14, throughput: 1312.72 infer/sec, latency 10621 usec
Concurrency: 15, throughput: 1311.58 infer/sec, latency 11392 usec
Concurrency: 16, throughput: 1311.23 infer/sec, latency 12158 usec
```


# tensorrt



```
 UNAVAILABLE: Internal: unable to load plan file to auto complete config: /models/resnet18_trt/1/model.plan
```

./trtexec --loadEngine=/models/resnet18_trt/1/model.plan     
```
[01/15/2026-07:20:09] [E] Error[1]: [runtime.cpp::parsePlan::455] Error Code 1: Serialization (Serialization assertion plan->header.magicTag == rt::kPLAN_MAGIC_TAG failed.Trying to load an engine created with incompatible serialization version. Check that the engine was not created using safety runtime, same OS was used and version compatibility parameters were set accordingly.)
[01/15/2026-07:20:09] [E] Engine deserialization failed
[01/15/2026-07:20:09] [E] Got invalid engine!
[01/15/2026-07:20:09] [E] Inference set up failed
&&&& FAILED TensorRT.trtexec [TensorRT v100001] # ./trtexec --loadEngine=/models/resnet18_trt/1/model.plan
```