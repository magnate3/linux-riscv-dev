

+  tritonserver
```
 sudo docker run  --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g" --privileged  -v /pytorch:/pytorch -u 0  --entrypoint bash  --name=fedml fedml-x86:v2 

```
 
```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g" --privileged   -u 0  -p 8000:8000 -p 8001:8001 -p 8002:8002 --name=triton \
-v $(pwd)/models:/models \
nvcr.io/nvidia/tritonserver:24.05-py3 tritonserver --model-repository=/models
```


```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/tritonserver:24.05-py3 bash
```

```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0  nvcr.io/nvidia/tritonserver:24.05-py3 \
perf_analyzer -m resnet18_pytorch --concurrency-range 1:16 -u localhost:8001
```

+ triton client

```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/tritonserver:24.05-py3-sdk bash
```


+ pytorch



```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/pytorch:24.05-py3 bash

```
