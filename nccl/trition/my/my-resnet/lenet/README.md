
+ server
```
root@ubuntux86:# docker run  -it --shm-size=256m --rm    -p8000:8000 -p8001:8001 -p8002:8002   -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models   nvcr.io/nvidia/tritonserver:20.12-py3 bash

```

```
tritonserver --model-repository=/models --strict-model-config=false

```

+ client(复用my-resnet)
```
root@ubuntux86:/workspace/models/lenet/my-resnet# python3 http_client.py 
Files already downloaded and verified
torch.Size([1, 1, 32, 32])
torch.Size([1, 1, 32, 32])
torch.Size([1, 3, 32, 32])
--- 0.01755666732788086 seconds ---
[<tritonclient.http._requested_output.InferRequestedOutput object at 0x7fc5f7df8d30>]
max of predict [2] 
['bird'] 
 [[ 1.752 -4.877  3.465  1.182  2.738  0.495  1.519 -1.212 -0.235 -4.478]]
```