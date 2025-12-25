


```
root@ubuntux86:/workspace/models/resnet/my-resnet# python3 http_client.py 
Files already downloaded and verified
torch.Size([1, 1, 32, 32])
torch.Size([1, 1, 32, 32])
torch.Size([1, 3, 32, 32])
--- 0.013181447982788086 seconds ---
[<tritonclient.http._requested_output.InferRequestedOutput object at 0x7f21ce4a9ca0>]
max of predict [5] 
['dog'] 
 [[ 1.952 -5.999  3.544  3.275  0.773  4.712  0.89  -2.074 -2.082 -4.474]]
```


+  server

```
nvcr.io/nvidia/tritonserver                                          24.07-py3
```

```
docker run -it --shm-size=2G  --rm    -p8000:8000 -p8001:8001 -p8002:8002   -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models   --name triton-ensemble-model triton-ensemble-model:v1  /bin/bash
```