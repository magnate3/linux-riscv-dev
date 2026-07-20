# triton-sandbox

A sandbox repository for studying Triton Inference Server features.

## Tracking by Detection

All stages of the algorithm are implemented as Triton models:
 - **Detection Preprocessing** (Python backend);
 - **Detection** (onnxruntime backend);
 - **Detection Postprocessing** (Python backend);
 - **Tracking** (Python backend).

The **Detection** model is based on [YOLOv9-c](https://github.com/WongKinYiu/yolov9) detector.

The **Tracking** model is [stateful](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/architecture.html#stateful-models) which allows it to differentiate inference requests from multiple clients using provided `CORRELATION ID`.

![Demo](https://github.com/voganesyan/triton-sandbox/blob/main/demo_gif/tracking-by-detection.gif)

Navigate to `tracking-by-detection` folder.
```bash
cd tracking-by-detection
```

### Launching Triton
Launch a `tritonserver` docker container.
```bash
docker run --gpus=all -it --shm-size=256m --rm    -p8000:8000 -p8001:8001 -p8002:8002   -v ${PWD}:/workspace/ -v ${PWD}/model_repository:/models   nvcr.io/nvidia/tritonserver:24.01-py3
```

Install dependencies for our Python backend scripts.
```bash
pip install -r /models/tracking/1/ocsort/requirements.txt
```

Launch Triton.
```bash
tritonserver --model-repository=/models
```

### Running Client
Run the client application.
```bash
python client.py --video test_data/MOT17-04-SDP-raw.webm
```


YOLOv8

```

wget https://rbln-public.s3.ap-northeast-2.amazonaws.com/images/people4.jpg

```
```
root@ubuntux86:/workspace/models# python3 http_client.py --url 127.0.0.1:8000 --image ./people4.jpg 
(1, 3, 640, 640)
Invoking model 'tracking_by_detection' using HTTP...
HTTP Inference failed: [400] in ensemble 'tracking_by_detection', [request id: <id_unknown>] input byte size mismatch for input 'images' for model 'detection'. Expected 4915200, got 0
root@ubuntux86:/workspace/models# 
```

```
root@ubuntux86:/workspace/models/triton-sandbox/tracking-by-detection# python3 client2.py 
Tensor shape: torch.Size([540, 960, 3])
Tensor dtype: torch.uint8
torch.Size([540, 960, 3])
torch.Size([540, 960, 3])
torch.Size([1024, 1024, 3])
torch.Size([1024, 1024, 3])
Traceback (most recent call last):
  File "client2.py", line 234, in <module>
    main()
  File "client2.py", line 222, in main
    results = client.infer(
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/grpc/_client.py", line 1565, in infer
    raise_error_grpc(rpc_error)
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/grpc/_utils.py", line 77, in raise_error_grpc
    raise get_error_grpc(rpc_error) from None
tritonclient.utils.InferenceServerException: [StatusCode.INVALID_ARGUMENT] in ensemble 'tracking_by_detection', [request id: <id_unknown>] input byte size mismatch for input 'images' for model 'detection'. Expected 4915200, got 0
root@ubuntux86:/workspace/models/triton-sandbox/tracking-by-detection# 
```

```
root@ubuntux86:/workspace/models/triton-sandbox/tracking-by-detection# python3 client_with_pad.py 
Original image shape (HWC): (540, 960, 3)
Tensor shape (CHW): torch.Size([3, 540, 960])
Tensor dtype: torch.float32
Pixel value range: min=0.0, max=1.0
torch.Size([3, 540, 960])
torch.Size([3, 540, 960])
torch.Size([1, 3, 1024, 1024])
torch.Size([1, 1024, 1024, 3])
Traceback (most recent call last):
  File "client_with_pad.py", line 233, in <module>
    main()
  File "client_with_pad.py", line 221, in main
    results = client.infer(
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/grpc/_client.py", line 1565, in infer
    raise_error_grpc(rpc_error)
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/grpc/_utils.py", line 77, in raise_error_grpc
    raise get_error_grpc(rpc_error) from None
tritonclient.utils.InferenceServerException: [StatusCode.INVALID_ARGUMENT] in ensemble 'tracking_by_detection', [request id: <id_unknown>] input byte size mismatch for input 'images' for model 'detection'. Expected 4915200, got 0
```

## References
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [BoxMOT](https://github.com/mikel-brostrom/yolo_tracking)
