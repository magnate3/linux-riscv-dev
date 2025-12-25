执行mkdir -p model_repository/yolov8n_ensemble/1
```
root@ubuntux86:# ls model_repository/yolov8n_ensemble/
config.pbtxt
root@ubuntux86:# mkdir -p model_repository/yolov8n_ensemble/1
root@ubuntux86:# 
```  
+ docker images
```
nvcr.io/nvidia/tritonserver                                          24.07-py3
```

# run  
```
root@ubuntux86:/workspace/models/yolov8# python3 http_client.py --url 127.0.0.1:8000 --image ./people4.jpg 
Invoking model 'yolov8n_ensemble' using HTTP...
 infer over and analyze result 

reuslt <tritonclient.http._infer_result.InferResult object at 0x7f6d635b3460> 

Traceback (most recent call last):
  File "http_client.py", line 145, in <module>
    output_boxes = results.as_numpy('final_boxes')
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/http/_infer_result.py", line 208, in as_numpy
    np_array = np_array.reshape(output["shape"])
ValueError: cannot reshape array of size 0 into shape (1,7)
```

```
root@ubuntux86:/workspace/models/yolov8# python3 grpc_client.py --url 127.0.0.1:8001 --image ./people4.jpg 
Invoking model 'yolov8n_ensemble' using gRPC...
Traceback (most recent call last):
  File "grpc_client.py", line 140, in <module>
    output_boxes = results.as_numpy('final_boxes')
  File "/usr/local/lib/python3.8/dist-packages/tritonclient/grpc/_infer_result.py", line 93, in as_numpy
    np_array = np_array.reshape(shape)
ValueError: cannot reshape array of size 0 into shape (1,7)
root@ubuntux86:/workspace/models/yolov8# python3 grpc_client.py --url 127.0.0.1:9000 --image ./people4.jpg 
Invoking model 'yolov8n_ensemble' using gRPC...
gRPC Inference failed: [StatusCode.UNAVAILABLE] failed to connect to all addresses; last error: UNKNOWN: ipv4:127.0.0.1:9000: Failed to connect to remote host: connect: Connection refused (111)
root@ubuntux86:/workspace/models/yolov8# 
```

##   换个模型

[yolov8n.onnx](https://huggingface.co/SpotLab/YOLOv8Detection/blob/3005c6751fb19cdeb6b10c066185908faf66a097/yolov8n.onnx)

```
root@ubuntux86:/workspace/models/yolov8# python3 grpc_client.py --url 127.0.0.1:8001 --image ./people4.jpg 
Invoking model 'yolov8n_ensemble' using gRPC...
No objects detected.
root@ubuntux86:/workspace/models/yolov8# python3 http_client.py --url 127.0.0.1:8000 --image ./people4.jpg 
Invoking model 'yolov8n_ensemble' using HTTP...
 infer over and analyze result 

reuslt <tritonclient.http._infer_result.InferResult object at 0x7f8df4f9e460> 

No objects detected.
root@ubuntux86:/workspace/models/yolov8# 
```