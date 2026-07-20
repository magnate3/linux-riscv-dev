# Triton Inference Python Clients

## Model Installation

```bash
wget -O models/yolo_v4/1/models.onnx https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx?raw=true

or
curl -LJo yolo_v4.onnx https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/yolov4/model/yolov4.onnx

wget -O models/maskrcnn_onnx/1/models.onnx https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx?raw=true

```
models/maskrcnn_onnx模型没下载，所以移出models/maskrcnn_onnx
```
root@ubuntux86:# tree  -L 3  models/
models/
└── yolo_v4
    ├── 1
    │   └── model.onnx
    ├── config.pbtxt
    └── yolo_v4_labels.txt

2 directories, 3 files
root@ubuntux86:# 
```

## Docker Triton Inference Server


```bash
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v models:/models nvcr.io/nvidia/tritonserver:20.12-py3 tritonserver --model-repository=/models
```


## Running Clients HTTP

### MASK R-NN

This repo contains a python client. More information [here](clients/maskrcnn).
```bash
cd clients/maskrcnn
python mascrnn_client.py person_dog.jpg
```

### Yolo V4

This repo contains a python client. More information [here](clients/yolo_v4).
```bash
cd clients/yolo_v4
python image_client.py -m yolo_v4 person_dog.jpg
```

```
python3 image_client.py -m yolo_v4 person_dog.jpg
Preprocessed image shape: (416, 416, 3)
Output shape: [(1, 52, 52, 3, 85), (1, 26, 26, 3, 85), (1, 13, 13, 3, 85)]
Output shape: [(1, 52, 52, 3, 85), (1, 26, 26, 3, 85), (1, 13, 13, 3, 85)]
[ WARN:0@0.855] global loadsave.cpp:1063 imwrite_ Unsupported depth image for selected encoder is fallbacked to CV_8U.
PASS
```


## Tasks in this repo


- [x] Added Mask R-CNN R-50-FPN (more info [here](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/mask-rcnn))
- [x] Added YOLOv4 (more info [here](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4))
- [x] Added Client Mask R-CNN R-50-FPN
- [x] Added Client YOLOv4 
- [ ] Add Triton server script 
- [ ] Add GRPC Clients
