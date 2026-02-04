# benchmark

##  make

```
/pytorch/ncnn/my-benchmark# cmake  -DNCNN_DIR=/pytorch/ncnn/build/install -DCMAKE_BUILD_TYPE=Release . -B build
```

## run

```
/pytorch/ncnn/my-benchmark/build# ./benchncnn  yolov5
task type = 1
loop_count = 4
num_threads = 8
powersave = 2
gpu_device = -1
cooling_down = 1
```

#  Evaluation


```
 cmake  -DNCNN_INSTALL_DIR=/pytorch/ncnn/build/install -DNCNN_SRC_DIR=/pytorch/ncnn/src -DCMAKE_BUILD_TYPE=Release . -B build
```


```
/pytorch/ncnn/my-benchmark/Evaluation/build# ./evaluate  resnet-int8 /pytorch/ncnn/build/imagenet-sample-images/
```

#  COCOEvaluator ncnn