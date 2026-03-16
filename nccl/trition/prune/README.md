

[推荐一款自动剪枝工具Torch-Pruning](https://zhuanlan.zhihu.com/p/694121198)   
`torchslim` `mobile-yolov5-pruning-distillation`   


```
ls ~/.cache/torch/hub/checkpoints/
resnet50-0676ba61.pth
```

#  docker


```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/pytorch:24.05-py3 bash
PYTHONPATH=$PYTHONPATH:/pytorch/prune/tinyimagenet
```
安装torch-pruning  
[https://github.com/VainF/Torch-Pruning](https://github.com/VainF/Torch-Pruning)
```
git clone https://github.com/VainF/Torch-Pruning.git
cd Torch-Pruning && pip install -e .
```
环境变量（可以忽略）   

```
export PYTHONPATH=$PYTHONPATH:/pytorch/prune/tinyimagene
```

# test1(有bug)


```
剪枝后
layer pnnx.Input not exists or registered
network graph not ready
```

+ python3 torch-pruning.py

+ py2onnx.py  
```
root@ubuntu:/pytorch/prune# ln -sf /pytorch/ResNet18_Cifar10_95.46/dataset dataset 
root@ubuntu:/pytorch/prune# python3 py2onnx.py
```

```
/pytorch/prune# python3 py2onnx.py 
data shape : torch.Size([128, 3, 32, 32])
data label: tensor([3, 8, 8])
torch.Size([128, 3, 32, 32])
[name: "output"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_param: "batch"
      }
      dim {
        dim_param: "anchors"
      }
    }
  }
}
]
```


+ pnnx


```
pip3 install pnnx -i https://pypi.tuna.tsinghua.edu.cn/simple
```


```
root@ubuntu:/pytorch/prune/ncnnx# pnnx  ../models/res50_model.onnx 

root@ubuntu:/pytorch/prune/ncnnx# ls ../models/
res50_model.ncnn.bin    res50_model.onnx      res50_model.pnnx.onnx   res50_model.pnnxsim.onnx  res50_model_pnnx.py
res50_model.ncnn.param  res50_model.pnnx.bin  res50_model.pnnx.param  res50_model_ncnn.py
```


+  benchncnn
```
/pytorch/ncnn/build/benchmark/benchncnn 4 8 0 0 1  param=./models/res50_model.pnnx.param  shape=[227,227,3]
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = 0
cooling_down = 1
layer pnnx.Input not exists or registered
network graph not ready
./models/res50_model.pnnx.param  min =    0.00  max =    0.00  avg =    0.00
```


#  成功方法

剪枝

```
python3 demo_prune.py 
mv model.onnx  ncnnx/
pnnx  ./ncnnx/model.onnx 
```
+ 剪枝前

```
/pytorch/ncnn/build/benchmark/benchncnn 4 8 0 0 1  param=/pytorch/ncnn/build/onnx2ncnn/model.ncnn.param   shape=[227,227,3]
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = 0
cooling_down = 1
/pytorch/ncnn/build/onnx2ncnn/model.ncnn.param  min =   14.87  max =   15.47  avg =   15.06
```

+ 剪枝后

```
/pytorch/ncnn/build/benchmark/benchncnn 4 8 0 0 1  param=./ncnnx/model.ncnn.param  shape=[227,227,3]
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = 0
cooling_down = 1
./ncnnx/model.ncnn.param  min =   13.01  max =   13.23  avg =   13.13
```