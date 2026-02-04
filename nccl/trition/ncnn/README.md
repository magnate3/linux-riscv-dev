

```
sudo  docker run --rm --net=host    --gpus=all -it    -e UID=root    --ipc host --shm-size="32g"  --privileged   -u 0   -v /pytorch:/pytorch  nvcr.io/nvidia/pytorch:24.05-py3 bash
```

# 编译ncnn转换工具

通过 `git clone https://github.com/Tencent/ncnn.git` 将ncnn的仓库拉取到本地，进行编译

安装编译环境的依赖

```bash
sudo apt update
sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev libopencv-dev
```

> ## 编译ncnn需要使用到 Vulkan 后端(没有使用)
要使用 Vulkan 后端，请安装 Vulkan 头文件、一个 vulkan 驱动程序加载器、GLSL 到 SPIR-V 编译器和 vulkaninfo 工具。或者从<https://vulkan.lunarg.com/sdk/home>下载并安装完整的 Vulkan SDK（大约 200MB；它包含所有头文件、文档和预构建的加载程序，以及一些额外的工具和所有源代码）

```
 apt install  libvulkan-dev vulkan-utils 
```

```bash
wget https://sdk.lunarg.com/sdk/download/1.2.182.0/linux/vulkansdk-linux-x86_64-1.2.182.0.tar.gz
tar xvf vulkansdk-linux-x86_64-1.2.182.0.tar.gz
export VULKAN_SDK=$(pwd)/1.2.182.0/x86_64
```

> ##  libutf8_range.a

```
 git clone https://github.com/protocolbuffers/utf8_range.git \
    && cd utf8_range \
    && cmake -Dutf8_range_ENABLE_TESTS=off -DCMAKE_INSTALL_PREFIX=/usr . -B build \
    && cd build \
    && make -j \
    && make install -j \
    && cd ../.. && rm -rf utf8_range \
    && ln -s /usr/lib/x86_64-linux-gnu/libutf8_validity.a /usr/lib/libutf8_validity.a \
    && ln -s /usr/lib/x86_64-linux-gnu/libutf8_range.a /usr/lib/libutf8_range.a
```

拉取ncnn的子仓库

```bash
cd ncnn
git submodule update --init
```

开始编译ncnn,-DNCNN_VULKAN=OFF    
```bash
mkdir -p build
cd build
 cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF  -DNCNN_BUILD_TOOLS=OFF  -DNCNN_SHARED_LIB=OFF -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
make -j$(nproc)
```


```
/pytorch/ncnn/build#  examples/yolov4 -h
fopen yolov4-tiny-opt.param failed
```


> ## NCNN_BUILD_TOOLS=ON (不需要)
但是tools/CMakeLists.txt注释add_subdirectory(caffe)    
```
 cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF  -DNCNN_BUILD_TOOLS=ON  -DNCNN_SHARED_LIB=OFF -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON ..
```

 
#  onnx2ncnn

```
pip3 install pnnx
```

```
 python onnx_exporter.py --save model.onnx 
```


> ##  resnet50
```
 pnnx  model.onnx 
 pnnx  model.onnx 
pnnxparam = model.pnnx.param
pnnxbin = model.pnnx.bin
pnnxpy = model_pnnx.py
pnnxonnx = model.pnnx.onnx
ncnnparam = model.ncnn.param
ncnnbin = model.ncnn.bin
ncnnpy = model_ncnn.py
fp16 = 1
optlevel = 2
device = cpu
inputshape = 
inputshape2 = 
customop = 
moduleop = 
get inputshape from traced inputs
inputshape = [1,3,128,128]f32
inputshape2 = [1,3,512,512]f32
############# pass_level0 onnx 
inline_containers ...                 0.00ms
eliminate_noop ...                    0.08ms
fold_constants ...                    0.05ms
canonicalize ...                      0.10ms
shape_inference ...               2026-01-29 14:19:38.171314080 [W:onnxruntime:pnnx, cpuid_info.cc:91 LogEarlyWarning] Unknown CPU vendor. cpuinfo_vendor value: 0
  403.37ms
fold_constants_dynamic_shape ...      0.11ms
inline_if_graph ...                   0.01ms
fuse_constant_as_attribute ...        0.10ms
eliminate_noop_with_shape ...         0.06ms
┌───────────────────┬──────────┬──────────┐
│                   │ orig     │ opt      │
├───────────────────┼──────────┼──────────┤
│ node              │ 122      │ 122      │
│ initializer       │ 108      │ 108      │
│ functions         │ 0        │ 0        │
├───────────────────┼──────────┼──────────┤
│ nn module op      │ 0        │ 0        │
│ custom module op  │ 0        │ 0        │
│ aten op           │ 0        │ 0        │
│ prims op          │ 0        │ 0        │
│ onnx native op    │ 122      │ 122      │
├───────────────────┼──────────┼──────────┤
│ Add               │ 16       │ 16       │
│ Conv              │ 53       │ 53       │
│ Flatten           │ 1        │ 1        │
│ Gemm              │ 1        │ 1        │
│ GlobalAveragePool │ 1        │ 1        │
│ MaxPool           │ 1        │ 1        │
│ Relu              │ 49       │ 49       │
└───────────────────┴──────────┴──────────┘
############# pass_level1 onnx
############# pass_level2
############# pass_level3
############# pass_level4
############# pass_level5
############# pass_ncnn
insert_reshape_global_pooling_forward torch.flatten_0 120
 
```

```
  ls
model.ncnn.bin  model.ncnn.param  model.onnx  model.pnnx.bin  model.pnnx.onnx  model.pnnx.param  model.pnnxsim.onnx  model_ncnn.py  model_pnnx.py
```


```
cat ../../model.ncnn.param 
7767517
107 123
Input                    in0                      0 1 in0
Convolution              convrelu_0               1 1 in0 1 0=64 1=7 11=7 12=1 13=2 14=3 2=1 3=2 4=3 5=1 6=9408 9=1
Pooling                  maxpool2d_1              1 1 1 2 0=0 1=3 11=3 12=2 13=1 2=2 3=1 5=1
Split                    splitncnn_0              1 2 2 3 4
Convolution              convrelu_1               1 1 4 5 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=4096 9=1
Convolution              convrelu_2               1 1 5 6 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864 9=1
Convolution              conv_54                  1 1 6 7 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
Convolution              conv_55                  1 1 3 8 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
BinaryOp                 add_0                    2 1 7 8 9 0=0
ReLU                     relu_5                   1 1 9 10
Split                    splitncnn_1              1 2 10 11 12
Convolution              convrelu_3               1 1 12 13 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384 9=1
Convolution              convrelu_4               1 1 13 14 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864 9=1
Convolution              conv_58                  1 1 14 15 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
BinaryOp                 add_1                    2 1 15 11 16 0=0
ReLU                     relu_8                   1 1 16 17
Split                    splitncnn_2              1 2 17 18 19
Convolution              convrelu_5               1 1 19 20 0=64 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384 9=1
Convolution              convrelu_6               1 1 20 21 0=64 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=36864 9=1
Convolution              conv_61                  1 1 21 22 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=16384
BinaryOp                 add_2                    2 1 22 18 23 0=0
ReLU                     relu_11                  1 1 23 24
Split                    splitncnn_3              1 2 24 25 26
Convolution              convrelu_7               1 1 26 27 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=32768 9=1
Convolution              convrelu_8               1 1 27 28 0=128 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=147456 9=1
Convolution              conv_64                  1 1 28 29 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
Convolution              conv_65                  1 1 25 30 0=512 1=1 11=1 12=1 13=2 14=0 2=1 3=2 4=0 5=1 6=131072
BinaryOp                 add_3                    2 1 29 30 31 0=0
ReLU                     relu_14                  1 1 31 32
Split                    splitncnn_4              1 2 32 33 34
Convolution              convrelu_9               1 1 34 35 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536 9=1
Convolution              convrelu_10              1 1 35 36 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456 9=1
Convolution              conv_68                  1 1 36 37 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
BinaryOp                 add_4                    2 1 37 33 38 0=0
ReLU                     relu_17                  1 1 38 39
Split                    splitncnn_5              1 2 39 40 41
Convolution              convrelu_11              1 1 41 42 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536 9=1
Convolution              convrelu_12              1 1 42 43 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456 9=1
Convolution              conv_71                  1 1 43 44 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
BinaryOp                 add_5                    2 1 44 40 45 0=0
ReLU                     relu_20                  1 1 45 46
Split                    splitncnn_6              1 2 46 47 48
Convolution              convrelu_13              1 1 48 49 0=128 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536 9=1
Convolution              convrelu_14              1 1 49 50 0=128 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=147456 9=1
Convolution              conv_74                  1 1 50 51 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=65536
BinaryOp                 add_6                    2 1 51 47 52 0=0
ReLU                     relu_23                  1 1 52 53
Split                    splitncnn_7              1 2 53 54 55
Convolution              convrelu_15              1 1 55 56 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=131072 9=1
Convolution              convrelu_16              1 1 56 57 0=256 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=589824 9=1
Convolution              conv_77                  1 1 57 58 0=1024 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
Convolution              conv_78                  1 1 54 59 0=1024 1=1 11=1 12=1 13=2 14=0 2=1 3=2 4=0 5=1 6=524288
BinaryOp                 add_7                    2 1 58 59 60 0=0
ReLU                     relu_26                  1 1 60 61
Split                    splitncnn_8              1 2 61 62 63
Convolution              convrelu_17              1 1 63 64 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144 9=1
Convolution              convrelu_18              1 1 64 65 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824 9=1
Convolution              conv_81                  1 1 65 66 0=1024 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
BinaryOp                 add_8                    2 1 66 62 67 0=0
ReLU                     relu_29                  1 1 67 68
Split                    splitncnn_9              1 2 68 69 70
Convolution              convrelu_19              1 1 70 71 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144 9=1
Convolution              convrelu_20              1 1 71 72 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824 9=1
Convolution              conv_84                  1 1 72 73 0=1024 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
BinaryOp                 add_9                    2 1 73 69 74 0=0
ReLU                     relu_32                  1 1 74 75
Split                    splitncnn_10             1 2 75 76 77
Convolution              convrelu_21              1 1 77 78 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144 9=1
Convolution              convrelu_22              1 1 78 79 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824 9=1
Convolution              conv_87                  1 1 79 80 0=1024 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
BinaryOp                 add_10                   2 1 80 76 81 0=0
ReLU                     relu_35                  1 1 81 82
Split                    splitncnn_11             1 2 82 83 84
Convolution              convrelu_23              1 1 84 85 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144 9=1
Convolution              convrelu_24              1 1 85 86 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824 9=1
Convolution              conv_90                  1 1 86 87 0=1024 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
BinaryOp                 add_11                   2 1 87 83 88 0=0
ReLU                     relu_38                  1 1 88 89
Split                    splitncnn_12             1 2 89 90 91
Convolution              convrelu_25              1 1 91 92 0=256 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144 9=1
Convolution              convrelu_26              1 1 92 93 0=256 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=589824 9=1
Convolution              conv_93                  1 1 93 94 0=1024 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=262144
BinaryOp                 add_12                   2 1 94 90 95 0=0
ReLU                     relu_41                  1 1 95 96
Split                    splitncnn_13             1 2 96 97 98
Convolution              convrelu_27              1 1 98 99 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=524288 9=1
Convolution              convrelu_28              1 1 99 100 0=512 1=3 11=3 12=1 13=2 14=1 2=1 3=2 4=1 5=1 6=2359296 9=1
Convolution              conv_96                  1 1 100 101 0=2048 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=1048576
Convolution              conv_97                  1 1 97 102 0=2048 1=1 11=1 12=1 13=2 14=0 2=1 3=2 4=0 5=1 6=2097152
BinaryOp                 add_13                   2 1 101 102 103 0=0
ReLU                     relu_44                  1 1 103 104
Split                    splitncnn_14             1 2 104 105 106
Convolution              convrelu_29              1 1 106 107 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=1048576 9=1
Convolution              convrelu_30              1 1 107 108 0=512 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=2359296 9=1
Convolution              conv_100                 1 1 108 109 0=2048 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=1048576
BinaryOp                 add_14                   2 1 109 105 110 0=0
ReLU                     relu_47                  1 1 110 111
Split                    splitncnn_15             1 2 111 112 113
Convolution              convrelu_31              1 1 113 114 0=512 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=1048576 9=1
Convolution              convrelu_32              1 1 114 115 0=512 1=3 11=3 12=1 13=1 14=1 2=1 3=1 4=1 5=1 6=2359296 9=1
Convolution              conv_103                 1 1 115 116 0=2048 1=1 11=1 12=1 13=1 14=0 2=1 3=1 4=0 5=1 6=1048576
BinaryOp                 add_15                   2 1 116 112 117 0=0
ReLU                     relu_50                  1 1 117 118
Pooling                  gap_0                    1 1 118 119 0=1 4=1
Reshape                  reshape_105              1 1 119 120 0=1 1=1 2=-1
Flatten                  flatten_106              1 1 120 121
InnerProduct             linear_104               1 1 121 out0 0=1000 1=1 2=2048000
```
in0和out0  两个参数    


> ## 量化前benchncnn test
```
./benchmark/benchncnn 4 8 0 -1 1 param=onnx2ncnn/model.ncnn.param   shape=[227,227,3]
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 1
onnx2ncnn/model.ncnn.param  min =   16.62  max =   21.99  avg =   18.83
```

> ## int8 量化后 benchncnn test

[量化参考quantized-int8-inference.md](https://github.com/Tencent/ncnn/blob/bdac11b74d769c7f399085c892dedbec78b8d278/docs/how-to-use-and-FAQ/quantized-int8-inference.md)  


[鲲鹏920环境，yolov8n模型int8量化速度比默认的fp16慢了50%](https://github.com/Tencent/ncnn/issues/5443)    

+ 下载图片   
```
git clone https://github.com/nihui/imagenet-sample-images.git
```

+  imagelist.txt
```
find imagenet-sample-images/ -type f > imagelist.txt
```


+ 生成resnet50.table     
```
./tools/quantize/ncnn2table   onnx2ncnn/model.ncnn.param  onnx2ncnn/model.ncnn.bin imagelist.txt resnet50.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
```


```
./tools/quantize/ncnn2table   onnx2ncnn/model.ncnn.param  onnx2ncnn/model.ncnn.bin imagelist.txt resnet50.table mean=[104,117,123] norm=[0.017,0.017,0.017] shape=[224,224,3] pixel=BGR thread=8 method=kl
mean = [104.000000,117.000000,123.000000]
norm = [0.017000,0.017000,0.017000]
shape = [224,224,3]
pixel = BGR
thread = 8
method = kl
---------------------------------------
count the absmax 0.00% [ 0 / 1026 ]
count the absmax 9.75% [ 100 / 1026 ]
count the absmax 19.49% [ 200 / 1026 ]
count the absmax 29.24% [ 300 / 1026 ]
count the absmax 38.99% [ 400 / 1026 ]
count the absmax 48.73% [ 500 / 1026 ]
count the absmax 58.48% [ 600 / 1026 ]
count the absmax 68.23% [ 700 / 1026 ]
count the absmax 77.97% [ 800 / 1026 ]
count the absmax 87.72% [ 900 / 1026 ]
count the absmax 97.47% [ 1000 / 1026 ]
build histogram 0.00% [ 0 / 1026 ]
build histogram 9.75% [ 100 / 1026 ]
build histogram 19.49% [ 200 / 1026 ]
build histogram 29.24% [ 300 / 1026 ]
build histogram 38.99% [ 400 / 1026 ]
build histogram 48.73% [ 500 / 1026 ]
build histogram 58.48% [ 600 / 1026 ]
build histogram 68.23% [ 700 / 1026 ]
build histogram 77.97% [ 800 / 1026 ]
build histogram 87.72% [ 900 / 1026 ]
build histogram 97.47% [ 1000 / 1026 ]
```

+  生成 resnet-int8.param  resnet-int8.bin
```
./tools/quantize/ncnn2int8 onnx2ncnn/model.ncnn.param  onnx2ncnn/model.ncnn.bin  int8-quant/resnet-int8.param int8-quant/resnet-int8.bin  resnet50.table
```

```
./tools/quantize/ncnn2int8 onnx2ncnn/model.ncnn.param  onnx2ncnn/model.ncnn.bin  int8-quant/resnet-int8.param int8-quant/resnet-int8.bin  resnet50.table
quantize_convolution convrelu_0
quantize_convolution convrelu_1
quantize_convolution convrelu_2
quantize_convolution conv_54
quantize_convolution conv_55
quantize_convolution convrelu_3
quantize_convolution convrelu_4
quantize_convolution conv_58
quantize_convolution convrelu_5
quantize_convolution convrelu_6
quantize_convolution conv_61
quantize_convolution convrelu_7
quantize_convolution convrelu_8
quantize_convolution conv_64
quantize_convolution conv_65
quantize_convolution convrelu_9
quantize_convolution convrelu_10
quantize_convolution conv_68
quantize_convolution convrelu_11
quantize_convolution convrelu_12
quantize_convolution conv_71
quantize_convolution convrelu_13
```

```
./benchmark/benchncnn 4 8 0 -1 1 param=int8-quant/resnet-int8.param   shape=[227,227,3]
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 1
int8-quant/resnet-int8.param  min =    7.89  max =   15.51  avg =   10.63
```

# 推理test

```
 python onnx_exporter.py --save model.onnx 
```


```
test# cmake . -B build
CMake Deprecation Warning at CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

```

```
onnx2ncnn/test/build# ./resnet ../../images/horse.png 
output size: 1000
detection time: 88 ms
228 = 5.034719
181 = 4.681878
190 = 4.639017
```


# benchmark

[perf-tools/ncnn-testsuite.sh](https://github.com/YingkunZhou/EdgeTransformerBench/blob/main/perf-tools/ncnn-testsuite.sh)

> ## cpu
```
./benchmark/benchncnn 4 8 0 -1 1
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = -1
cooling_down = 1
          squeezenet  min =    2.32  max =    2.40  avg =    2.36
     squeezenet_int8  min =    1.79  max =    1.81  avg =    1.80
           mobilenet  min =    2.83  max =    3.33  avg =    2.99
      mobilenet_int8  min =    1.64  max =    1.68  avg =    1.65
        mobilenet_v2  min =    4.33  max =    4.48  avg =    4.40
        mobilenet_v3  min =    2.36  max =    2.47  avg =    2.39
          shufflenet  min =    2.74  max =    2.76  avg =    2.75
       shufflenet_v2  min =    2.47  max =    2.47  avg =    2.47
             mnasnet  min =    2.96  max =    3.05  avg =    3.00
     proxylessnasnet  min =    3.53  max =    3.59  avg =    3.57
     efficientnet_b0  min =    5.54  max =    5.56  avg =    5.55
   efficientnetv2_b0  min =    5.73  max =    5.81  avg =    5.76
        regnety_400m  min =    6.66  max =    6.79  avg =    6.70
           blazeface  min =    0.95  max =    0.99  avg =    0.97
           googlenet  min =    7.78  max =    8.02  avg =    7.87
      googlenet_int8  min =    5.25  max =    5.32  avg =    5.30
            resnet18  min =    5.08  max =    5.64  avg =    5.25
       resnet18_int8  min =    3.82  max =    3.98  avg =    3.87
             alexnet  min =    5.15  max =    5.30  avg =    5.22
               vgg16  min =   29.00  max =   29.79  avg =   29.41
          vgg16_int8  min =   17.94  max =   18.02  avg =   17.98
            resnet50  min =   13.39  max =   13.81  avg =   13.51
       resnet50_int8  min =    7.20  max =    7.31  avg =    7.23
      squeezenet_ssd  min =    6.38  max =    6.47  avg =    6.41
 squeezenet_ssd_int8  min =    5.21  max =    5.26  avg =    5.23
       mobilenet_ssd  min =    5.54  max =    5.64  avg =    5.58
  mobilenet_ssd_int8  min =    3.33  max =    3.37  avg =    3.35
      mobilenet_yolo  min =   14.47  max =   14.75  avg =   14.64
  mobilenetv2_yolov3  min =   10.36  max =   10.52  avg =   10.43
         yolov4-tiny  min =   13.59  max =   13.80  avg =   13.66
           nanodet_m  min =    5.09  max =    5.18  avg =    5.11
    yolo-fastest-1.1  min =    2.81  max =    2.82  avg =    2.82
      yolo-fastestv2  min =    2.85  max =    2.88  avg =    2.87
  vision_transformer  min =   80.05  max =   86.92  avg =   83.11
          FastestDet  min =    3.01  max =    3.03  avg =    3.03
```




# gpu


```
./benchmark/benchncnn 4 8 0 0 1 param=onnx2ncnn/model.ncnn.param   shape=[227,227,3]
loop_count = 4
num_threads = 8
powersave = 0
gpu_device = 0
cooling_down = 1
onnx2ncnn/model.ncnn.param  min =   16.49  max =   16.56  avg =   16.52
```

# python



```
/pytorch/ncnn# git submodule init && git submodule update
/pytorch/ncnn# python setup.py install
```

日志
```
byte-compiling /usr/local/lib/python3.10/dist-packages/ncnn/utils/download.py to download.cpython-310.pyc
byte-compiling /usr/local/lib/python3.10/dist-packages/ncnn/utils/__init__.py to __init__.cpython-310.pyc
byte-compiling /usr/local/lib/python3.10/dist-packages/ncnn/utils/visual.py to visual.cpython-310.pyc
```

## 测试




```
/pytorch/ncnn/python# python3 tests/benchmark.py 
loop_count = 4
num_threads = 32
powersave = 0
gpu_device = -1
cooling_down = True
          squeezenet  min =    2.73  max =    2.74  avg =    2.73

```


#   ultralytics


```
pip install ultralytics
```


# cifar10 测试



```
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
```

#  lite.ai.toolkit




[lite.ai.toolkit](https://github.com/xlite-dev/lite.ai.toolkit/blob/7ce0c9810e977650c9f85a4dd1d391ebb975d04a/README.md)


#  imagenet1k 

    - [ImageNet1k](https://www.image-net.org)
    > <img src="/readme_supply/imagenet1k_dataset.png" width=30% height=30%></img>
    - [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
    > <img src="/readme_supply/cifar10_dataset.png" width=20% height=20%></img>
```
curl -o plane.jpg http://images.cocodataset.org/test2017/000000030207.jpg
curl -o food.jpg http://images.cocodataset.org/test2017/000000228503.jpg
curl -o sport.jpg http://images.cocodataset.org/test2017/000000133861.jpg
curl -o dog.jpg https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
```

+ ncnn 自带的imagenet-sample-images
```
find ./ -name n02395406_hog.JPEG
./imagenet-sample-images/n02395406_hog.JPEG
```

> ## 测试 n02395406_hog.JPEG

```
/pytorch/ncnn/build/onnx2ncnn/test/build# ./resnet  ../n02395406_hog.JPEG 
output size: 1000
detection time: 102 ms
341 = 5.296703
342 = 4.192142
296 = 3.810430
```
最大是

```
341 = 5.296703
```

和get_label_data_from_filename获取的结果一致 code, label, name =('n02395406', 341, 'hog')

+  运行get_label_data_from_filename函数

```
def get_label_data_from_filename(filename: str, dataset_name: str) -> Tuple[str, int, str]:
    """
    Return name from filename, or (name, label) regarding a name to label dictionnary

    Args:
        filename (str): filename
        datapath (str): path to the dataset

    Returns:
        str|Tuple[str, int, str]: (code, label, name)
    """
    if dataset_name == DATASET_3["name"] or dataset_name == DATASET_1["name"]:
        file_number = get_file_number_from_filename(filename, dataset_name)
        label_idx = imagebet1K_val_groundtruth_labels.get(file_number, None)
        label_code = imagenet1K_labels_to_codes.get(label_idx, None)
        label_name = imagenet1K_labels_to_names.get(label_idx, None)
    else:
        label_code = get_label_code_from_filename(filename, dataset_name)
        label_idx = imagenet1K_codes_to_labels.get(label_code, None)
        label_name = imagenet1K_labels_to_names.get(label_idx, None)
```


```
python3 datasets.py 
Expected n04380533, but got n04380533 for ILSVRC2012_val_00000026_n04380533.JPEG in imagenet_val_images (50K images)
Expected n03710193, but got n03710193 for ILSVRC2012_val_00000152_n03710193.JPEG in imagenet_val_images (50K images)
Expected n02089078, but got n02089078 for n02089078_black-and-tan_coonhound.JPEG in imagenet-sample-images (1K images)
Expected n02395406, but got n02395406 for n02395406_hog.JPEG in imagenet-sample-images (1K images)
-- Tests of get_label_code_from_filename() passed
root@USA-HP-Z2-Tower-G9-Workstation-Desktop-PC:/pytorch/ncnn/build/onnx2ncnn/test/imagenet1k# vim datasets.py 
root@USA-HP-Z2-Tower-G9-Workstation-Desktop-PC:/pytorch/ncnn/build/onnx2ncnn/test/imagenet1k# python3 datasets.py 
Expected n04380533, but got n04380533 for ILSVRC2012_val_00000026_n04380533.JPEG in imagenet_val_images (50K images)
Expected n03710193, but got n03710193 for ILSVRC2012_val_00000152_n03710193.JPEG in imagenet_val_images (50K images)
Expected n02089078, but got n02089078 for n02089078_black-and-tan_coonhound.JPEG in imagenet-sample-images (1K images)
Expected n02395406, but got n02395406 for n02395406_hog.JPEG in imagenet-sample-images (1K images)
-- Tests of get_label_code_from_filename() passed
 code, label, name =('n02395406', 341, 'hog')
```

> ## C++ match

```
#include <iostream>
#include <string>
#include <regex>

int main() {
    std::string text = "2023-10-05";
    std::regex pattern(R"((\d{4})-(\d{2})-(\d{2}))"); // 定义分组
    std::smatch matches;

    // 类似 re.match(pattern, text)
    if (std::regex_match(text, matches, pattern)) {
        // matches[0] 是整个匹配的字符串
        std::cout << "Full match: " << matches[0] << std::endl;
        
        // matches[1], [2], [3] 是各个分组
        std::cout << "Year: " << matches[1] << std::endl;
        std::cout << "Month: " << matches[2] << std::endl;
        std::cout << "Day: " << matches[3] << std::endl;
    } else {
        std::cout << "Match failed" << std::endl;
    }

    return 0;
}
```