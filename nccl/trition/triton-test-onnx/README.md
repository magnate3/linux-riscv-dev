# triton-test-onnx

## Getting Started

```bash
$ ./run.sh
```

```bash
Sending build context to Docker daemon  38.61MB
Step 1/12 : FROM ubuntu:18.04
 ---> c3c304cb4f22
Step 2/12 : ENV DEBIAN_FRONTEND=noninteractive
 ---> Using cache
 ---> c1a1c9b8e0fb
Step 3/12 : ENV LC_ALL=C.UTF-8
 ---> Using cache
 ---> 8134ec205f27
Step 4/12 : ENV LANG=C.UTF-8
 ---> Using cache
 ---> 76b58e3e35a1
Step 5/12 : RUN apt-get update     && apt-get install -y --no-install-recommends         python3         python3-pip         curl         g++         python3-dev         build-essential         cmake         git         zlib1g-dev         libssl-dev     && apt-get autoremove -y     && rm -rf /var/lib/apt/lists/*     && rm -rf /var/cache/apt/archives/partial/*
 ---> Using cache
 ---> 9348e5a288da
Step 6/12 : RUN rm -f /usr/bin/python /usr/bin/pip     && ln -s /usr/bin/python3 /usr/bin/python     && ln -s /usr/bin/pip3 /usr/bin/pip
 ---> Using cache
 ---> 0c095307f8ed
Step 7/12 : WORKDIR /app
 ---> Using cache
 ---> 75ae938cd936
Step 8/12 : ARG TRITON_CLIENTS_URL=https://github.com/NVIDIA/triton-inference-server/releases/download/v1.13.0/v1.13.0_ubuntu1804.clients.tar.gz
 ---> Using cache
 ---> 8b007179bea2
Step 9/12 : RUN mkdir -p /opt/nvidia/triton-clients     && curl -L ${TRITON_CLIENTS_URL} | tar xvz -C /opt/nvidia/triton-clients
 ---> Using cache
 ---> 09bc2f41e193
Step 10/12 : RUN pip install --no-cache-dir --upgrade setuptools wheel     && pip install --no-cache-dir /opt/nvidia/triton-clients/python/*.whl
 ---> Using cache
 ---> 6c63978d180f
Step 11/12 : COPY ./main.py ./
 ---> 0b6236dbfd20
Step 12/12 : ENTRYPOINT ["python", "main.py", "infer"]
 ---> Running in 77321d2ca898
Removing intermediate container 77321d2ca898
 ---> ed3dc9aa5120
Successfully built ed3dc9aa5120
Successfully tagged triton-test:latest
c50f41778b573a24588412db8faed574858db0857b07d6d65cd335dc87017eb4
fdae29749b1b5d9d91aacd602f3b31d8705b5c8418dff0eed1396b8faaecc95b
Wait until TRITON (192.168.176.2) is ready.......done
NVIDIA Release 20.03.1 (build 12830698)

Copyright (c) 2018-2020, NVIDIA CORPORATION.  All rights reserved.

Various files include modifications (c) NVIDIA CORPORATION.  All rights reserved.
NVIDIA modifications are covered by the license terms that apply to the underlying
project or file.

2020-06-11 22:50:40.004838: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.2
I0611 22:50:40.038525 1 metrics.cc:164] found 1 GPUs supporting NVML metrics
I0611 22:50:40.044062 1 metrics.cc:173]   GPU 0: Quadro GV100
I0611 22:50:40.044537 1 server.cc:130] Initializing Triton Inference Server
I0611 22:50:40.180066 1 server_status.cc:55] New status tracking for model 'test_model'
I0611 22:50:40.180141 1 model_repository_manager.cc:723] loading: test_model:1
I0611 22:50:40.214573 1 onnx_backend.cc:203] Creating instance test_model_0_0_gpu0 on GPU 0 (7.0) using model.onnx
I0611 22:50:43.082232 1 model_repository_manager.cc:888] successfully loaded 'test_model' version 1
Starting endpoints, 'inference:0' listening on
I0611 22:50:43.084915 1 grpc_server.cc:1942] Started GRPCService at 0.0.0.0:8001
I0611 22:50:43.084957 1 http_server.cc:1428] Starting HTTPService at 0.0.0.0:8000
I0611 22:50:43.127329 1 http_server.cc:1443] Starting Metrics Service at 0.0.0.0:8002
Trying to check TRITON server health ...
index: (2, 0.4643319547176361, 'class3') value: (1, 0.30929985642433167, 'class2') class: (0, 0.22636820375919342, 'class1')
index: (1, 0.36855363845825195, 'class2') value: (2, 0.3355056941509247, 'class3') class: (0, 0.29594069719314575, 'class1')
index: (0, 0.4354143440723419, 'class1') value: (2, 0.3288537263870239, 'class3') class: (1, 0.23573192954063416, 'class2')
index: (2, 0.40138566493988037, 'class3') value: (0, 0.31763410568237305, 'class1') class: (1, 0.28098025918006897, 'class2')
index: (0, 0.3852733075618744, 'class1') value: (2, 0.3509983718395233, 'class3') class: (1, 0.2637283504009247, 'class2')
index: (0, 0.4151337742805481, 'class1') value: (1, 0.3550426661968231, 'class2') class: (2, 0.22982361912727356, 'class3')
index: (1, 0.3793274462223053, 'class2') value: (2, 0.35955652594566345, 'class3') class: (0, 0.26111605763435364, 'class1')
index: (0, 0.4355199337005615, 'class1') value: (1, 0.30359625816345215, 'class2') class: (2, 0.26088377833366394, 'class3')
index: (2, 0.4768834114074707, 'class3') value: (0, 0.2949536144733429, 'class1') class: (1, 0.2281629592180252, 'class2')
index: (0, 0.3654273748397827, 'class1') value: (1, 0.35484451055526733, 'class2') class: (2, 0.27972814440727234, 'class3')
triton-test has finished.
Stopping TRITON
```

Model file is available at `models/test_model/1/model.onnx` which was prebuilt by using the following command:

```bash
pip install torch numpy
python main.py model
```

