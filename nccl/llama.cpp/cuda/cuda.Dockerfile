ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.4.0
# Target the CUDA build image
#ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG BASE_CUDA_RUN_CONTAINER=nvcr.io/nvidia/cuda:11.6.2-devel-ubuntu20.04
#ARG BASE_CUDA_RUN_CONTAINER=nvcr.io/nvidia/cuda:11.6.2-runtime-ubuntu20.04
#ARG BASE_CUDA_RUN_CONTAINER=nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
#ARG BASE_CUDA_RUN_CONTAINER=nvcr.io/nvidia/cuda:12.4.0-runtime-ubuntu22.04
#ARG BASE_CUDA_RUN_CONTAINER=nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_RUN_CONTAINER} AS build

# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=86
#ARG CUDA_DOCKER_ARCH=default

#定义时区参数
ENV TZ=Asia/Shanghai
#设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo '$TZ' > /etc/timezone
#RUN apt-get remove --purge -y cmake=3.15.* 
#RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add -
#RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
# 更新包索引并安装curl（如果需要的话）
RUN apt-get update && apt-get install -y curl
# 安装必要的工具，如wget和软件属性公共密钥环工具（apt-transport-https）
RUN apt-get install -y wget software-properties-common

# 添加VTK的PPA源（注意：这里使用的是Debian而非Ubuntu的PPA源）
RUN wget -q -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"

# 更新软件包列表以包含新的PPA源
RUN apt-get update

# 安装VTK
RUN apt-get install -y vtk7

RUN apt-get update && \
    apt-get install -y build-essential cmake python3 python3-pip git libssl-dev libgomp1
#RUN pip install --upgrade cmake==3.18.2
WORKDIR /app

COPY . .
ARG GGML_CPU_ARM_ARCH="armv8-a"
RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
    export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=${GGML_CPU_ARM_ARCH} -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined .     && \
    cmake --build build --config Release -j$(nproc)
#GGML_CPU_ARM_ARCH="armv8-a"
#ARCH=$(uname -m)
#RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
#    export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
#    fi
#if [ "$ARCH" = "amd64" ]; then
#    echo "Building for x86_64 (amd64)..."
#    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
#elif [ "$ARCH" = "arm64" ]; then
#    echo "Building for ARM64..."
#    cmake -S . -B build  -DLLAMA_CURL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=${GGML_CPU_ARM_ARCH} ${CMAKE_ARGS} -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON  -DLLAMA_BUILD_TESTS=OFF -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined .
#    #cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=${GGML_CPU_ARM_ARCH} -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
#    cmake --build build --config Release -j$(nproc)
#else
#    echo "Unsupported architecture: $ARCH"
#    exit 1
#fi
#&& \
#    cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . && \
#    cmake --build build --config Release -j$(nproc)

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

RUN apt-get update \
    && apt-get install -y libgomp1 curl\
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete
## Base image
#FROM ${BASE_CUDA_RUN_CONTAINER} AS base
#
#RUN apt-get update \
#    && apt-get install -y libgomp1 curl\
#    && apt autoremove -y \
#    && apt clean -y \
#    && rm -rf /tmp/* /var/tmp/* \
#    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
#    && find /var/cache -type f -delete
#
#COPY --from=build /app/lib/ /app
#
#### Full
#FROM base AS full
#
#COPY --from=build /app/full /app
#
#WORKDIR /app
#
#RUN apt-get update \
#    && apt-get install -y \
#    git \
#    python3 \
#    python3-pip \
#    && pip install --upgrade pip setuptools wheel \
#    && pip install --break-system-packages -r requirements.txt \
#    && apt autoremove -y \
#    && apt clean -y \
#    && rm -rf /tmp/* /var/tmp/* \
#    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
#    && find /var/cache -type f -delete
#
#
#ENTRYPOINT ["/app/tools.sh"]
#
#### Light, CLI only
#FROM base AS light
#
#COPY --from=build /app/full/llama-cli /app/full/llama-completion /app
#
#WORKDIR /app
#
#ENTRYPOINT [ "/app/llama-cli" ]
#
#### Server, Server only
#FROM base AS server
#
#ENV LLAMA_ARG_HOST=0.0.0.0
#
#COPY --from=build /app/full/llama-server /app
#
#WORKDIR /app
#
#HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]
#
#ENTRYPOINT [ "/app/llama-server" ]
