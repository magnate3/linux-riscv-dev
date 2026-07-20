FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        curl \
        g++ \
        python3-dev \
        build-essential \
        cmake \
        git \
        zlib1g-dev \
        libssl-dev \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/cache/apt/archives/partial/*

RUN rm -f /usr/bin/python /usr/bin/pip \
    && ln -s /usr/bin/python3 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# Download TRITON Client
ARG TRITON_CLIENTS_URL=https://github.com/NVIDIA/triton-inference-server/releases/download/v1.13.0/v1.13.0_ubuntu1804.clients.tar.gz
RUN mkdir -p /opt/nvidia/triton-clients \
    && curl -L ${TRITON_CLIENTS_URL} | tar xvz -C /opt/nvidia/triton-clients

RUN pip install --no-cache-dir --upgrade setuptools wheel \
    && pip install --no-cache-dir /opt/nvidia/triton-clients/python/*.whl

COPY ./main.py ./

ENTRYPOINT ["python", "main.py", "infer"]