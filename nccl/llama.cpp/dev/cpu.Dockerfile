ARG UBUNTU_VERSION=22.04

FROM ubuntu:$UBUNTU_VERSION AS base

#ARG TARGETARCH

RUN apt-get update && \
    apt-get install -y build-essential git cmake libssl-dev libgomp1 curl

RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && pip install --upgrade pip setuptools wheel \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

#not need to set ENTRYPOINT,otherwise will start docker fail
#ENTRYPOINT ['/bin/bash']
