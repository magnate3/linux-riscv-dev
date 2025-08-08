FROM arm64v8/ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
USER root
# Some essentials
RUN apt-get update && apt-get -y install \
	vim \
	iputils-ping \
	traceroute \
	iproute2 \
	nmap \
    net-tools \
    python3-requests \
    python3-flask

# Install base build packages dependencies - step 1
RUN apt-get update && apt-get -y install \
    git \
    cmake \
    g++ \
    pkg-config \
    autoconf \
    automake \
    libtool \
    libfftw3-dev \
    libusb-1.0-0-dev \
    wget \
    libusb-dev

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        libcurl4-openssl-dev \
	libssl-dev \
        libbpf-dev \
        libnuma-dev \
        pkg-config \
        python3-pip \
        python3-pyelftools \
        wget \
        libatomic1 \
        libbpf-dev \
        libnuma-dev \
        pciutils \
        python3 \
        python3-pyelftools \
        # For MLX5 driver
        ibverbs-providers \
        libibverbs-dev 

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    libboost-all-dev\
    libasio-dev \
    zlib1g-dev 


#RUN mkdir /mnt/huge-1048576kB > /dev/null
#COPY dpdk-stable-16.07.2 /root/dpdk/.
#ENV HOME /root
#ENV RTE_SDK /root/sw_package/dpdk-stable-16.07.2
#ENV RTE_TARGET=x86_64-native-linuxapp-gcc
#ENV PATH “$PATH:$RTE_SDK/$RTE_TARGET/app”
#WORKDIR /root/dpdk/x86_64-native-linuxapp-gcc/app
#RUN pip3 install meson ninja
#RUN  ln -s /usr/bin/make /usr/bin/gmake
RUN [ -e /usr/bin/gmake ] || ln -s /usr/bin/make /usr/bin/gmake
#RUN mkdir /app
#RUN rm -rf /app
#WORKDIR /app
#RUN wget https://cmake.org/files/v3.5/cmake-3.5.2.tar.gz
#RUN tar -zxf cmake-3.5.2.tar.gz -C /app && cd /app/cmake-3.5.2
##COPY  cmake-3.5.0 /app
##WORKDIR  /usr/src/app/cmake-3.5.0
##RUN cd /root/prog/cmake-3.5.0
#RUN ./bootstrap && make && make install && rm -rf /app/cmake-3.5.2 /app/cmake-3.5.2.tar.gz
#ENTRYPOINT ["/start.sh"]
#ADD entrypoint.sh /opt/entrypoint.sh
#ADD app.py /opt/app.py

#EXPOSE 8085/tcp

#ENTRYPOINT ["/bin/bash", "/opt/entrypoint.sh"]
#ENTRYPOINT ["/bin/bash"]
ENTRYPOINT  ["/bin/bash",  "-c"]
#ENTRYPOINT  ["/bin/bash", "-l", "-c"]
ENTRYPOINT  ["/bin/bash", "-l", "-c"]
ENTRYPOINT  ["/bin/bash", "-l", "-c"]
