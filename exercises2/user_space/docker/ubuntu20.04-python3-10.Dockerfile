FROM yellow.hub.cambricon.com/pytorch/base/x86_64/pytorch:v0.1-x86_64-ubuntu20.04

#shell
SHELL ["/bin/bash", "-c"]

# install the time package
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
		&& apt-get install -y tzdata \
		&& ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
		&& dpkg-reconfigure --frontend noninteractive tzdata
	
#install base packages
# libsndfile1 - used in spectral ops test case.
# numactl - used in MLPerf network for performace, eg: BERT-Large.
RUN apt-get install -y wget aptitude vim make cmake git tcl-dev tk-dev software-properties-common libsqlite3-dev libopenblas-dev \
		&& aptitude -y install zlib1g-dev libffi-dev libssl-dev libncurses-dev libbz2-dev libxml2-dev libxslt1-dev libgdbm-dev libreadline-dev liblzma-dev
RUN apt-get install -y libibverbs-dev libboost-all-dev libgflags-dev libgoogle-glog-dev libopencv-dev zip rsync libsndfile1 gpg-agent apt-transport-https ca-certificates numactl libeigen3-dev libjpeg62 perl

# install extra dependencies
RUN mkdir -p /tmp/extra-dependencies
#install latest cmake. See https://askubuntu.com/questions/355565/how-do-i-install-the-latest-version-of-cmake-from-the-command-line
ARG version=3.24
ARG build=1
RUN apt remove -y --purge --auto-remove cmake && \
    wget https://cmake.org/files/v$version/cmake-$version.$build.tar.gz && \
    tar -xzvf cmake-$version.$build.tar.gz && \
    pushd cmake-$version.$build/ && \
    ./bootstrap && \
    make -j$(nproc) && \
    make install

#update the gcc version
RUN apt-get install -y gcc g++

# Install IB
RUN wget -qO - https://www.mellanox.com/downloads/ofed/RPM-GPG-KEY-Mellanox | apt-key add - && \
    mkdir -p /etc/apt/sources.list.d && \
    wget -q -nc --no-check-certificate -P /etc/apt/sources.list.d https://linux.mellanox.com/public/repo/mlnx_ofed/5.3-1.0.0.1/ubuntu20.04/mellanox_mlnx_ofed.list && \
    apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ibverbs-providers \
    ibverbs-utils \
    libibmad-dev \
    libibmad5 \
    libibumad-dev \
    libibumad3 \
    libibverbs-dev \
    libibverbs1 \
    librdmacm-dev \
    librdmacm1 && \
    rm -rf /var/lib/apt/lists/*

## opencv
RUN mkdir -p /tmp/extra-dependencies
RUN cd /tmp/extra-dependencies && \
	wget https://github.com/opencv/opencv/archive/3.4.14.zip && \
	unzip 3.4.14.zip && \
	cd opencv-3.4.14/ && mkdir build && cd build && \
	cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
	make -j && make install
## gflags
RUN cd /tmp/extra-dependencies && \
    git clone https://github.com/gflags/gflags && \
	cd gflags && git checkout v2.2.2 && \
	mkdir build && cd build && \
	cmake -DGFLAGS_NAMESPACE=gflags -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_CXX_FLAGS="-std=c++14 -D _GLIBCXX_USE_CXX11_ABI=0" .. && \
	make -j && make install
## glog
RUN cd /tmp/extra-dependencies && \
    git clone https://github.com/google/glog && \
#	git clone https://gitee.com/boxingcao/glog.git && \
    cd glog && mkdir build && cd build && \
	cmake -DGFLAGS_NAMESPACE=google -DCMAKE_CXX_FLAGS=-fPIC -DBUILD_SHARED_LIBS=ON -DCMAKE_CXX_FLAGS="-std=c++14 -D _GLIBCXX_USE_CXX11_ABI=0" .. && \
	make -j && make install
RUN rm -rf /tmp/extra-dependencies

ARG python_version="3.10"
# install python3.10
RUN if [[ ${python_version} == "3.10" ]]; then cd /tmp && \
    wget http://gitlab-software.cambricon.com/neuware/software/framework/pytorch/extra-dependencies/-/raw/master/python/Python-3.10.8.tgz && \
    tar xvf Python-3.10.8.tgz && \
    cd Python-3.10.8 && \
    ./configure --prefix=/opt/py3.10 --enable-ipv6 --with-ensurepip=no  --with-computed-gotos --with-system-ffi --enable-loadable-sqlite-extensions --with-tcltk-includes=-I/opt/py3.10/include '--with-tcltk-libs=-L/opt/py3.10/lib -ltcl8.6 -ltk8.6' --enable-optimizations --with-lto --enable-shared 'CFLAGS=-march=nocona -mtune=haswell -ftree-vectorize -fPIC -fstack-protector-strong -fno-plt -O2 -pipe' 'LDFLAGS=-Wl,-O2 -Wl,--sort-common -Wl,--as-needed -Wl,-z,relro -Wl,-z,now -Wl,-rpath,/opt/py3.10/lib -L/opt/py3.10/lib' 'CPPFLAGS=-DNDEBUG -D_FORTIFY_SOURCE=2 -O2 -I/opt/py3.10/include' PKG_CONFIG_PATH=/opt/py3.10/lib/pkgconfig --with-ensurepip=install && \
    make -j && \
    make altinstall && \
    # use the following 2 lines to compile python without tests    
    # make -j8 build_all && \
    # make -j8 altinstall && \
    rm /tmp/Python-3.10.8.tgz && \
    rm -rf /tmp/Python-3.10.8; \
    fi

ENV CPLUS_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$CPLUS_INCLUDE_PATH
ENV C_INCLUDE_PATH=/opt/py${python_version}/include/python${python_version}m:$C_INCLUDE_PATH

RUN if [[ ${python_version} == "3.10" ]]; then \
    /opt/py3.10/bin/pip3.10 config set global.index-url  http://mirrors.aliyun.com/pypi/simple && \
    /opt/py3.10/bin/pip3.10 config set install.trusted-host mirrors.aliyun.com  && \
    /opt/py3.10/bin/pip3.10 install virtualenv && \
    # set ln
    ln -sf /opt/py3.10/bin/python3.10 /usr/bin/python3 && \
    ln -sf /opt/py3.10/bin/pip3.10 /usr/bin/pip && \
    ln -sf /opt/py3.10/bin/virtualenv /usr/bin/virtualenv && \
    ln -sf /usr/bin/cmake3 /usr/bin/cmake; \
    fi

# fetch and install bazel
ARG BAZEL_VERSION="3.4.1"
RUN wget -O ~/bazel.sh http://gitlab.software.cambricon.com/neuware/platform/extra-dependencies/raw/master/bazel/bazel-${BAZEL_VERSION}-installer-linux-x86_64.sh && \
    chmod +x ~/bazel.sh && \
    ~/bazel.sh && \
    rm -f ~/bazel.sh

# Install Open MPI
ENV MPI_HOME=/usr/local/openmpi
ENV PATH=${MPI_HOME}/bin:$PATH
ENV LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
 
RUN mkdir -p $MPI_HOME
RUN cd /tmp && \
    apt-get -y install bzip2 && \
    wget http://gitlab.software.cambricon.com/neuware/platform/extra-dependencies/-/raw/master/mpi/openmpi-4.1.0.tar.bz2 && \
    tar -jxvf openmpi-4.1.0.tar.bz2 && \
    cd openmpi-4.1.0 && \
    ./configure --prefix=$MPI_HOME --enable-orterun-prefix-by-default && \
    make -j && \
    make install && \
    rm /tmp/openmpi-4.1.0.tar.bz2 && \
    rm -rf /tmp/openmpi-4.1.0
