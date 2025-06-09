# Build image on top of NVidia MXnet image

# Pin Key Package Versions
MOFED_VER=5.8-6.0.4.2
AZML_SDK_VER=1.25.0

# Other required variables for MOFED drivers
OS_VER=ubuntu20.04
PLATFORM=x86_64

### Install Mellanox Drivers ###
#apt-get update && apt-get install -y libcap2 libfuse-dev && \
#wget --quiet http://content.mellanox.com/ofed/MLNX_OFED-${MOFED_VER}/MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz 
tar -xvf MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}.tgz && \
    MLNX_OFED_LINUX-${MOFED_VER}-${OS_VER}-${PLATFORM}/mlnxofedinstall --user-space-only --without-fw-update --all --without-neohost-backend --force


