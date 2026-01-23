CUDA_HOME ?= /usr/local/cuda
ROCM_HOME ?= /opt/rocm
MSCCLPP_HOME ?= /pytorch/mscclpp/install

# Check if nvcc exists, otherwise use hipcc
ifeq ($(shell which $(CUDA_HOME)/bin/nvcc 2>/dev/null),)
    COMPILER := $(ROCM_HOME)/bin/hipcc
    ARCH_FLAG := -D__HIP_PLATFORM_AMD__=1
else
    COMPILER := $(CUDA_HOME)/bin/nvcc
    ARCH_FLAG := -arch=native
endif

TARGET = gpu_ping_pong_mp
SRC = gpu_ping_pong_mp.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(COMPILER) $(ARCH_FLAG) -o $@ $<  -I$(MSCCLPP_HOME)/include -L$(MSCCLPP_HOME)/lib -lmscclpp

clean:
	rm -f $(TARGET)
