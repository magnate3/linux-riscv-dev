CC=nvcc

DEBUG=0

ifneq ($(DEBUG), 0)
CFLAGS=-O0 -g -G -DDEBUG=0
else
CFLAGS=-O3 -g -lineinfo
endif

CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE = $(CUDA_PATH)/include
CUDA_LIBS = $(CUDA_PATH)/lib64
CUDA_BIN  = $(CUDA_PATH)/bin
NVCC = $(CUDA_BIN)/nvcc

CFLAGS+=-Xcompiler=-Wall -maxrregcount=32 -arch=sm_75
#CFLAGS+=`pkg-config libibverbs --cflags --libs`
CFLAGS+=-I$(CUDA_INCLUDE) -L$(CUDA_LIBS)
# Use to find out shared memory size
# CFLAGS+=--ptxas-options=-v 

FILES=naiveGpipeMain threeElementGpipeMain gpipeMain

all: $(FILES)

naiveGpipeMain: naiveGpipeMain.o
	$(NVCC) --link $(CFLAGS) $^ -o $@

threeElementGpipeMain: threeElementGpipeMain.o
	$(NVCC) --link $(CFLAGS) $^ -o $@

gpipeMain: gpipeMain.o
	$(NVCC) --link $(CFLAGS) $^ -o $@ -lcuda

naiveGpipeMain.o: naiveGpipeMain.cu naiveGpipe.cu testCommon.cu common.cuh timer.h
threeElementGpipeMain.o: threeElementGpipeMain.cu threeElementGpipe.cu testCommon.cu timer.h 
gpipeMain.o: gpipeMain.cu gpipe.cu memoryManagmentHelper.cuh connectionHelper.cuh gpipe.cuh testCommon.cu timer.h common.cuh

%.o: %.cu
	nvcc --compile $< $(CFLAGS) -o $@

clean::
	rm -f *.o $(FILES)
