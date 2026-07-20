#include ../../common/make.config
CUDA_DIR := /usr/local/cuda
CUDA_LIB_DIR := $(CUDA_DIR)/lib
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
     	CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif

CC := $(CUDA_DIR)/bin/nvcc

INCLUDE := $(CUDA_DIR)/include

SRC = bench.cu

EXE = bench

all: gpud baseline

release: $(SRC)
	$(CC) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

baseline: $(SRC)
	$(CC) $(SRC) -g -DHAVE_LINUX_PERF_EVENT_H -DREADING_FROM_BINARY -DOUTPUT_TO_BINARY -o $(EXE)_hg -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -lcudart 

gpud: $(SRC)
	$(CC) $(SRC) -g -DHAVE_LINUX_PERF_EVENT_H -DGPUD -DREADING_FROM_BINARY -DOUTPUT_TO_BINARY -o $(EXE)_gpud -I$(INCLUDE) -L$(CUDA_LIB_DIR) -lcuda -lcudart -lnvmed

clean: $(SRC)
	rm -f $(EXE) $(EXE)_gpud
