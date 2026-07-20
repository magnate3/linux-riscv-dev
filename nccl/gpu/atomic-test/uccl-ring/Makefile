CXX      := g++
NVCC     := nvcc
CXXFLAGS := -std=c++17 -O3 -g -Wall
NVCCFLAGS:= -O3 -g -G -Xcompiler -Wall -arch=sm_80 

CUDA_HOME ?= /usr/local/cuda
GDRCOPY_LIBDIR = $(abspath ../../../../thirdparty/gdrcopy/src)

INCLUDES := -I./ -I$(CUDA_HOME)/include
LIBS := -L$(CUDA_HOME)/lib -lcudart -lcuda -lnuma \
        -L$(GDRCOPY_LIBDIR) -lgdrapi

RPATH = -Xlinker -rpath -Xlinker $(GDRCOPY_LIBDIR)

#COMMON_CPP := c2d_fifo.cc
#COMMON_CU  := test_lock_throughput.cu

TEST_SRC   :=  test_lock_throughput.cu
BENCH_SRC  := cons_pro.cu
BENCH_FULL_SRC  := bench_full_fifo.cu

COMMON_OBJ := $(COMMON_CPP:.cc=.o) $(COMMON_CU:.cu=.o)
TEST_OBJ   := $(TEST_SRC:.cc=.o)
BENCH_OBJ  := $(BENCH_SRC:.cu=.o)
BENCH_FULL_OBJ  := $(BENCH_FULL_SRC:.cu=.o)

TEST_BIN  := test_lock_throughput
BENCH_BIN := cons_pro
BENCH_FULL_BIN := bench_full_fifo

#ALL_BIN   :=  $(BENCH_BIN) $(BENCH_FULL_BIN)
ALL_BIN   := $(TEST_BIN)  $(BENCH_BIN)


.PHONY: all test bench clean

all: $(ALL_BIN)

test: $(TEST_BIN)

bench: $(BENCH_BIN) $(BENCH_FULL_BIN)

%.o: %.cc
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(INCLUDES)

%.o: %.cu
	$(NVCC) -c $< -o $@ $(NVCCFLAGS) $(INCLUDES)

$(TEST_BIN): $(COMMON_OBJ) $(TEST_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_BIN): $(COMMON_OBJ) $(BENCH_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

$(BENCH_FULL_BIN): $(COMMON_OBJ) $(BENCH_FULL_OBJ)
	$(NVCC) $^ -o $@ $(LIBS) $(RPATH) $(NVCCFLAGS)

clean:
	rm -f *.o *.d $(ALL_BIN)
