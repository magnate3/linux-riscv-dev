# Compiler and flags
NVCC := g++
NCCL_DIR = /workspace/nccl-latest
C10D_DIR = ./c10d

MPI_INC = -I/usr/lib/x86_64-linux-gnu/openmpi/include/
MPI_LIB =  /usr/lib/x86_64-linux-gnu/libmpi.so
TORCH_INC = -I/usr/local/lib/python3.9/site-packages/torch/include/torch/csrc/api/include/ -I /usr/local/lib/python3.9/site-packages/torch/include
LIBTORCH_LIB = /usr/local/lib/python3.9/site-packages/torch/lib
CUDA_LIB =  ./libcuda 
LIBS  += -Wl,-rpath=$(LIBTORCH_LIB) -L$(LIBTORCH_LIB) -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch
LIBS  += -Wl,-rpath=$(MPI_LIB) -L$(MPI_LIB) -lmpi -lmpi_cxx
LIBS  += -Wl,-rpath=$(CUDA_LIB) -L$(CUDA_LIB) -lcuda
#CFLAGS := -std=c++17   -g -I$(NCCL_DIR)/build/include  -I$(NCCL_DIR)/fake_cuda/include  -I$(NCCL_DIR)/src/include -I$(NCCL_DIR)/src/graph  -I$(NCCL_DIR)/src/include/plugin $(TORCH_INC)   -ldl -L$(NCCL_DIR)/build/lib -lnccl -lpthread $(LIBS)
CFLAGS := -D_GLIBCXX_USE_CXX11_ABI=0  -g -I$(NCCL_DIR)/build/include  -I$(NCCL_DIR)/fake_cuda/include  -I$(NCCL_DIR)/src/include -I$(NCCL_DIR)/src/graph  -I$(NCCL_DIR)/src/include/plugin -I./ $(TORCH_INC) $(MPI_INC)  -ldl -L$(NCCL_DIR)/build/lib -lnccl -lpthread $(LIBS)
# Targets
TARGETS := main

# Source files
#SRCS := $(C10D_DIR)/ProcessGroupNCCL.cpp   main.cpp
SRCS  := $(wildcard *.cpp) $(wildcard ./$(C10D_DIR)/*.cpp)  $(wildcard ./$(C10D_DIR)/test/ProcessGroupNCCLTest.cpp)
SRCS  := $(filter-out %GlooDeviceFactory.cpp,$(SRCS))
SRCS  := $(filter-out %ProcessGroupGloo.cpp,$(SRCS))
#SRCS  := $(filter-out %_main.cc,$(SRCS))
#OBJ  := $(SRC:%.cpp=%.o)

# Object files
OBJS := $(SRCS:.cpp=.o)

# Default target
all: $(TARGETS)

# Rule to build server
main: $(OBJS)
	$(NVCC)  -o $@ $^ $(CFLAGS)


# Pattern rule to build object files
%.o: %.cpp
	$(NVCC) $(CFLAGS)  -c $< -o $@

# Clean rule
clean:
	rm -rf $(TARGETS) $(OBJS)

.PHONY: all clean
