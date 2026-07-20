# Compiler and flags
NVCC := g++
NCCL_DIR = /workspace/nccl-latest
C10D_DIR = ./
LIBTORCH_LIB = /usr/local/lib/python3.9/site-packages/torch/lib
LIBS  += -Wl,-rpath=$(LIBTORCH_LIB) -L$(LIBTORCH_LIB) -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -ltorch
CFLAGS := -g -I$(NCCL_DIR)/build/include  -I$(NCCL_DIR)/fake_cuda/include  -I$(NCCL_DIR)/src/include -I$(NCCL_DIR)/src/graph  -I$(NCCL_DIR)/src/include/plugin -I/usr/local/lib/python3.9/site-packages/torch/include  -I../  -ldl -L$(NCCL_DIR)/build/lib -lnccl -lpthread $(LIBS)

# Targets
TARGETS := main

# Source files
SRCS := $(C10D_DIR)/ProcessGroupNCCL.cpp   main.cpp

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
