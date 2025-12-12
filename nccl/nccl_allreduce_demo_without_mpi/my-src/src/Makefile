# Compiler and flags
NVCC := g++
NCCL_DIR = /workspace/nccl-latest
CFLAGS := -g -I$(NCCL_DIR)/build/include  -I$(NCCL_DIR)/fake_cuda/include  -I$(NCCL_DIR)/src/include -I$(NCCL_DIR)/src/graph  -I$(NCCL_DIR)/src/include/plugin -ldl -L$(NCCL_DIR)/build/lib -lnccl -lpthread 

# Targets
TARGETS := main

# Source files
SRCS := nccl_dbg.cpp gpu_worker.cpp    tcp_socket.cpp   main.cpp

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
