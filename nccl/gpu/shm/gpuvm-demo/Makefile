NVCC ?= nvcc
TARGET = gpuvm
all: $(TARGET)

$(TARGET): gpuvm.cpp page_allocator.cpp
	$(NVCC) $^ -o $@ -O2 -g -lcuda -std=c++17

clean:
	$(RM) $(TARGET)
