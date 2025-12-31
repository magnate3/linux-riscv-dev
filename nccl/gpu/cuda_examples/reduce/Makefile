NVCC = nvcc
CXXFLAGS = -std=c++11
NVCCFLAGS = $(CXXFLAGS) -O2 -arch=sm_80
TARGET = my_program
KERNELS = $(wildcard kernel_v*.cu)

all: $(TARGET)

$(TARGET): $(KERNELS) main.cpp kernel.h
	$(NVCC) $(NVCCFLAGS) $(KERNELS) main.cpp -o $(TARGET)

clean:
	rm -f $(TARGET)