NVCC        := nvcc
TARGET      := main
NVCCFLAGS   := -arch=sm_80 -O2

SRCS        := main.cpp kernel*.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCCFLAGS) -o $(TARGET) $(SRCS)

clean:
	rm -f $(TARGET)