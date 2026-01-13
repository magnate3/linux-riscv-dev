NVCC ?= nvcc
TARGET = quad
all: $(TARGET)

$(TARGET):  hello.cu
	$(NVCC) $^ -o $@ -O2 -g -lcuda -std=c++17

clean:
	$(RM) $(TARGET)
