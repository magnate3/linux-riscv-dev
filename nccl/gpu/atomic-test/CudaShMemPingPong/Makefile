# Compiler
NVCC = nvcc

# Flags
CFLAGS = -g -arch=sm_80 -Xcompiler -O3 -Xcicc -O3 -lineinfo

# Output file
OUTPUT = MP.out

# Source file
SRC = MP_base.cu

# Header files
HEADERS = $(wildcard *.h *.hpp *.cuh)

# Default target
all: $(OUTPUT)

# Build target
$(OUTPUT): $(SRC) $(HEADERS)
	$(NVCC) $(CFLAGS) -o $@ $<

# Clean target
clean:
	rm -f $(OUTPUT)