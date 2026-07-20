# Compiler and flags
MPICC = mpicc
CFLAGS = -Wall -Wextra -O2 -g
MPI_LDFLAGS = -libverbs -lrdmacm
CUDA_LDFLAGS = -lcuda -lcudart -L/usr/local/cuda/lib64 -I/usr/local/cuda/include

# Targets
PERF_TARGETS = ib_perf_multicast ib_perf_ud_unicast ib_perf_rc_unicast

# Sources
PERF_SOURCES = ib_perf_multicast.c ib_perf_ud_unicast.c ib_perf_rc_unicast.c

# Objects
PERF_OBJECTS = $(PERF_SOURCES:.c=.o)

# Default target
all: $(PERF_TARGETS)

# Individual targets
ib_perf_multicast: ib_perf_multicast.c
	$(MPICC) $(CFLAGS) -o $@ $< $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
	scp -P 12345 $@ snail01:/app/ib_tests/
	scp -P 12345 $@ snail02:/app/ib_tests/
	scp -P 12345 $@ snail03:/app/ib_tests/
	scp -P 12345 $@ tvm01:/app/ib_tests/
	scp -P 12345 $@ tvm02:/app/ib_tests/

ib_perf_ud_unicast: ib_perf_ud_unicast.c
	$(MPICC) $(CFLAGS) -o $@ $< $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
	scp -P 12345 $@ snail01:/app/ib_tests/
	scp -P 12345 $@ snail02:/app/ib_tests/
	scp -P 12345 $@ snail03:/app/ib_tests/
	scp -P 12345 $@ tvm01:/app/ib_tests/
	scp -P 12345 $@ tvm02:/app/ib_tests/

ib_perf_rc_unicast: ib_perf_rc_unicast.c
	$(MPICC) $(CFLAGS) -o $@ $< $(MPI_LDFLAGS) $(CUDA_LDFLAGS)
	scp -P 12345 $@ snail01:/app/ib_tests/
	scp -P 12345 $@ snail02:/app/ib_tests/
	scp -P 12345 $@ snail03:/app/ib_tests/
	scp -P 12345 $@ tvm01:/app/ib_tests/
	scp -P 12345 $@ tvm02:/app/ib_tests/

test: test-mcast test-ud_unicast test-rc_unicast
perf: run-mcast run-ud_unicast run-rc_unicast

# ------------------------------------------------------------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------------------------------------------------------------
MEM_TYPE = host
USE_IMM = 0
LOG_LEVEL = 0
ITERATIONS = 100

IMM_FLAGS =
ifeq ($(USE_IMM), 1)
	IMM_FLAGS = --with-imm
endif

# Run performance test with specific parameters
# ------------------------------------------------------------------------------------------------------------------------------------
# Multicast performance test
# ------------------------------------------------------------------------------------------------------------------------------------
run-mcast: ib_perf_multicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_0 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 2 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 3 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host snail02:1 $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host snail02:1 $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_2 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 1 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host snail03:1 $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host snail03:1 $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_2 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 1 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host tvm01:1   $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host tvm01:1   $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_2 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 1 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host tvm02:1   $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_2 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 1 -l 1024 -u 33554432 $(IMM_FLAGS) \
	# : -n 1 --host tvm02:1   $(FLAGS) -x LOG_LEVEL=$(LOG_LEVEL) -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -w 0 -i $(ITERATIONS) -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \

test-mcast: ib_perf_multicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_multicast -d mlx5_0 -l 8192 -u 8192 -w 1 -i 2 -m $(MEM_TYPE) -g 2 $(IMM_FLAGS) \
	: -n 1 --host snail02:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -l 8192 -u 8192 -w 1 -i 2 -m $(MEM_TYPE) -g 0 $(IMM_FLAGS) \

# test-mcast-cuda-perf: ib_perf_multicast
# 	mpirun \
# 	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=2 -- perf record -g --call-graph dwarf /app/ib_tests/ib_perf_multicast -d mlx5_0 -l 262144 -u 262144 -w 0 -i 100 -m cuda -g 2 \
# 	: -n 1 --host snail02:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_multicast -d mlx5_1 -l 262144 -u 262144 -w 0 -i 100 -m cuda -g 0 \

# ------------------------------------------------------------------------------------------------------------------------------------
# UD unicast performance test
# ------------------------------------------------------------------------------------------------------------------------------------
run-ud_unicast: ib_perf_ud_unicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -- /app/ib_tests/ib_perf_ud_unicast -d mlx5_0 -w 0 -i 100 -m $(MEM_TYPE) -g 2 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host snail02:1 $(FLAGS) -- /app/ib_tests/ib_perf_ud_unicast -d mlx5_1 -w 0 -i 100 -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \

run-ud_unicast: ib_perf_ud_unicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -- /app/ib_tests/ib_perf_ud_unicast -d mlx5_0 -w 0 -i 100 -m $(MEM_TYPE) -g 2 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host tvm02:1 $(FLAGS) -- /app/ib_tests/ib_perf_ud_unicast -d mlx5_1 -w 0 -i 100 -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \

test-ud_unicast: ib_perf_ud_unicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_ud_unicast -d mlx5_0 -l 64 -u 64 -w 1 -i 2 -m $(MEM_TYPE) -g 2 $(IMM_FLAGS) \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_ud_unicast -d mlx5_1 -l 64 -u 64 -w 1 -i 2 -m $(MEM_TYPE) -g 3 $(IMM_FLAGS) \

# ------------------------------------------------------------------------------------------------------------------------------------
# RC unicast performance test
# ------------------------------------------------------------------------------------------------------------------------------------
run-rc_unicast: ib_perf_rc_unicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -- /app/ib_tests/ib_perf_rc_unicast -d mlx5_0 -w 0 -i 100 -m $(MEM_TYPE) -g 2 -l 1024 -u 33554432 $(IMM_FLAGS) \
	: -n 1 --host tvm02:1 $(FLAGS) -- /app/ib_tests/ib_perf_rc_unicast -d mlx5_1 -w 0 -i 100 -m $(MEM_TYPE) -g 0 -l 1024 -u 33554432 $(IMM_FLAGS) \

test-rc_unicast: ib_perf_rc_unicast
	mpirun \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_rc_unicast -d mlx5_0 -l 8192 -u 8192 -w 1 -i 2 -m $(MEM_TYPE) -g 2 $(IMM_FLAGS) \
	: -n 1 --host snail01:1 $(FLAGS) -x LOG_LEVEL=2 -- /app/ib_tests/ib_perf_rc_unicast -d mlx5_1 -l 8192 -u 8192 -w 1 -i 2 -m $(MEM_TYPE) -g 3 $(IMM_FLAGS) \

# ------------------------------------------------------------------------------------------------------------------------------------
# Help target
# ------------------------------------------------------------------------------------------------------------------------------------
help:
	@echo "Available targets:"
	@echo "  all                       - Build all targets"
	@echo "  ib_perf_multicast         - Build multicast performance test"
	@echo "  ib_perf_ud_unicast        - Build UD unicast performance test"
	@echo "  ib_perf_rc_unicast        - Build RC unicast performance test"
	@echo "  run-mcast                 - Run multicast performance test"
	@echo "  run-ud_unicast            - Run UD unicast performance test"
	@echo "  run-rc_unicast            - Run RC unicast performance test"
	@echo "  test-mcast                - Test multicast performance test"
	@echo "  test-ud_unicast           - Test UD unicast performance test"
	@echo "  test-rc_unicast           - Test RC unicast performance test"
	@echo "  clean                     - Clean build artifacts"
	@echo "  help                      - Show this help"

# Clean target
clean:
	rm -f $(PERF_OBJECTS) $(PERF_TARGETS)

# Flags for multi-host execution
FLAGS = --mca plm_rsh_args "-p 12345"

.PHONY: all clean perf-host perf-cuda test-host test-cuda help test-mcast-cuda-nsys test-ud_unicast-cuda-nsys test-rc_unicast-cuda-nsys
