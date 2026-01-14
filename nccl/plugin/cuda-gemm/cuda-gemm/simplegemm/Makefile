NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a
NVCC_LDFLAGS = -lcublas -lcuda # --keep # -lineinfo

gemm: main.cu gemm.cu pingpong.cu stmatrix.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $< -o $@

maxreg: maxreg.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $^ -o $@
