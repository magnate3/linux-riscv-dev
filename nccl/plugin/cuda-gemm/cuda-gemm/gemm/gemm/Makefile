NVCC_FLAGS = -std=c++20 -O3 -DNDEBUG -w
NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -arch=sm_90a
NVCC_LDFLAGS =   -lcudart -lcublas -lcublasLt -lcuda  -lcupti # --keep # -lineinfo

gemm:  gemm.cu
	nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) $< -o $@
