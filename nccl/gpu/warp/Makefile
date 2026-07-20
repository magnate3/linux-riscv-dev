NVCC = nvcc
#NVCC_FLAGS = -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr  -O3  -lcublas  -lcuda -lcudadevrt --relocatable-device-code true
NVCC_FLAGS =   -O3  -lcublas  -lcuda -lcudadevrt --relocatable-device-code true

CU_APPS = simpleDivergence  demo thread  sgemmV2 comm 
#CU_APPS = simpleDivergence  demo thread gmem_bw_sim

all: $(CU_APPS) $(C_APPS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(CU_APPS) $(C_APPS)
