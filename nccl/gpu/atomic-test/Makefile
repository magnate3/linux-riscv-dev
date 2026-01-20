NVCC = nvcc
NVCC_FLAGS = -arch=sm_89 --expt-extended-lambda --expt-relaxed-constexpr  -O3  -lcublas  -lcuda -lcudadevrt --relocatable-device-code true -I/usr/local/cuda/include

CU_APPS = atom atom2   atom3 atom4 main   sample2  sample4  sample5 con_pro con_pro2 con_pro3
#CU_APPS = simpleDivergence  demo thread gmem_bw_sim

all: $(CU_APPS) $(C_APPS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(CU_APPS) $(C_APPS)
