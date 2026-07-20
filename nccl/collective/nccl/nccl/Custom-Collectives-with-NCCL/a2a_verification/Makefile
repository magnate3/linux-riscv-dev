CXX=nvcc
NCCLPATH = /user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd
MPIPATH = /user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-mpich-8.1.30-wb5peugemrg2ebx7psp2iz2abmqy3rgz
CXXFLAGS=-O2 -gencode arch=compute_90,code=sm_90 -std=c++11 -I${MPIPATH}/include -I${CUDA_HOME}/include -I${NCCLPATH}/include -lcublas

libs = -L${MPIPATH}/lib -lmpi
libs2 = -L${CUDA_HOME}/lib64 -lcudart
libs3 = -L${NCCLPATH}/lib -lnccl


all: main

main: main.cu
	$(CXX) $(CXXFLAGS) $(libs) $(libs2) $(libs3) -lcublas main.cu -o main

clean:
	rm -rf main

.PHONY: all clean

