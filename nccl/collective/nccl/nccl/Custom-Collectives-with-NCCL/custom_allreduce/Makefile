CXX=nvcc
#NCCLPATH = /user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/nccl-2.20.3-1-l2d55ru4lupkbzxnqm2b5nrvlgnt3v3s
#MPIPATH = /user-environment/linux-sles15-neoverse_v2/gcc-13.2.0/cray-mpich-8.1.30-mrkhzmus5xo47kryelzzh3mkxu3kfv6j
NCCLPATH = /user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd
MPIPATH = /user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/cray-mpich-8.1.30-wb5peugemrg2ebx7psp2iz2abmqy3rgz
CXXFLAGS=-O3 -arch=sm_90 -std=c++11 -I${MPIPATH}/include -I${CUDA_HOME}/include -I${NCCLPATH}/include -lcublas

libs = -L${MPIPATH}/lib -lmpi
libs2 = -L${CUDA_HOME}/lib64 -lcudart
libs3 = -L${NCCLPATH}/lib -lnccl


all: main

main: main.o allreduce.o
	$(CXX) $(CXXFLAGS) $(libs) $(libs2) $(libs3) -lcublas main.o allreduce.o -o main

main.o: main.cu
	$(CXX) $(CXXFLAGS) -c main.cu

allreduce.o: ../allreduce.cu
	$(CXX) $(CXXFLAGS) -c ../allreduce.cu

clean:
	rm -rf main main.o allreduce.o

.PHONY: all clean

