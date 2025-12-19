#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include "cublas.h"
//#include "../gather.h"
#include "../allreduce.h"

#define cudaErrorCheck(call)                                                              \
do{                                                                                       \
	cudaError_t cuErr = call;                                                             \
	if(cudaSuccess != cuErr){                                                             \
		printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
		exit(0);                                                                            \
	}                                                                                     \
}while(0)


#define NCCLCHECK(call) do {                         \
  ncclResult_t r = call;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)



int main(int argc, char** argv){

    //MPI_Initilization
	MPI_Init(&argc,&argv);
	int size;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // NCCL & CUDA Initilization
    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s;

    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    cudaErrorCheck(cudaSetDevice(0));
    cudaErrorCheck(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    
    srand(time(NULL) + rank);
    int i = 4;
    int N = 1<<i;

    //--------------REGULAR-------------
    if(rank == 0) std::cout<<"REGULAR"<<std::endl;
    // Allocate memory for Ar, B_testr, and B_verr
    double *Ar = (double*)malloc(N * sizeof(double));
    double *B_testr = (double*)malloc(N *  sizeof(double));
    double *B_verr = (double*)malloc(N *  sizeof(double));

    if (Ar == NULL || B_testr == NULL || B_verr == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int i = 0; i < N; i++) {
        Ar[i] = (double)rand() / (double)RAND_MAX;
        // Ar[i] = 1.0;
        if (rank == 0) {
            //std::cout << "A[" << i << "]: " << Ar[i] << std::endl;
        }
    }

    double *d_Ar, *d_Br;
    cudaErrorCheck(cudaMalloc(&d_Ar, N * sizeof(double)));
    cudaErrorCheck(cudaMalloc(&d_Br, N * sizeof(double)));
    cudaErrorCheck(cudaMemcpy(d_Ar, Ar, N * sizeof(double), cudaMemcpyHostToDevice));


    NCCLCHECK(allred_ring_pipe_n(d_Ar, d_Br, N, ncclDouble,ncclSum, comm, s,8));

    MPI_Allreduce(Ar, B_verr, N, MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);


    // cudaErrorCheck(cudaMemcpy(A, d_A, N * sizeof(double), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(B_testr, d_Br, N * sizeof(double), cudaMemcpyDeviceToHost));
    if (rank == 0) {
        bool func_corr = true;
        for(int i = 0; i < N ; i++) {
            printf("B_ver[%i]: %.15f \n", i, B_verr[i]);
            printf("B_test[%i]: %.15f \n", i, B_testr[i]);
            if(abs(B_verr[i] - B_testr[i]) > 0.000001) {
                func_corr = false;
                std::cout << "This one was false!" << std::endl;
            }
        }
        if (func_corr) {
            std::cout << "Function works as intended!" << std::endl;
        } else {
            std::cout << "Function doesn't work as intended :(" << std::endl;
        }
    }

    cudaFree(d_Ar);
    cudaFree(d_Br);
    free(Ar);
    free(B_testr);
    free(B_verr);

    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
        
}
