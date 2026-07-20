#ifndef ALLREDUCE_H
#define ALLREDUCE_H

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include "cublas.h"

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

__global__ void add_vec(double *a, const double *b, int N);
__global__ void max_vec(double *a, const double *b, int N);

ncclResult_t allred_ring(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_ring_aware(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_ring_pipelined(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_butterfly(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_ring_pipelined2(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_ring_pipe_n(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int pipeline);
ncclResult_t allred_ring_pipe_max(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_tree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_dbtree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_custom(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_custom2(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int pipeline);
ncclResult_t allred_custom_seg(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int pipeline);
ncclResult_t allred_custom3(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t allred_custom4(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
#endif // ALLREDUCE_H

