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

__global__ void add_vec(double *a, const double *b, int N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
   if(id < N) a[id] = a[id] + b[id];
}

__global__ void max_vec(double *a, const double *b, int N)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < N) {
        if(b[id] >= a[id]){
            a[id] =  b[id];
        }
    }
}



ncclResult_t allred_ring(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    //"parallel ring" implementation for reduce 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(count) / thr_per_blk );
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    int next = (rank+1)%size;
    int last = ((rank-1)%size+size)%size;

    double one = 1.0;
    cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    
    for (int i = 0; i < size-1; i++){
        cudaMemcpy(tempbuff, recvbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        if (rank%2 == 0){
            
            ncclSend(tempbuff, count,datatype, next,comm,stream);
            ncclRecv(recvbuff, count,datatype, last,comm,stream);
            
            //cublasDaxpy((int)count, one, sendbuff, 1, recvbuff, 1);
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff,sendbuff,count);
        }else
        {
            
            ncclRecv(recvbuff, count,datatype, last,comm,stream);
            ncclSend(tempbuff, count,datatype, next,comm,stream);
            
            //cublasDaxpy((int)count, one, sendbuff, 1, recvbuff , 1);
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff,sendbuff,count);
        }
    }
    
    //cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(tempbuff);
    return ncclSuccess;
    
}



ncclResult_t allred_ring_cublas(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    //"parallel ring" implementation for reduce 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(count) / thr_per_blk );
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    int next = (rank+1)%size;
    int last = ((rank-1)%size+size)%size;

    double one = 1.0;
    cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    cublasHandle_t handle;
    for (int i = 0; i < size-1; i++){
        cudaMemcpy(tempbuff, recvbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        if (rank%2 == 0){
            
            ncclSend(tempbuff, count,datatype, next,comm,stream);
            ncclRecv(recvbuff, count,datatype, last,comm,stream);
            
            cublasDaxpy((int)count, one, sendbuff, 1, recvbuff, 1);
            //add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff,sendbuff,count);
        }else
        {
            
            ncclRecv(recvbuff, count,datatype, last,comm,stream);
            ncclSend(tempbuff, count,datatype, next,comm,stream);
            
            cublasDaxpy((int)count, one, sendbuff, 1, recvbuff , 1);
            //add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff,sendbuff,count);
        }
    }
    
    //cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(tempbuff);
    return ncclSuccess;
    
}


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
    cudaErrorCheck(cudaSetDevice(rank));
    cudaErrorCheck(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));


    int loop_count = 200;
    int bytesize = 29;
    srand(time(NULL) + rank);


    // Test custom kernel speed
    std::cout << "Custom Kernel" << std::endl;
    for(int i=0; i<=bytesize;i++){
        long int N = 1 << i;
        double *A = (double*)malloc(N*sizeof(double));
        double *B = (double*)malloc(N*sizeof(double));
        double *d_A;
        double *d_B;
        double times[loop_count];
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
            B[i] = (double)rand()/(double)RAND_MAX;
        }
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
        cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice) );
        double start_time, stop_time, elapsed_time;
        int thr_per_blk = 256;
        int blk_in_grid = ceil( float(N) / thr_per_blk );
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);
        for(int j=0; j<=loop_count; j++){
            start_time = MPI_Wtime();
            add_vec<<<blk_in_grid, thr_per_blk >>>(d_A, d_B, N);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[j] = stop_time - start_time;
        }


        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        // double num_GB = (double)num_B / (double)B_in_GB;
        // double avg_time_per_transfer = elapsed_time / ((double)loop_count);

        if(rank == 0) printf("Message size (B): %10li\n", num_B);
        if(rank == 0){
            for (int j = 0; j < loop_count; j++){
                printf("%.17g\n", times[j]);
            }
        }
        
        
        // if(rank == 0) printf("Message size (B): %10li, Transfer Time (s): %15.9f, Effective Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, size*num_GB/avg_time_per_transfer );
        cudaErrorCheck( cudaFree(d_A) );
        cudaErrorCheck( cudaFree(d_B) );
        free(A);
        free(B);
    }

    //Test cuBlas library speed
    std::cout << std::endl;
    std::cout << "cuBlas" << std::endl;
    for(int i=0;i<=bytesize;i++){
        long int N = 1 << i;
        double *A = (double*)malloc(N*sizeof(double));
        double *B = (double*)malloc(N*sizeof(double));
        double *d_A;
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
            B[i] = (double)rand()/(double)RAND_MAX;
        }
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
        cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice) );
        double start_time, stop_time, elapsed_time;
        double one = 1.0;
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);
        
        double times[loop_count];
        for(int j=0; j<=loop_count; j++){
            start_time = MPI_Wtime();
            cublasDaxpy(N, one, d_A, 1, d_B, 1);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[j] = stop_time - start_time;
        }
        
        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        // double num_GB = (double)num_B / (double)B_in_GB;
        // double avg_time_per_transfer = elapsed_time / ((double)loop_count);

        if(rank == 0) printf("Message size (B): %10li\n", num_B);
        if(rank == 0){
            for (int j = 0; j < loop_count; j++){
                printf("%.17g\n", times[j]);
            }
        }
        // if(rank == 0) printf("Message size (B): %10li, Transfer Time (s): %15.9f, Effective Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, size*num_GB/avg_time_per_transfer );
        cudaErrorCheck( cudaFree(d_A) );
        cudaErrorCheck( cudaFree(d_B) );
        free(A);
        free(B);
    }

    
    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
        
}
