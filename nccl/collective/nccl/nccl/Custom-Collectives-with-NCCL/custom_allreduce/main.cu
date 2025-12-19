#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include "cublas.h"
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

    setbuf(stdout, NULL);

    int loop_count = 16;
    int bytesize = 26;
    srand(time(NULL) + rank);
    
    // //Ring
    // if(rank == 0)
    //     std::cout<<std::endl  <<"Ring" << std::endl;

    // for(int i=0; i<=bytesize; i++){

    //     long int N = 1 << i;

    //     // Allocate memory for A on CPU
    //     double *A = (double*)malloc(N*sizeof(double));
    //     double *d_A;
    //     cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
    //     double *d_B;
    //     cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
    //     double *B = (double*)malloc(N*sizeof(double));

    //     // Warmups
    //     for(int j=1; j<=5; j++){
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
    //         NCCLCHECK(allred_ring(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
    //     }

    //     double times[loop_count];

    //     double start_time, stop_time, elapsed_time;

    //     for(int j=0; j<loop_count; j++){
    //         //create random data and copy to device
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

    //         //synchronize
    //         cudaDeviceSynchronize();
    //         cudaStreamSynchronize(s);
    //         MPI_Barrier(MPI_COMM_WORLD);   

    //         //start timing
    //         start_time = MPI_Wtime();
    //         NCCLCHECK(allred_ring(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
    //         cudaStreamSynchronize(s);
    //         stop_time = MPI_Wtime();
    //         times[j] = stop_time - start_time;
            
    //     }
    
    // }
    
    //Ring
    if(rank == 0)
        std::cout<<std::endl  <<"Seg4" << std::endl;

    for(int i=0; i<=bytesize; i++){

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
        double *B = (double*)malloc(N*sizeof(double));

        // Warmups
        for(int j=1; j<=5; j++){
            for(int i=0; i<N; i++){
                A[i] = (double)rand()/(double)RAND_MAX;
            }
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            NCCLCHECK(allred_ring_pipe_max(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
        }

        double times[loop_count];

        double start_time, stop_time, elapsed_time;

        for(int j=0; j<loop_count; j++){
            //create random data and copy to device
            for(int i=0; i<N; i++){
                A[i] = (double)rand()/(double)RAND_MAX;
            }
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

            //synchronize
            cudaDeviceSynchronize();
            cudaStreamSynchronize(s);
            MPI_Barrier(MPI_COMM_WORLD);   

            //start timing
            start_time = MPI_Wtime();
            NCCLCHECK(allred_ring_pipe_max(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[j] = stop_time - start_time;
            
        }
        

        //calculate time statistics
        double sum = 0;
        for(int t=0; t<loop_count; t++){
            sum += times[t];
        }
        double average = sum / loop_count;
        
        long int num_B = 8*N;
        long int B_in_GB = 1 << 30;
        double num_GB = (double)num_B / (double)B_in_GB;
        double avg_time_per_transfer = sum / ((double)loop_count);

        if(rank == 0) printf("Message size (B): %10li\n", num_B);
        if(rank == 0){
            for (int i = 0; i < loop_count; i++){
                printf("%f\n", times[i]);
            }
        }
        
        cudaErrorCheck( cudaFree(d_A) );
        cudaErrorCheck( cudaFree(d_B) );
        free(A);
        free(B);
    }

   
    // //Tree
    // if(rank == 0)
    //     std::cout <<std::endl <<"DBTree" << std::endl;

    // for(int i=0; i<=bytesize; i++){

    //     long int N = 1 << i;

    //     // Allocate memory for A on CPU
    //     double *A = (double*)malloc(N*sizeof(double));
    //     double *d_A;
    //     cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
    //     double *d_B;
    //     cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
    //     double *B = (double*)malloc(N*sizeof(double));

    //     // Warmups
    //     for(int j=1; j<=5; j++){
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
    //         NCCLCHECK(allred_dbtree(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
    //     }

    //     double times[loop_count];

    //     double start_time, stop_time, elapsed_time;

    //     for(int j=0; j<loop_count; j++){
    //         //create random data and copy to device
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

    //         //synchronize
    //         cudaDeviceSynchronize();
    //         cudaStreamSynchronize(s);
    //         MPI_Barrier(MPI_COMM_WORLD);   

    //         //start timing
    //         start_time = MPI_Wtime();
    //         NCCLCHECK(allred_dbtree(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
    //         cudaStreamSynchronize(s);
    //         stop_time = MPI_Wtime();
    //         times[j] = stop_time - start_time;
            
    //     }
        

    //     //calculate time statistics
    //     double sum = 0;
    //     for(int t=0; t<loop_count; t++){
    //         sum += times[t];
    //     }
    //     double average = sum / loop_count;
        
    //     long int num_B = 8*N;
    //     long int B_in_GB = 1 << 30;
    //     double num_GB = (double)num_B / (double)B_in_GB;
    //     double avg_time_per_transfer = sum / ((double)loop_count);

    //     if(rank == 0) printf("Message size (B): %10li\n", num_B);
    //     if(rank == 0){
    //         for (int i = 0; i < loop_count; i++){
    //             printf("%f\n", times[i]);
    //         }
    //     }
        
    //     cudaErrorCheck( cudaFree(d_A) );
    //     cudaErrorCheck( cudaFree(d_B) );
    //     free(A);
    //     free(B);
    // }


    // //NCCL
    // if(rank == 0)
    //     std::cout <<std::endl <<"NCCL" << std::endl;

    // for(int i=0; i<=bytesize; i++){

    //     long int N = 1 << i;

    //     // Allocate memory for A on CPU
    //     double *A = (double*)malloc(N*sizeof(double));
    //     double *d_A;
    //     cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
    //     double *d_B;
    //     cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
    //     double *B = (double*)malloc(N*sizeof(double));

    //     // Warmups
    //     for(int j=1; j<=5; j++){
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
    //         NCCLCHECK(ncclAllReduce(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
    //     }

    //     double times[loop_count];

    //     double start_time, stop_time, elapsed_time;

    //     for(int j=0; j<loop_count; j++){
    //         //create random data and copy to device
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

    //         //synchronize
    //         cudaDeviceSynchronize();
    //         cudaStreamSynchronize(s);
    //         MPI_Barrier(MPI_COMM_WORLD);   

    //         //start timing
    //         start_time = MPI_Wtime();
    //         NCCLCHECK(ncclAllReduce(d_A, d_B, N, ncclDouble, ncclSum, comm, s));
    //         cudaStreamSynchronize(s);
    //         stop_time = MPI_Wtime();
    //         times[j] = stop_time - start_time;
            
    //     }
        

    //     //calculate time statistics
    //     double sum = 0;
    //     for(int t=0; t<loop_count; t++){
    //         sum += times[t];
    //     }
    //     double average = sum / loop_count;
        
    //     long int num_B = 8*N;
    //     long int B_in_GB = 1 << 30;
    //     double num_GB = (double)num_B / (double)B_in_GB;
    //     double avg_time_per_transfer = sum / ((double)loop_count);

    //     if(rank == 0) printf("Message size (B): %10li\n", num_B);
    //     if(rank == 0){
    //         for (int i = 0; i < loop_count; i++){
    //             printf("%f\n", times[i]);
    //         }
            
    //     }
        
    //     cudaErrorCheck( cudaFree(d_A) );
    //     cudaErrorCheck( cudaFree(d_B) );
    //     free(A);
    //     free(B);
    // }

    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
        
}
