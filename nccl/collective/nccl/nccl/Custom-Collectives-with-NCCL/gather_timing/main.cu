#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>
#include "../gather.h"
#include "../scatter.h"

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

    int loop_count = 20;
    int bytesize = 26;
    srand(time(NULL) + rank);


    // if(rank == 0)
    //     std::cout <<std::endl <<"ScatterPrimitive" << std::endl;

    // for(int i=0; i<=bytesize; i++){

    //     int N = 1 << i;

    //     // Allocate memory for A on CPU
    //     double *A = (double*)malloc(N*sizeof(double)*size);
    //     double *d_A;
    //     cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)*size) );
    //     double *d_B;
    //     cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
    //     double *B = (double*)malloc(N*sizeof(double));

    //     // Warmups
    //     for(int j=1; j<=5; j++){
    //         if (rank == 0) {
    //             for(int i=0; i<N*size; i++){
    //                 A[i] = (double)rand()/(double)RAND_MAX;
    //             }
    //             cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double)*size, cudaMemcpyHostToDevice) );
    //         }
    //         NCCLCHECK(scatter_primitive(d_A, d_B, N, ncclDouble, 0, comm, s));
    //     }

    //     double times[loop_count];

    //     double start_time, stop_time, elapsed_time;

    //     for(int j=0; j<loop_count; j++){
    //         //create random data and copy to device
    //         if (rank == 0) {
    //             for(int i=0; i<N*size; i++){
    //                 A[i] = (double)rand()/(double)RAND_MAX;
    //             }
    //             cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double)*size, cudaMemcpyHostToDevice) );
    //         }

    //         //synchronize
    //         cudaDeviceSynchronize();
    //         cudaStreamSynchronize(s);
    //         MPI_Barrier(MPI_COMM_WORLD);   

    //         //start timing
    //         start_time = MPI_Wtime();
    //         NCCLCHECK(scatter_primitive(d_A, d_B, N, ncclDouble, 0, comm, s));
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
    //     free(A);
    //     cudaErrorCheck( cudaFree(d_B) );
    //     free(B);
    // }

    // if(rank == 0)
    //     std::cout <<std::endl <<"ScatterTree" << std::endl;

    // for(int i=0; i<=bytesize; i++){

    //     int N = 1 << i;

    //     // Allocate memory for A on CPU
    //     double *A = (double*)malloc(N*sizeof(double)*size);
    //     double *d_A;
    //     cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)*size) );
    //     double *d_B;
    //     cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
    //     double *B = (double*)malloc(N*sizeof(double));

    //     // Warmups
    //     for(int j=1; j<=5; j++){
    //         if (rank == 0) {
    //             for(int i=0; i<N*size; i++){
    //                 A[i] = (double)rand()/(double)RAND_MAX;
    //             }
    //             cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double)*size, cudaMemcpyHostToDevice) );
    //         }
    //         NCCLCHECK(scatter_tree(d_A, d_B, N, ncclDouble, 0, comm, s));
    //     }

    //     double times[loop_count];

    //     double start_time, stop_time, elapsed_time;

    //     for(int j=0; j<loop_count; j++){
    //         //create random data and copy to device
    //         if (rank == 0) {
    //             for(int i=0; i<N*size; i++){
    //                 A[i] = (double)rand()/(double)RAND_MAX;
    //             }
    //             cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double)*size, cudaMemcpyHostToDevice) );
    //         }

    //         //synchronize
    //         cudaDeviceSynchronize();
    //         cudaStreamSynchronize(s);
    //         MPI_Barrier(MPI_COMM_WORLD);   

    //         //start timing
    //         start_time = MPI_Wtime();
    //         NCCLCHECK(scatter_tree(d_A, d_B, N, ncclDouble, 0, comm, s));
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
    //     free(A);
    //     cudaErrorCheck( cudaFree(d_B) );
    //     free(B);
    // }

    // //MPI Gather
    // if(rank == 0)
    //     std::cout <<std::endl <<"MPI" << std::endl;

    // for(int i=0; i<=bytesize; i++){

    //     int N = 1 << i;

    //     // Allocate memory
    //     double *A = (double*)malloc(N * sizeof(double));
    //     double *B = (double*)malloc(N * sizeof(double) * size);
    //     if (A == NULL || B == NULL) {
    //         fprintf(stderr, "Memory allocation failed\n");
    //         MPI_Abort(MPI_COMM_WORLD, 1);
    //     }
    //     double *d_A;
    //     cudaErrorCheck(cudaMalloc(&d_A, N * sizeof(double)));
    //     double *d_B;
    //     cudaErrorCheck(cudaMalloc(&d_B, N * sizeof(double) * size));


    //     // Warmups
    //     for(int j=1; j<=5; j++){
    //         for(int i=0; i<N; i++){
    //             A[i] = (double)rand()/(double)RAND_MAX;
    //         }
    //         cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
    //         MPI_Gather(d_A, N, MPI_DOUBLE, d_B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
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
    //         MPI_Gather(d_A, N, MPI_DOUBLE, d_B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //         cudaStreamSynchronize(s);
    //         stop_time = MPI_Wtime();
    //         times[j] = stop_time - start_time;
    //         //if(rank==0) printf("Time: %f\n", times[j]);


    //         // //Verify Results
    //         // double *ver = (double*)malloc(N*sizeof(double));
    //         // cudaErrorCheck( cudaMemcpy(ver, d_B, N*sizeof(double)*size, cudaMemcpyDeviceToHost) );
    //         // MPI_Gather(A, N, MPI_DOUBLE, B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //         // if (rank == 0 && *B != *ver){
    //         //     std::cout << "Allreduce is " << *ver << " instead of "<< *B << std::endl;   
    //         // }
    //         // free(ver);
    //     }
        

    //     //calculate time statistics
    //     double sum = 0;
    //     for(int t=0; t<loop_count; t++){
    //         sum += times[t];
    //     }
    //     double average = sum / loop_count;
    //     double variance = 0;
    //     for(int t=0; t<loop_count; t++){
    //         variance += (times[t] - average) * (times[t] - average);
    //     }
    //     variance = variance / loop_count;
    //     double std_dev = sqrt(variance);
    //     double min_time = times[0];
    //     double max_time = times[0];

    //     for(int t = 1; t < loop_count; t++) {
    //         if(times[t] < min_time) {
    //             min_time = times[t];
    //         }
    //         if(times[t] > max_time) {
    //             max_time = times[t];
    //         }
    //     }
        
    //     // if (rank == 0) {
    //     //     printf("\n");
    //     //     printf("Message size (B): %10li\n", 8*N);
    //     //     printf("Average Time (s): %15.9f\n", average);
    //     //     printf("Min Time (s): %15.9f\n", min_time);
    //     //     printf("Max Time (s): %15.9f\n", max_time);
    //     //     printf("Std Dev: %15.9f\n", std_dev);
    //     //     printf("\n");
    //     // }
        
    //     long int num_B = 8*N;
    //     long int B_in_GB = 1 << 30;
    //     double num_GB = (double)num_B / (double)B_in_GB;
    //     double avg_time_per_transfer = sum / ((double)loop_count);

    //     if(rank == 0) printf("Message size (B): %10li, Transfer Time (s): %15.9f, Effective Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, size*num_GB/avg_time_per_transfer );


    //     cudaErrorCheck( cudaFree(d_A) );
    //     free(A);
    //     cudaErrorCheck( cudaFree(d_B) );
    //     free(B);
    // }


if(rank == 0)
        std::cout <<std::endl <<"Custom_Prim" << std::endl;

    for(int i=0; i<=bytesize; i++){

        int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)*size) );
        double *B = (double*)malloc(N*sizeof(double)*size);

        // Warmups
        for(int j=1; j<=5; j++){
            for(int i=0; i<N; i++){
                A[i] = (double)rand()/(double)RAND_MAX;
            }
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            NCCLCHECK(gather_custom(d_A, d_B, N, ncclDouble, 0, comm, s));
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
            NCCLCHECK(gather_custom(d_A, d_B, N, ncclDouble, 0, comm, s));
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
        free(A);
        cudaErrorCheck( cudaFree(d_B) );
        free(B);
    }

    if(rank == 0)
        std::cout <<std::endl <<"Custom_Dist" << std::endl;

    for(int i=0; i<=bytesize; i++){

        int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)*size) );
        double *B = (double*)malloc(N*sizeof(double)*size);

        // Warmups
        for(int j=1; j<=5; j++){
            for(int i=0; i<N; i++){
                A[i] = (double)rand()/(double)RAND_MAX;
            }
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            NCCLCHECK(gather_custom2(d_A, d_B, N, ncclDouble, 0, comm, s));
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
            NCCLCHECK(gather_custom2(d_A, d_B, N, ncclDouble, 0, comm, s));
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
        free(A);
        cudaErrorCheck( cudaFree(d_B) );
        free(B);
    }

    if(rank == 0)
        std::cout <<std::endl <<"Custom_Tree" << std::endl;

    for(int i=0; i<=bytesize; i++){

        int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)*size) );
        double *B = (double*)malloc(N*sizeof(double)*size);

        // Warmups
        for(int j=1; j<=5; j++){
            for(int i=0; i<N; i++){
                A[i] = (double)rand()/(double)RAND_MAX;
            }
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            NCCLCHECK(gather_custom3(d_A, d_B, N, ncclDouble, 0, comm, s));
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
            NCCLCHECK(gather_custom3(d_A, d_B, N, ncclDouble, 0, comm, s));
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
        free(A);
        cudaErrorCheck( cudaFree(d_B) );
        free(B);
    }

// if(rank == 0)
//         std::cout <<std::endl <<"GatherTree" << std::endl;

//     for(int i=0; i<=bytesize; i++){

//         int N = 1 << i;

//         // Allocate memory for A on CPU
//         double *A = (double*)malloc(N*sizeof(double));
//         double *d_A;
//         cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
//         double *d_B;
//         cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)*size) );
//         double *B = (double*)malloc(N*sizeof(double)*size);

//         // Warmups
//         for(int j=1; j<=5; j++){
//             for(int i=0; i<N; i++){
//                 A[i] = (double)rand()/(double)RAND_MAX;
//             }
//             cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
//             NCCLCHECK(gather_tree(d_A, d_B, N, ncclDouble, 0, comm, s));
//         }

//         double times[loop_count];

//         double start_time, stop_time, elapsed_time;

//         for(int j=0; j<loop_count; j++){
//             //create random data and copy to device
//             for(int i=0; i<N; i++){
//                 A[i] = (double)rand()/(double)RAND_MAX;
//             }
//             cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

//             //synchronize
//             cudaDeviceSynchronize();
//             cudaStreamSynchronize(s);
//             MPI_Barrier(MPI_COMM_WORLD);   

//             //start timing
//             start_time = MPI_Wtime();
//             NCCLCHECK(gather_tree(d_A, d_B, N, ncclDouble, 0, comm, s));
//             cudaStreamSynchronize(s);
//             stop_time = MPI_Wtime();
//             times[j] = stop_time - start_time;
            
//         }
        

//         //calculate time statistics
//         double sum = 0;
//         for(int t=0; t<loop_count; t++){
//             sum += times[t];
//         }
//         double average = sum / loop_count;
        
//         long int num_B = 8*N;
//         long int B_in_GB = 1 << 30;
//         double num_GB = (double)num_B / (double)B_in_GB;
//         double avg_time_per_transfer = sum / ((double)loop_count);

//         if(rank == 0) printf("Message size (B): %10li\n", num_B);
//         if(rank == 0){
//             for (int i = 0; i < loop_count; i++){
//                 printf("%f\n", times[i]);
//             }
//         }

//         cudaErrorCheck( cudaFree(d_A) );
//         free(A);
//         cudaErrorCheck( cudaFree(d_B) );
//         free(B);
//     }

    
if(rank == 0)
        std::cout <<std::endl <<"ncclAllgather" << std::endl;

    for(int i=0; i<=bytesize; i++){

        int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)*size) );
        double *B = (double*)malloc(N*sizeof(double)*size);

        // Warmups
        for(int j=1; j<=5; j++){
            for(int i=0; i<N; i++){
                A[i] = (double)rand()/(double)RAND_MAX;
            }
            cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
            NCCLCHECK(ncclAllGather(d_A, d_B, N, ncclDouble,comm, s));
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
            NCCLCHECK(ncclAllGather(d_A, d_B, N, ncclDouble,comm, s));
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[j] = stop_time - start_time;
            //if(rank==0) printf("Time: %f\n", times[j]);


            // //Verify Results
            // double *ver = (double*)malloc(N*sizeof(double));
            // cudaErrorCheck( cudaMemcpy(ver, d_B, N*sizeof(double)*size, cudaMemcpyDeviceToHost) );
            // MPI_Gather(A, N, MPI_DOUBLE, B, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            // if (rank == 0 && *B != *ver){
            //     std::cout << "Allreduce is " << *ver << " instead of "<< *B << std::endl;   
            // }
            // free(ver);
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
        free(A);
        cudaErrorCheck( cudaFree(d_B) );
        free(B);
    }

    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
        
}
