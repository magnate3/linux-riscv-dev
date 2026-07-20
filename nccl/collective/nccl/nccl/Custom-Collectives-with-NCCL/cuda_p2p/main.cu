#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

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

	MPI_Init(&argc,&argv);
	int size;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    // Map MPI ranks to GPUs
    cudaErrorCheck(cudaSetDevice(0));
	
    int loop_count = 100;
    int bytesize = 29;

    ncclUniqueId id;
    ncclComm_t comm;
    cudaStream_t s;
    //get NCCL unique ID at rank 0 and broadcast it to all others
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    cudaErrorCheck(cudaStreamCreate(&s));

    
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    // cudaErrorCheck(cudaDeviceEnablePeerAccess((rank+1) % num_devices,0));


    //MPI CPU-to-CPU
    if(rank == 0)
        std::cout <<std::endl <<"MPI CPU-to-CPU" << std::endl;

    for(int i=0; i<=bytesize; i++){

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *B = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        for(int i=0; i<N; i++){
            B[i] = (double)rand()/(double)RAND_MAX;
        }

        MPI_Request request1;
        MPI_Request request2;

        // Warm-up loop
        for(int i=0; i<5; i++){
            if(rank == 0){
				MPI_Isend(A, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request1);
				MPI_Irecv(B, N, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
			else if(rank == 1){
				MPI_Irecv(A, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request1);
				MPI_Isend(B, N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
        }
        
        double start_time, stop_time, elapsed_time;
        
        double times[loop_count];

        for(int j=0; j<loop_count; j++){
            start_time = MPI_Wtime();
            if(rank == 0){
				MPI_Isend(A, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request1);
				MPI_Irecv(B, N, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
			else if(rank == 1){
				MPI_Irecv(A, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request1);
				MPI_Isend(B, N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
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
            for (int j = 0; j < loop_count; j++){
                printf("%.17g\n", times[j]);
            }
        }

        // long int num_B = 8*N;
        // long int B_in_GB = 1 << 30;
        // double num_GB = (double)num_B / (double)B_in_GB;
        // double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        // if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

        free(A);
        free(B);
    }

    // //CUDA staged MPI
    // if(rank == 0)
    //     std::cout <<std::endl <<"CUDA staged MPI" << std::endl;
    // for(int i=0; i<=bytesize; i++){

    //     int N = 1 << i;

    //     // Allocate memory for A on CPU
    //     double *A = (double*)malloc(N*sizeof(double));
    //     // Initialize all elements of A to random values
    //     for(int i=0; i<N; i++){
    //         A[i] = (double)rand()/(double)RAND_MAX;
    //     }

    //     double *d_A;
	// 	cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
	// 	cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

        

    //     // Warm-up loop
    //     for(int i=1; i<=5; i++){
    //         if(rank == 0){
	// 			cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
	// 			MPI_Send(A, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
	// 			MPI_Recv(A, N, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// 			cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
	// 		}
	// 		else if(rank == 1){
	// 			MPI_Recv(A, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// 			cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
	// 			cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
	// 			MPI_Send(A, N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	// 		}
    //     }
        
    //     double start_time, stop_time, elapsed_time;
    //     start_time = MPI_Wtime();

    //     for(int i=1; i<=loop_count; i++){
    //         if(rank == 0){
	// 			cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
	// 			MPI_Send(A, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD);
	// 			MPI_Recv(A, N, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// 			cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
	// 		}
	// 		else if(rank == 1){
	// 			MPI_Recv(A, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
	// 			cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
	// 			cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
	// 			MPI_Send(A, N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
	// 		}
    //     }

    //     stop_time = MPI_Wtime();
    //     elapsed_time = stop_time - start_time;


    //     long int num_B = 8*N;
    //     long int B_in_GB = 1 << 30;
    //     double num_GB = (double)num_B / (double)B_in_GB;
    //     double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

    //     if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    //     cudaErrorCheck( cudaFree(d_A) );
    //     free(A);
    // }


    //CUDA-aware MPI
    if(rank == 0)
        std::cout <<std::endl <<"CUDA-aware MPI" << std::endl;
    
    for(int i=0; i<=bytesize; i++){

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *B = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        for(int i=0; i<N; i++){
            B[i] = (double)rand()/(double)RAND_MAX;
        }
        double *d_A;
		cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
        cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice) );

        MPI_Request request1;
        MPI_Request request2;

        // Warm-up loop
        for(int i=0; i<5; i++){
            if(rank == 0){
				MPI_Isend(d_A, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request1);
				MPI_Irecv(d_B, N, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
			else if(rank == 1){
				MPI_Irecv(d_A, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request1);
				MPI_Isend(d_B, N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
        }
        
        double start_time, stop_time;

        double times[loop_count];
        for(int j=0; j<loop_count; j++){
            start_time = MPI_Wtime();
            if(rank == 0){
				MPI_Isend(d_A, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &request1);
				MPI_Irecv(d_B, N, MPI_DOUBLE, 1, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
			else if(rank == 1){
				MPI_Irecv(d_A, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &request1);
				MPI_Isend(d_B, N, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &request2);
                MPI_Wait(&request1, MPI_STATUS_IGNORE);
                MPI_Wait(&request2, MPI_STATUS_IGNORE);
			}
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
            for (int j = 0; j < loop_count; j++){
                printf("%.17g\n", times[j]);
            }
        }

        // long int num_B = 8*N;
        // long int B_in_GB = 1 << 30;
        // double num_GB = (double)num_B / (double)B_in_GB;
        // double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        // if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );
        cudaErrorCheck( cudaFree(d_B) );
        cudaErrorCheck( cudaFree(d_A) );
        free(A);
        free(B);
    }
    



    //NCCL Send/Recv
    if(rank == 0)
        std::cout <<std::endl <<"NCCL" << std::endl;

    for(int i=0; i<=bytesize; i++){

        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        double *B = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        for(int i=0; i<N; i++){
            B[i] = (double)rand()/(double)RAND_MAX;
        }
        double *d_A;
		cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );
        double *d_B;
        cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
        cudaErrorCheck( cudaMemcpy(d_B, B, N*sizeof(double), cudaMemcpyHostToDevice) );

        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);

        // Warm-up loop
        for(int i=0; i<5; i++){
            if(rank == 0){
                ncclGroupStart();
                ncclSend(d_A, N, ncclDouble, 1, comm, s);
                ncclRecv(d_B, N, ncclDouble, 1, comm, s);
                ncclGroupEnd();
			}
			else if(rank == 1){
                ncclGroupStart();
                ncclRecv(d_A, N, ncclDouble, 0, comm, s);
                ncclSend(d_B, N, ncclDouble, 0, comm, s);
                ncclGroupEnd();
			}
            cudaStreamSynchronize(s);
        }
        
        double start_time, stop_time, elapsed_time;
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);
        
        double times[loop_count];
        for(int j=0; j<loop_count; j++){
            start_time = MPI_Wtime();
            if(rank == 0){
                ncclGroupStart();
                ncclSend(d_A, N, ncclDouble, 1, comm, s);
                ncclRecv(d_B, N, ncclDouble, 1, comm, s);
                ncclGroupEnd();
			}
			else if(rank == 1){
                ncclGroupStart();
                ncclRecv(d_A, N, ncclDouble, 0, comm, s);
                ncclSend(d_B, N, ncclDouble, 0, comm, s);
                ncclGroupEnd();
			}
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
            for (int j = 0; j < loop_count; j++){
                printf("%.17g\n", times[j]);
            }
        }

        //Verify  Results
        

        // long int num_B = 8*N;
        // long int B_in_GB = 1 << 30;
        // double num_GB = (double)num_B / (double)B_in_GB;
        // double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

        // if(rank == 0) printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

        
        cudaErrorCheck( cudaFree(d_A) );
        cudaErrorCheck( cudaFree(d_B) );
        free(A);
        free(B);
    }



    

	
    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
	if(rank == 0)
        std::cout << std::endl << std::endl;
	
	
	
	MPI_Finalize();
    return 0;
}
