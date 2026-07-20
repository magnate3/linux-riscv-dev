#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <algorithm>

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


ncclResult_t alltoall_spreadout(double* sendbuff,size_t count1,ncclDataType_t datatype1,double* recvbuff,size_t count2,ncclDataType_t datatype2, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int count = count1;
    ncclDataType_t datatype = datatype1;

    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * size * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }

    

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*size*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

    
    
    for(int i = 0 ; i < size;i++){
        ncclGroupStart();
        int recvfrom = (rank + i)%size;
        int sendto = (rank - i +size)%size;
        
        ncclSend(sendbuff + (sendto*count), count, datatype, sendto, comm, stream);
        ncclRecv(recvbuff2 + (recvfrom*count), count, datatype, recvfrom, comm, stream);
        ncclGroupEnd();
        
    }
    if(needtofree){
        cudaMemcpy(sendbuff, recvbuff2, count * size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaFree(recvbuff2);
    }
    
    //cudaFree(recvbuff2);
    return ncclSuccess;
}

ncclResult_t alltoall_prim(double* sendbuff,size_t count1,ncclDataType_t datatype1,double* recvbuff,size_t count2,ncclDataType_t datatype2, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int count = count1;
    ncclDataType_t datatype = datatype1;

    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * size * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }

    

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*size*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

    
    ncclGroupStart();
    for(int i = 0 ; i < size;i++){
        
        int recvfrom = (rank + i)%size;
        int sendto = (rank - i +size)%size;
        
        ncclSend(sendbuff + (sendto*count), count, datatype, sendto, comm, stream);
        ncclRecv(recvbuff2 + (recvfrom*count), count, datatype, recvfrom, comm, stream);        
    }
    ncclGroupEnd();

    if(needtofree){
        cudaMemcpy(sendbuff, recvbuff2, count * size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaFree(recvbuff2);
    }
    
    //cudaFree(recvbuff2);
    return ncclSuccess;
}


ncclResult_t alltoall_bruck(double* sendbuff,size_t count1,ncclDataType_t datatype1,double* recvbuff,size_t count2,ncclDataType_t datatype2, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int count = count1;
    ncclDataType_t datatype = datatype1;
    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * size * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }


    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*size*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

    double* sendblock;
    cudaMalloc(&sendblock, count*(size/2)*sizeof(double));

    double* recvblock;
    cudaMalloc(&recvblock, count*(size/2)*sizeof(double));

    // Rotation Phase
    cudaMemcpy(recvbuff2, sendbuff + rank, count * (size-rank) * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaMemcpy(recvbuff2 + count*(size-rank), sendbuff, count * rank * sizeof(double), cudaMemcpyDeviceToDevice);

    // Communication Phases
    int binValBit = 0;
    for(int i = 1; i < size; i*=2){
        int sendto = (rank + i) % size;
        int recvfrom = (rank - i + size) % size;

        ncclGroupStart();
        int sent = 0;
        for (int j = 0; j < size; j++){
            if ((j & (1 << binValBit)) != 0) {
                cudaMemcpy(sendblock+sent*count, recvbuff2+j*count, count*sizeof(double), cudaMemcpyDeviceToDevice);
                sent = sent+1;
            }
        }
        ncclSend(sendblock, sent*count, datatype, sendto, comm, stream);
        ncclRecv(recvblock, sent*count, datatype, recvfrom, comm, stream);
        ncclGroupEnd();
        int recvd = 0;
        for (int j = 0; j < size; j++){
            if ((j & (1 << binValBit)) != 0) {
                cudaMemcpy(recvbuff2+j*count, recvblock+recvd*count, count*sizeof(double), cudaMemcpyDeviceToDevice);
                recvd = recvd+1;
            }
        }
        binValBit = binValBit + 1;
    }
    cudaFree(sendblock);
    cudaFree(recvblock);


    //Inverse Rotation Phase
    double *tempbuff;
    cudaMalloc(&tempbuff, count*size*sizeof(double));
    cudaMemcpy(tempbuff, recvbuff2, count * (size-rank) * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < size; i++){
        int getPos = (rank-i+size)%size;
        cudaMemcpy(recvbuff2 +i*count, tempbuff + getPos*count, count*sizeof(double), cudaMemcpyDeviceToDevice);
    }
    cudaFree(tempbuff);

    if(needtofree){
        cudaMemcpy(sendbuff, recvbuff2, count * size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaFree(recvbuff2);
    }
    
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
    cudaErrorCheck(cudaSetDevice(0));
    cudaErrorCheck(cudaStreamCreate(&s));
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));


    int loop_count = 20;
    int bytesize = 23;
    srand(time(NULL) + rank);
    // MPI 
    if(rank == 0)
        std::cout <<std::endl <<"MPI Host" << std::endl;

    for(int i=0; i<=bytesize; i++){

        int n = (1 << i);
        int N = n*size;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        // double *d_A;
        // cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		// cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

        
        // Warm-up loop
        for(int i=1; i<=5; i++){
            MPI_Alltoall(MPI_IN_PLACE,n,MPI_DOUBLE,A,n,MPI_DOUBLE,MPI_COMM_WORLD);
        }

        double start_time, stop_time;
        double times[loop_count];
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);

        for(int i=1; i<=loop_count; i++){   
            start_time = MPI_Wtime();
            MPI_Alltoall(MPI_IN_PLACE,n,MPI_DOUBLE,A,n,MPI_DOUBLE,MPI_COMM_WORLD);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[i-1] = stop_time - start_time;
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
       
        
        free(A);
    }

    // MPI 
    if(rank == 0)
    std::cout <<std::endl <<"MPI CUDA-Aware" << std::endl;

    for(int i=0; i<=bytesize; i++){

        int n = (1 << i);
        int N = n*size;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );


        // Warm-up loop
        for(int i=1; i<=5; i++){
            MPI_Alltoall(MPI_IN_PLACE,n,MPI_DOUBLE,d_A,n,MPI_DOUBLE,MPI_COMM_WORLD);
        }

        double start_time, stop_time;
        double times[loop_count];
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);
        

        for(int i=1; i<=loop_count; i++){   
            start_time = MPI_Wtime();
            MPI_Alltoall(MPI_IN_PLACE,n,MPI_DOUBLE,d_A,n,MPI_DOUBLE,MPI_COMM_WORLD);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[i-1] = stop_time - start_time;
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
    }


    if(rank == 0)
        std::cout <<std::endl <<"Spread-out" << std::endl;

    for(int i=0; i<=bytesize; i++){
        int n = (1 << i);
        int N = n*size;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

        
        // Warm-up loop
        for(int i=1; i<=5; i++){
            alltoall_spreadout(d_A,n,ncclDouble,d_A,n,ncclDouble,comm,s);
        }
        
        double start_time, stop_time;
        double times[loop_count];
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);

        for(int i=1; i<=loop_count; i++){
            start_time = MPI_Wtime();
            alltoall_spreadout(d_A,n,ncclDouble,d_A,n,ncclDouble,comm,s);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[i-1] = stop_time - start_time;
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
    }

    if(rank == 0)
        std::cout <<std::endl <<"Prim" << std::endl;

    for(int i=0; i<=bytesize; i++){
        int n = (1 << i);
        int N = n*size;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
		cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

        
        // Warm-up loop
        for(int i=1; i<=5; i++){
            alltoall_prim(d_A,n,ncclDouble,d_A,n,ncclDouble,comm,s);
        }
        
        double start_time, stop_time, elapsed_time;
        double times[loop_count];
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);

        for(int i=1; i<=loop_count; i++){
            start_time = MPI_Wtime();
            alltoall_prim(d_A,n,ncclDouble,d_A,n,ncclDouble,comm,s);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[i-1] = stop_time - start_time;
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
    }

    if(rank == 0)
    std::cout <<std::endl <<"Bruck" << std::endl;

    for(int i=0; i<=bytesize; i++){
        int n = (1 << i);
        int N = n*size;

        // Allocate memory for A on CPU
        double *A = (double*)malloc(N*sizeof(double));
        // Initialize all elements of A to random values
        for(int i=0; i<N; i++){
            A[i] = (double)rand()/(double)RAND_MAX;
        }
        double *d_A;
        cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
        cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

        
        // Warm-up loop
        for(int i=1; i<=5; i++){
            alltoall_bruck(d_A,n,ncclDouble,d_A,n,ncclDouble,comm,s);
        }
        
        double start_time, stop_time, elapsed_time;
        double times[loop_count];
        cudaDeviceSynchronize();
        cudaStreamSynchronize(s);
        

        for(int i=1; i<=loop_count; i++){
            start_time = MPI_Wtime();
            alltoall_bruck(d_A,n,ncclDouble,d_A,n,ncclDouble,comm,s);
            cudaStreamSynchronize(s);
            stop_time = MPI_Wtime();
            times[i-1] = stop_time - start_time;
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
    }





    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
        
}
