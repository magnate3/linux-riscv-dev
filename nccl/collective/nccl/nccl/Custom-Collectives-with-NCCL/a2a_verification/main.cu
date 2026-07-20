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
    int i = 1;
    //int N = 1 << i;
    int N = i*size;
    
    //--------------REGULAR-------------
    if(rank == 0) std::cout<<"REGULAR"<<std::endl;
    double *Ar = (double*)malloc(N*sizeof(double));
    double *B_verr = (double*)malloc(N*sizeof(double));
    double *B_testr = (double*)malloc(N*sizeof(double));

    for(int i=0; i<N; i++){
        Ar[i] = (double)rand()/(double)RAND_MAX;
        //std::cout<<"A["<<i<<"]: "<<A[i]<<std::endl;
    }

    double *d_Ar,*d_Br;
    cudaErrorCheck( cudaMalloc(&d_Ar, N*sizeof(double)) );
    cudaErrorCheck( cudaMalloc(&d_Br, N*sizeof(double)) );
    cudaErrorCheck( cudaMemcpy(d_Ar, Ar, N*sizeof(double), cudaMemcpyHostToDevice) );

    

    alltoall_bruck(d_Ar,i,ncclDouble,d_Br,i,ncclDouble,comm,s);

    MPI_Alltoall(Ar,i,MPI_DOUBLE,B_verr,i,MPI_DOUBLE,MPI_COMM_WORLD);
    
    //cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaMemcpy(B_testr, d_Br, N*sizeof(double), cudaMemcpyDeviceToHost) );

    if (rank==0){
        bool func_corr = true;
        for(int i=0; i<N; i++){
            printf("B_ver[%i]: %.15f \n",i,B_verr[i]);
            printf("B_test[%i]: %.15f \n",i,B_testr[i]);
            if(abs(B_verr[i] - B_testr[i]) > 0.000001){
                func_corr = false;
                std::cout<<"This one was false!"<<std::endl;
            } 
        }
        if (func_corr){
            std::cout<<"Function works as intended!"<<std::endl;
        }else{
            std::cout<<"Function doesn't works as intended :("<<std::endl;
        }
    }


    cudaFree(d_Ar);
    cudaFree(d_Br);
    free(Ar);
    free(B_testr);
    free(B_verr);


    //------------INPLACE--------------
    if(rank == 0) std::cout<<"INPLACE"<<std::endl;
    double *A = (double*)malloc(N*sizeof(double));
    double *B_ver = (double*)malloc(N*sizeof(double));
    double *B_test = (double*)malloc(N*sizeof(double));

    for(int i=0; i<N; i++){
        A[i] = (double)rand()/(double)RAND_MAX;
        //std::cout<<"A["<<i<<"]: "<<A[i]<<std::endl;
    }

    double *d_A,*d_B;
    cudaErrorCheck( cudaMalloc(&d_A, N*sizeof(double)) );
    cudaErrorCheck( cudaMalloc(&d_B, N*sizeof(double)) );
    cudaErrorCheck( cudaMemcpy(d_A, A, N*sizeof(double), cudaMemcpyHostToDevice) );

    

    alltoall_bruck(d_A,i,ncclDouble,d_A,i,ncclDouble,comm,s);

    MPI_Alltoall(MPI_IN_PLACE,i,MPI_DOUBLE,A,i,MPI_DOUBLE,MPI_COMM_WORLD);
    
    //cudaErrorCheck( cudaMemcpy(A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );
    cudaErrorCheck( cudaMemcpy(B_test, d_A, N*sizeof(double), cudaMemcpyDeviceToHost) );

    if (rank==0){
        bool func_corr = true;
        for(int i=0; i<N; i++){
            printf("B_ver[%i]: %.15f \n",i,A[i]);
            printf("B_test[%i]: %.15f \n",i,B_test[i]);
            if(abs(A[i] - B_test[i]) > 0.000001){
                func_corr = false;
                std::cout<<"This one was false!"<<std::endl;
            } 
        }
        if (func_corr){
            std::cout<<"Function works as intended!"<<std::endl;
        }else{
            std::cout<<"Function doesn't works as intended :("<<std::endl;
        }
    }


    cudaFree(d_A);
    cudaFree(d_B);
    free(A);
    free(B_test);
    free(B_ver);
    

    cudaErrorCheck(cudaStreamSynchronize(s));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
        
}
