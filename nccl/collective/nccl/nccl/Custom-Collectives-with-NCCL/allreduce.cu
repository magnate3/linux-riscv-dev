#include "allreduce.h"

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



ncclResult_t allred_ring(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    //"parallel ring" implementation for reduce 
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

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

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < size-1; i++){
        
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff, count,datatype, next,comm,stream);
            ncclRecv(recvbuff2, count,datatype, last,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,sendbuff,count);
        }else
        {
            ncclGroupStart();
            ncclRecv(recvbuff2, count,datatype, last,comm,stream);
            ncclSend(tempbuff, count,datatype, next,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,sendbuff,count);
        }
        cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    if(needtofree){
        cudaFree(recvbuff2);
    }
    cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(tempbuff);
    return ncclSuccess;
    
}

ncclResult_t allred_ring_aware(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
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

    
    for (int i = 0; i < size-1; i++){
        if (rank%2 == 0){
            
            MPI_Send( tempbuff , count , MPI_DOUBLE, next , 0 , MPI_COMM_WORLD);
            MPI_Recv( recvbuff , count , MPI_DOUBLE , last , 0 , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
            max_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff,count);

        }else
        {
            MPI_Recv( recvbuff , count , MPI_DOUBLE , last , 0 , MPI_COMM_WORLD , MPI_STATUS_IGNORE);
            MPI_Send( tempbuff , count , MPI_DOUBLE, next , 0 , MPI_COMM_WORLD);
            
            max_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff,count);
        }
    }
    recvbuff = tempbuff;
    cudaFree(tempbuff);
    return ncclSuccess;
    
}

ncclResult_t allred_ring_pipelined(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    //"parallel ring" implementation for reduce 
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int pipe = std::min(size,(int)count);

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

    for (int i = 0; i < size-1; i++){
        int send_index = ((rank-i)%size+size)%size;
        int recv_index = ((rank-i-1)%size+size)%size;
        int buffer_size = count/pipe;
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(recvbuff+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
            max_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff+ recv_index,recvbuff+ recv_index,pipe);

        }else
        {
            ncclGroupStart();
            ncclRecv(recvbuff+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
            max_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff+ recv_index,recvbuff+ recv_index,pipe);
        }

    }
    cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    for (int i = 0; i < size-1; i++){
        int send_index = ((rank-i+1)%size+size)%size;
        int recv_index = ((rank-i)%size+size)%size;
        int buffer_size = count/pipe;
        if (rank%2 == 0){
            
            ncclSend(recvbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(recvbuff+ recv_index, buffer_size,datatype, last,comm,stream);
            

        }else
        {
            
            ncclRecv(recvbuff+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(recvbuff + send_index, buffer_size,datatype, next,comm,stream);
            
        }

    }

    cudaFree(tempbuff);
    return ncclSuccess;
}

ncclResult_t allred_butterfly(const double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    
    int size;
    int rank;
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


    for(int i = 1;i<size;i=i*2){
        bool send_forward = rank%(i*2) < i;
        int rank_exchange;
        if(send_forward){
            rank_exchange = rank + i;
            ncclSend(tempbuff, count,datatype, rank_exchange,comm,stream);
            ncclRecv(recvbuff, count,datatype, rank_exchange,comm,stream);
            max_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff,count);
        }
        else
        {
            rank_exchange = rank - i;
            ncclRecv(recvbuff, count,datatype, rank_exchange,comm,stream);
            ncclSend(tempbuff, count,datatype, rank_exchange,comm,stream);
            max_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff,count);
        }
    }

    cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    cudaFree(tempbuff);
    return ncclSuccess;
}


ncclResult_t allred_ring_pipelined2( double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    //"parallel ring" implementation for reduce 
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

    int pipe = std::min(size,(int)count);

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

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < size-1; i++){
        int send_index = ((rank-i)%size+size)%size;
        int recv_index = ((rank-i-1)%size+size)%size;
        int buffer_size = count/pipe;
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
            max_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2+ recv_index,sendbuff+ recv_index,buffer_size);

        }else
        {
            ncclGroupStart();
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
            max_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2+ recv_index,sendbuff+ recv_index,buffer_size);
        }
        cudaMemcpy(tempbuff+recv_index, recvbuff2+recv_index, buffer_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    

    for (int i = 0; i < size-1; i++){
        int send_index = ((rank-i+1)%size+size)%size;
        int recv_index = ((rank-i)%size+size)%size;
        int buffer_size = count/pipe;
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
        }else
        {
            ncclGroupStart();
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
        }

    }
    cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    cudaFree(tempbuff);
    return ncclSuccess;
}


ncclResult_t allred_ring_pipe_n( double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int pipeline){
    //"parallel ring" implementation for reduce 
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

    //int pipe = std::min(size,(int)count);
    //int pipe = pipeline;
    int pipe = std::min(pipeline,(int)count);
    pipe = std::min(pipe,size);

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
    int buffer_size = std::max((int)round(count/pipe),1);
    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < size-1; i++){
        int send_index = (((rank-i)%pipe+pipe)%pipe ) * buffer_size;
        int recv_index = (((rank-i-1)%pipe+pipe)%pipe ) * buffer_size;
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2+ recv_index,sendbuff+ recv_index,buffer_size);

        }else
        {
            ncclGroupStart();
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2+ recv_index,sendbuff+ recv_index,buffer_size);
        }
        cudaMemcpy(tempbuff+recv_index, recvbuff2+recv_index, buffer_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    

    for (int i = 0; i < pipe-1; i++){
        int send_index = (((rank-size-i+1)%pipe+pipe)%pipe) * buffer_size;
        int recv_index = (((rank-size-i)%pipe+pipe)%pipe) * buffer_size;
        
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
        }else
        {
            ncclGroupStart();
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
        }

    }
    cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    cudaFree(tempbuff);
    return ncclSuccess;
}

ncclResult_t allred_ring_pipe_max( double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    //"parallel ring" implementation for reduce 
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    int pipeline = size;
    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }

    //int pipe = std::min(size,(int)count);
    //int pipe = pipeline;
    int pipe = std::min(pipeline,(int)count);
    pipe = std::min(pipe,size);

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
    int buffer_size = std::max((int)round(count/pipe),1);
    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < size-1; i++){
        int send_index = (((rank-i)%pipe+pipe)%pipe ) * buffer_size;
        int recv_index = (((rank-i-1)%pipe+pipe)%pipe ) * buffer_size;
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2+ recv_index,sendbuff+ recv_index,buffer_size);

        }else
        {
            ncclGroupStart();
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2+ recv_index,sendbuff+ recv_index,buffer_size);
        }
        cudaMemcpy(tempbuff+recv_index, recvbuff2+recv_index, buffer_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    
    

    for (int i = 0; i < pipe-1; i++){
        int send_index = (((rank-size-i+1)%pipe+pipe)%pipe) * buffer_size;
        int recv_index = (((rank-size-i)%pipe+pipe)%pipe) * buffer_size;
        
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
        }else
        {
            ncclGroupStart();
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
        }

    }
    cudaMemcpy(recvbuff, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    cudaFree(tempbuff);
    return ncclSuccess;
}

ncclResult_t allred_tree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }


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

    
    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    int redmod = 2;
    while(redmod <= size){
        cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        if (rank%redmod != 0)
        {
            ncclSend( recvbuff2 , count , datatype , rank-(redmod/2) , comm , stream);
            break;
        }else
        {
            
            ncclRecv( recvbuff2 , count , datatype , rank+(redmod/2) , comm , stream);
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count);
            
        }
        redmod = redmod*2;
    }

    int bcastmod = size;
    while(bcastmod >= 2){
        if (rank%(bcastmod/2) != 0){
            bcastmod = bcastmod/2;
            continue;
        }
        if (rank%bcastmod == 0){
            ncclSend( recvbuff2 , count , datatype , rank+(bcastmod/2) , comm , stream);
        }else{
            ncclRecv( recvbuff2 , count , datatype , rank-(bcastmod/2) , comm , stream);
        }
        bcastmod = bcastmod/2;
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    cudaFree(tempbuff);
    return ncclSuccess;
}



ncclResult_t allred_dbtree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    double *recvbuff_extra;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }
    cudaMalloc(&recvbuff_extra, count*sizeof(double));


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 1024;
    int blk_in_grid = ceil( float(count) / thr_per_blk );
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    int count2 = count/2;
    int count1 = count - count2;


    //Reduction
    int redmod = 2;
    int rank2 = (rank+1)%size;
    bool broken1 = false;
    int rcvd1 = 0;
    int rcvd2 = 0;
    bool broken2 = false;
    while(redmod<= size){
        cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        rcvd1 = 0;
        rcvd2 = 0;
        ncclGroupStart();
        if (rank%redmod != 0 && !broken1)
        {
            //Send Message
            int sendto;
            if(rank%(redmod*2) > redmod){
                sendto = rank -(redmod/2);
            }else{
                sendto = rank + (redmod/2);
            }
            sendto = sendto%size;
            
            ncclSend( recvbuff2 , count1 , datatype , sendto , comm , stream);
            
            broken1 = true;
        }else if (!broken1)
        {
            //Recv Message(s)
            if(rank%(redmod*2) != 0){
                ncclRecv( recvbuff2 , count1 , datatype , rank+(redmod/2) , comm , stream);
                ncclRecv( recvbuff_extra , count1 , datatype , rank-(redmod/2) , comm , stream);
                rcvd1 = 2;
            }else if(redmod >= size){
                ncclRecv( recvbuff2 , count1 , datatype , rank+(redmod/2) , comm , stream);
                rcvd1 = 1;
            }
            
            
        }

        if (rank2%redmod != 0 && !broken2)
        {
            //Send Message
            int sendto;
            if(rank2%(redmod*2) > redmod){
                sendto = rank2 -(redmod/2);
            }else{
                sendto = rank2 + (redmod/2);
            }
            sendto = sendto%size;
            ncclSend( recvbuff2 + count1 , count2 , datatype , (sendto-1+size)%size , comm , stream);
            broken2 = true;
        }else if (!broken2)
        {
            //Recv Message(s)
            if(rank2%(redmod*2) != 0){
                ncclRecv( recvbuff2 + count1 , count2 , datatype , (rank2+(redmod/2)-1)%size , comm , stream);
                ncclRecv( recvbuff_extra + count1 , count2 , datatype , (rank2-(redmod/2)-1)%size , comm , stream);
                rcvd2 = 2;
            }else if(redmod >= size){
                ncclRecv( recvbuff2 + count1 , count2 , datatype , (rank2+(redmod/2)-1)%size , comm , stream);
                rcvd2 = 1;
            }
            
            
        }
        ncclGroupEnd();

        if (rcvd1 == 2){
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count1);
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,recvbuff_extra,count1);
        }else if(rcvd1 == 1){
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count1);
        }

        if (rcvd2 == 2){
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2 + count1,tempbuff + count1,count2);
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2 + count1,recvbuff_extra + count1,count2);
        }else if(rcvd2 == 1){
            add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2 + count1,tempbuff + count1,count2);
        }
        
        redmod = redmod*2;


    }

    broken1 = false;
    broken2 = false;
    //Broadcast
    int bcastmod = size;
    while(bcastmod >= 2){
        ncclGroupStart();
        if (rank%(bcastmod/2) == 0 && !broken1){
            if (rank%bcastmod == 0){
                //Send message(s)
                if(bcastmod == size){
                    ncclSend( recvbuff2 , count1 , datatype , rank+(bcastmod/2) , comm , stream);
                }else{
                    int sendto1 = rank+(bcastmod/2);
                    int sendto2 = rank-(bcastmod/2);
                    ncclSend( recvbuff2 , count1 , datatype , sendto1 , comm , stream);
                    ncclSend( recvbuff2 , count1 , datatype , sendto2 , comm , stream);
                }
                broken1 = true;
            }else{
                //Receive Message
                int recvfrom;
                if(rank%(bcastmod*2) > bcastmod){
                    recvfrom = rank - (bcastmod/2);
                }else{
                    recvfrom = rank + (bcastmod/2);
                }
                recvfrom = recvfrom%size;
                ncclGroupStart();
                ncclRecv( recvbuff2 , count1 , datatype , recvfrom , comm , stream);
                ncclGroupEnd();
            }
            
        }
        if (rank2%(bcastmod/2) == 0 && !broken2){
            if (rank2%bcastmod == 0){
                //Send message(s)
                if(bcastmod == size){
                    ncclSend( recvbuff2 + count1  , count2 , datatype , (rank2+(bcastmod/2)-1+size)%size , comm , stream);
                }else{
                    int sendto1 = rank2+(bcastmod/2);
                    int sendto2 = rank2-(bcastmod/2);
                    
                    ncclSend( recvbuff2 + count1 , count2 , datatype , (sendto1-1+size)%size , comm , stream);
                    ncclSend( recvbuff2 + count1, count2 , datatype , (sendto2-1+size)%size , comm , stream);
                }
                broken2 = true;
            }else{
                //Receive Message
                int recvfrom;
                if(rank2%(bcastmod*2) > bcastmod){
                    recvfrom = rank2 - (bcastmod/2);
                }else{
                    recvfrom = rank2 + (bcastmod/2);
                }
                recvfrom = recvfrom%size;
                ncclRecv( recvbuff2 + count1 , count2 , datatype , (recvfrom-1+size)%size , comm , stream);
                
            }
        }
        ncclGroupEnd();
        bcastmod = bcastmod/2;
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    cudaFree(recvbuff_extra);
    cudaFree(tempbuff);
    return ncclSuccess;
}



ncclResult_t allred_custom(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(count) / thr_per_blk );

    int procPerNode = 4;

    
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);


    //intranode
    if(rank%procPerNode == 0){
        for(int i = rank+1;i< rank+procPerNode;i++){
            // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
            ncclRecv(recvbuff2, count, datatype, i,comm,stream);
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff2,count);
        }
    }
    else{
        
        ncclSend(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
    }
    
    //internode
    int ringsize = size/procPerNode;
    int next = (rank+procPerNode)%size;
    int last = ((rank-procPerNode)%size+size)%size;
    cudaMemcpy(recvbuff2,tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < ringsize-1; i++){
        ncclGroupStart();
        if(rank%procPerNode == 0){
            // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
            ncclSend(recvbuff2,count,datatype,next,comm,stream);
            ncclRecv(recvbuff2,count,datatype,last,comm,stream);
        }
        ncclGroupEnd();
        add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff2,count);

    }

    //intranode
    if(rank%procPerNode == 0){
        // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        ncclGroupStart();
        for(int i = rank+1;i< rank+procPerNode;i++){
            ncclSend(tempbuff, count, datatype, i,comm,stream);
            // add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count);
        }
        ncclGroupEnd();
        cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else{
        ncclGroupStart();
        ncclRecv(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
        ncclGroupEnd();
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    
    cudaFree(tempbuff);
    return ncclSuccess;
}



ncclResult_t allred_custom2(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int pipeline){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(count) / thr_per_blk );

    int procPerNode = 4;

    
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);


    //intranode
    if(rank%procPerNode == 0){
        for(int i = rank+1;i< rank+procPerNode;i++){
            // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
            ncclRecv(recvbuff2, count, datatype, i,comm,stream);
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff2,count);
        }
    }
    else{
        
        ncclSend(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
    }
    
    // Internode communication using a pipelined ring algorithm
    int ringsize = size / procPerNode;
    int next = (rank + procPerNode) % size;
    int last = ((rank - procPerNode) % size + size) % size;

    // Determine the pipeline size (number of chunks)
    int pipe = std::min(pipeline, (int)count);
    int buffer_size = std::max((int)round(count/pipe),1);

    cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    // rank = rank / procPerNode;
    for (int i = 0; i < ringsize-1; i++){
        int send_index = (((rank-i)%pipe+pipe)%pipe ) * buffer_size;
        int recv_index = (((rank-i-1)%pipe+pipe)%pipe ) * buffer_size;
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(recvbuff2+ send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff+ recv_index,recvbuff2+ recv_index,buffer_size);

        }else
        {
            ncclGroupStart();
            ncclRecv(recvbuff2+ recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(recvbuff2 + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff+ recv_index,recvbuff2+ recv_index,buffer_size);
        }
        // cudaMemcpy(tempbuff+recv_index, recvbuff2+recv_index, buffer_size * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < pipe-1; i++){
        int send_index = (((rank-size-i+1)%pipe+pipe)%pipe) * buffer_size;
        int recv_index = (((rank-size-i)%pipe+pipe)%pipe) * buffer_size;
        
        if (rank%2 == 0){
            ncclGroupStart();
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclGroupEnd();
        }else
        {
            ncclGroupStart();
            ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream);
            ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream);
            ncclGroupEnd();
        }

    }
    // MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    //intranode
    if(rank%procPerNode == 0){
        // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        ncclGroupStart();
        for(int i = rank+1;i< rank+procPerNode;i++){
            ncclSend(tempbuff, count, datatype, i,comm,stream);
            // add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count);
        }
        ncclGroupEnd();
        cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else{
        ncclGroupStart();
        ncclRecv(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
        ncclGroupEnd();
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    
    cudaFree(tempbuff);
    return ncclSuccess;
}


ncclResult_t allred_custom_seg(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream, int pipeline){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 64;
    int blk_in_grid = (count + thr_per_blk - 1) / thr_per_blk;


    int procPerNode = 4;
    // double *B_verr = (double*)malloc(count *  sizeof(double));
    
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);


    //intranode
    if(rank%procPerNode == 0){
        for(int i = rank+1;i< rank+procPerNode;i++){
            // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
            ncclRecv(recvbuff2, count, datatype, i,comm,stream);
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff2,count);
            // if(rank==4){
            //     std::cout << "added" << i << std::endl;
                
            // }
        }
    }
    else{
        
        ncclSend(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
    }


    // cudaErrorCheck(cudaMemcpy(B_verr, tempbuff, count * sizeof(double), cudaMemcpyDeviceToHost));
    // if (rank == 0) {
    //     printf("ENDINTRA\n");
    //     for(int i = 0; i < count ; i++) {
    //         printf("B_ver[%i]: %.15f \n", i, B_verr[i]);
    //     }
    // }




    
    // Internode communication using a pipelined ring algorithm
    int ringsize = size / procPerNode;
    int next = (rank + procPerNode) % size;
    int last = ((rank - procPerNode) % size + size) % size;
    

    // Determine the pipeline size (number of chunks)
    int pipe = std::min(pipeline, (int)count);
    pipe = std::min(pipe,ringsize);
    int buffer_size = std::max((int)round(count/pipe),1);
    int ringrank = rank/procPerNode;
    cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < ringsize - 1; i++) {
        int send_index = (((ringrank - i) % pipe + pipe) % pipe) * buffer_size;
        int recv_index = (((ringrank - i - 1) % pipe + pipe) % pipe) * buffer_size;
    
        if (rank % procPerNode == 0) {
            NCCLCHECK(ncclGroupStart());
            // if(ringrank%2 == 0){
            //     NCCLCHECK(ncclSend(tempbuff + send_index, buffer_size, datatype, next, comm, stream));
            //     NCCLCHECK(ncclRecv(recvbuff2 + recv_index, buffer_size, datatype, last, comm, stream));
            // }else{
            //     NCCLCHECK(ncclRecv(recvbuff2 + recv_index, buffer_size, datatype, last, comm, stream));
            //     NCCLCHECK(ncclSend(tempbuff + send_index, buffer_size, datatype, next, comm, stream));
            // }
            NCCLCHECK(ncclSend(tempbuff + send_index, buffer_size, datatype, next, comm, stream));
            NCCLCHECK(ncclRecv(recvbuff2 + recv_index, buffer_size, datatype, last, comm, stream));
            NCCLCHECK(ncclGroupEnd());
            cudaStreamSynchronize(stream);
            add_vec<<<blk_in_grid, thr_per_blk>>>(tempbuff + recv_index, recvbuff2 + recv_index, buffer_size);
            
        }else{
            NCCLCHECK(ncclGroupStart());
            NCCLCHECK(ncclGroupEnd());
        }
    }
    // printf("COMPLETEHEALF");

    


    // if(rank==0){
    //     std::cout << "first done" << pipe << std::endl;
    // }
    // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    for (int i = 0; i < pipe-1; i++){
        int send_index = (((ringrank-ringsize-i+1)%pipe+pipe)%pipe) * buffer_size;
        int recv_index = (((ringrank-ringsize-i)%pipe+pipe)%pipe) * buffer_size;
        
        if(rank%procPerNode == 0){
            ncclGroupStart();
            NCCLCHECK(ncclSend(tempbuff + send_index, buffer_size,datatype, next,comm,stream));
            NCCLCHECK(ncclRecv(tempbuff + recv_index, buffer_size,datatype, last,comm,stream));
            ncclGroupEnd();
        }

    }
    
    // cudaErrorCheck(cudaMemcpy(B_verr, tempbuff, count * sizeof(double), cudaMemcpyDeviceToHost));
    // if (rank == 0) {
    //     printf("ENDINTER\n");
    //     for(int i = 0; i < count ; i++) {
    //         printf("B_ver[%i]: %.15f \n", i, B_verr[i]);
    //     }
    // }
    
    //intranode
    if(rank%procPerNode == 0){
        // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        ncclGroupStart();
        for(int i = rank+1;i< rank+procPerNode;i++){
            ncclSend(tempbuff, count, datatype, i,comm,stream);
            // add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count);
        }
        ncclGroupEnd();
        cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else{
        ncclGroupStart();
        ncclRecv(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
        ncclGroupEnd();
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }


    // cudaErrorCheck(cudaMemcpy(B_verr, tempbuff, count * sizeof(double), cudaMemcpyDeviceToHost));
    // if (rank == 0) {
    //     printf("END\n");
    //     for(int i = 0; i < count ; i++) {
    //         printf("B_ver[%i]: %.15f \n", i, B_verr[i]);
    //     }
    // }
    
    cudaFree(tempbuff);
    return ncclSuccess;
}


ncclResult_t allred_custom3(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(count) / thr_per_blk );

    int procPerNode = 4;

    
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);


    //intranode
    if(rank%procPerNode == 0){
        for(int i = rank+1;i< rank+procPerNode;i++){
            // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
            ncclRecv(recvbuff2, count, datatype, i,comm,stream);
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff2,count);
        }
    }
    else{
        
        ncclSend(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
    }
    
    //internode
    int ringsize = size/procPerNode;
    // Internode communication using a double tree algorithm
    int count2 = count / 2;  // Split the data into two halves
    int count1 = count - count2;  // Handle uneven splits

    int redmod = 2;
    while (redmod <= ringsize) {
        cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        ncclGroupStart();
        // First half of the data
        if (rank % redmod != 0) {
            // Send the first half to the parent
            ncclSend(recvbuff2, count1, datatype, rank - (redmod / 2), comm, stream);
            break;
        } else {
            // Receive the first half from the child
            ncclRecv(recvbuff2, count1, datatype, rank + (redmod / 2), comm, stream);
            add_vec<<<blk_in_grid, thr_per_blk>>>(recvbuff2, tempbuff, count1);
        }

        // Second half of the data
        if ((rank + 1) % redmod != 0) {
            // Send the second half to the parent
            ncclSend(recvbuff2 + count1, count2, datatype, rank - (redmod / 2), comm, stream);
            break;
        } else {
            // Receive the second half from the child
            ncclRecv(recvbuff2 + count1, count2, datatype, rank + (redmod / 2), comm, stream);
            add_vec<<<blk_in_grid, thr_per_blk>>>(recvbuff2 + count1, tempbuff + count1, count2);
        }
        ncclGroupEnd();

        redmod *= 2;
    }

    // Broadcast phase
    int bcastmod = ringsize;
    while (bcastmod >= 2) {
        // First half of the data
        ncclGroupStart();
        if (rank % bcastmod == 0) {
            // Send the first half to the children
            ncclSend(recvbuff2, count1, datatype, rank + (bcastmod / 2), comm, stream);
        } else if (rank % (bcastmod / 2) == 0) {
            // Receive the first half from the parent
            ncclRecv(recvbuff2, count1, datatype, rank - (bcastmod / 2), comm, stream);
        }

        // Second half of the data
        if ((rank + 1) % bcastmod == 0) {
            // Send the second half to the children
            ncclSend(recvbuff2 + count1, count2, datatype, rank + (bcastmod / 2), comm, stream);
        } else if ((rank + 1) % (bcastmod / 2) == 0) {
            // Receive the second half from the parent
            ncclRecv(recvbuff2 + count1, count2, datatype, rank - (bcastmod / 2), comm, stream);
        }
        ncclGroupEnd();
        bcastmod /= 2;
    }

    //intranode
    if(rank%procPerNode == 0){
        // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        ncclGroupStart();
        for(int i = rank+1;i< rank+procPerNode;i++){
            ncclSend(tempbuff, count, datatype, i,comm,stream);
            // add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count);
        }
        ncclGroupEnd();
        cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else{
        ncclGroupStart();
        ncclRecv(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
        ncclGroupEnd();
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    
    cudaFree(tempbuff);
    return ncclSuccess;
}


ncclResult_t allred_custom4(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    double *recvbuff2;
    bool needtofree = false;
    if(sendbuff == recvbuff){
        cudaMalloc(&recvbuff2, count*sizeof(double));
        needtofree = true;
    }else{
        recvbuff2 = recvbuff;
    }


    if (size==1)
    {
        cudaMemcpy(recvbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        return ncclSuccess;
    }
    
    //Set params
    int thr_per_blk = 256;
    int blk_in_grid = ceil( float(count) / thr_per_blk );

    int procPerNode = 4;

    
    
    //create a temporary buffer
    double *tempbuff;
    cudaMalloc(&tempbuff, count*sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    cudaMemcpy(recvbuff2, sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);


    //intranode
    if(rank%procPerNode == 0){
        for(int i = rank+1;i< rank+procPerNode;i++){
            // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
            ncclRecv(recvbuff2, count, datatype, i,comm,stream);
            add_vec<<<blk_in_grid, thr_per_blk >>>(tempbuff,recvbuff2,count);
        }
    }
    else{
        
        ncclSend(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
    }
    
    //internode
    // Internode communication using a simple tree algorithm

    // Reduction Phase
    int redmod = 2;
    while (redmod <= size) {
        cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);

        if (rank % redmod != 0) {
            // Send data to the parent
            int parent = rank - (redmod / 2);
            ncclSend(recvbuff2, count, datatype, parent, comm, stream);
            break;  // Exit the loop after sending to the parent
        } else {
            // Receive data from the child
            int child = rank + (redmod / 2);
            if (child < size) {
                ncclRecv(recvbuff2, count, datatype, child, comm, stream);
                add_vec<<<blk_in_grid, thr_per_blk>>>(tempbuff, recvbuff2, count);
                cudaStreamSynchronize(stream);  // Ensure kernel completion
            }
        }

        redmod *= 2;
    }

    // Broadcast Phase
    int bcastmod = size;
    while (bcastmod >= 2) {
        if (rank % bcastmod == 0) {
            // Send data to the children
            int child = rank + (bcastmod / 2);
            if (child < size) {
                ncclSend(tempbuff, count, datatype, child, comm, stream);
            }
        } else if (rank % (bcastmod / 2) == 0) {
            // Receive data from the parent
            int parent = rank - (bcastmod / 2);
            ncclRecv(tempbuff, count, datatype, parent, comm, stream);
        }

        bcastmod /= 2;
    }

    //intranode
    if(rank%procPerNode == 0){
        // cudaMemcpy(tempbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
        ncclGroupStart();
        for(int i = rank+1;i< rank+procPerNode;i++){
            ncclSend(tempbuff, count, datatype, i,comm,stream);
            // add_vec<<<blk_in_grid, thr_per_blk >>>(recvbuff2,tempbuff,count);
        }
        ncclGroupEnd();
        cudaMemcpy(recvbuff2, tempbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    }
    else{
        ncclGroupStart();
        ncclRecv(recvbuff2, count, datatype,(rank/procPerNode)*procPerNode,comm,stream);
        ncclGroupEnd();
    }

    cudaMemcpy(recvbuff, recvbuff2, count * sizeof(double), cudaMemcpyDeviceToDevice);
    if(needtofree){
        cudaFree(recvbuff2);
    }
    
    cudaFree(tempbuff);
    return ncclSuccess;
}
