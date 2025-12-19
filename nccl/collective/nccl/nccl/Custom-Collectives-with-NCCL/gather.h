#ifndef GATHER_H
#define GATHER_H

#include <mpi.h>

// Function declarations
ncclResult_t gather_inbuilt(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t gather_primitive(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t gather_tree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

//Function definitions

ncclResult_t gather_inbuilt(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    ncclResult_t res = ncclGroupStart();
    res = ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
    res = ncclGroupEnd();
    return res;
}

ncclResult_t gather_primitive(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    ncclResult_t res;
    if(rank == root)
    {
        cudaMemcpy(recvbuff + (count * rank), sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
        res = ncclGroupStart();
        for(int i=0; i<size; i++)
        {
            if(i == rank)
                continue;
            res = ncclRecv(recvbuff+(count*i), count, datatype, i, comm, stream);
        }
        res = ncclGroupEnd();
    }
    else
    {
        res = ncclGroupStart();
        res = ncclSend(sendbuff, count, datatype, root, comm, stream);
        res = ncclGroupEnd();
    }
    return res;
}


ncclResult_t gather_tree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    ncclResult_t res;
    
    cudaMemcpy(recvbuff + (count * rank), sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);

    int rankdiv = 1;
    while(rankdiv < size){
        // res = ncclGroupStart();
        if(rank%rankdiv == 0)
        {
            
            if(rank%(rankdiv*2) == 0)
            {
                int recvfrom = rank+rankdiv;
                if (recvfrom < size){
                    res = ncclRecv(recvbuff+(count*recvfrom), count*rankdiv, datatype, recvfrom, comm, stream);
                }
            }
            else
            {
                int sendto = rank-rankdiv;
                res = ncclSend(recvbuff+(count*rank), count*rankdiv, datatype, sendto, comm, stream);
                
            }
            
            
        }
        // res = ncclGroupEnd();
        rankdiv = rankdiv*2;
    }
    
    return res;
}


ncclResult_t gather_custom(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    ncclResult_t res;
    
    cudaMemcpy(recvbuff + (count * rank), sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    int procPerNode = 4;


    //Intranode
    ncclGroupStart();
    if(rank%procPerNode == 0)
    {
        for(int i=1; i<procPerNode; i++)
        {
            res = ncclRecv(recvbuff+count*(rank+i), count, datatype, rank+i, comm, stream);
        }
    }
    else
    {
        res = ncclSend(sendbuff, count, datatype, rank-(rank%procPerNode), comm, stream);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(stream);

    //Internode
    ncclGroupStart();
    if(rank == 0){
        for(int i=1; i<size/procPerNode; i++)
        {
            res = ncclRecv(recvbuff+count*i*procPerNode, count*procPerNode, datatype, i*procPerNode, comm, stream);
        }
    }
    else if(rank%procPerNode == 0)
    {
        res = ncclSend(recvbuff + count*rank, count*procPerNode, datatype, 0, comm, stream);
    }
    ncclGroupEnd();
    
    return res;
}

ncclResult_t gather_custom2(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    ncclResult_t res;
    
    cudaMemcpy(recvbuff + (count * rank), sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    int procPerNode = 4;


    //Intranode
    ncclGroupStart();
    if(rank%procPerNode == 0)
    {
        for(int i=1; i<procPerNode; i++)
        {
            res = ncclRecv(recvbuff+count*(rank+i), count, datatype, rank+i, comm, stream);
        }
    }
    else
    {
        res = ncclSend(sendbuff, count, datatype, rank-(rank%procPerNode), comm, stream);
    }
    ncclGroupEnd();
    // cudaStreamSynchronize(stream);

    //Internode
    ncclGroupStart();
    if(rank < procPerNode){
        int nextRank = rank * procPerNode;
        if(nextRank == 0)
            nextRank += procPerNode*procPerNode;
        while(nextRank < size){
            // ncclGroupStart();
            res = ncclRecv(recvbuff + count*nextRank, count*procPerNode, datatype, nextRank, comm, stream);
            // printf("Rank %d: recv from %d\n", rank, nextRank);
            nextRank += procPerNode*procPerNode;
            // ncclGroupEnd();
        }
    }
    else if(rank%procPerNode == 0)
    {   
        // for(int i=0; i<rank/(procPerNode*procPerNode); i++)
        // {
        //     ncclGroupStart();
        //     ncclGroupEnd();
        // }
        // ncclGroupStart();
        int sendto = (rank/procPerNode)%procPerNode;
        // printf("Rank %d: send to %d\n", rank, sendto);
        res = ncclSend(recvbuff + count*rank, count*procPerNode, datatype, sendto, comm, stream);
        // ncclGroupEnd();
    }
    ncclGroupEnd();
    cudaStreamSynchronize(stream);
    //Intranode
    // double* tmpbuff = (double*)malloc(count*size*sizeof(double));
    // cudaMemcpy(tmpbuff, recvbuff, count*size*sizeof(double), cudaMemcpyDeviceToHost);
    // if(rank == 1){
    //     for(int i=0; i<size; i++)
    //     {
    //         printf("Rank %d: recvbuff[%d] = %f\n", rank, i, tmpbuff[i]);
    //     }
    // }
    
    if(rank<procPerNode && rank !=0){
        int nextRank = rank * procPerNode;
        while(nextRank < size){
            ncclGroupStart();
            res = ncclSend(recvbuff + count*nextRank, count*procPerNode, datatype, 0, comm, stream);
            // printf("Rank %d: send %d to 0\n", rank, nextRank);
            nextRank += procPerNode*procPerNode;
            ncclGroupEnd();
        }
    }
    else if(rank == 0)
    {
        ncclGroupStart();
        int nextRank = 1 * procPerNode;
        while(nextRank < size){
            if(nextRank%(procPerNode*procPerNode) == 0){
                ncclGroupEnd();
                ncclGroupStart();
            }
            int recvfrom = (nextRank/procPerNode)%procPerNode;
            if(recvfrom == 0){
                nextRank += procPerNode;
                continue;
            }
            res = ncclRecv(recvbuff + count*nextRank, count*procPerNode, datatype, recvfrom, comm, stream);
            // printf("Rank %d: recv  %d from %d\n", rank, nextRank, recvfrom);
            nextRank += procPerNode;
        }
        ncclGroupEnd();
    }

    
    // free(tmpbuff);
    return res;
}


ncclResult_t gather_custom3(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    ncclResult_t res;
    
    cudaMemcpy(recvbuff + (count * rank), sendbuff, count * sizeof(double), cudaMemcpyDeviceToDevice);
    int procPerNode = 4;


    //Intranode
    ncclGroupStart();
    if(rank%procPerNode == 0)
    {
        for(int i=1; i<procPerNode; i++)
        {
            res = ncclRecv(recvbuff+count*(rank+i), count, datatype, rank+i, comm, stream);
        }
    }
    else
    {
        res = ncclSend(sendbuff, count, datatype, rank-(rank%procPerNode), comm, stream);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(stream);

    // Internode (Tree functionality involving only root nodes)
    if (rank % procPerNode == 0) {
        int treeRank = rank / procPerNode;  // Logical rank in the tree
        int treeSize = size / procPerNode; // Number of root nodes

        int step = 1;
        while (step < treeSize) {
            if (treeRank % (2 * step) == 0) {
                // Receive data from the child node
                int child = rank + (step * procPerNode);
                if (child < size) {
                    res = ncclRecv(recvbuff + count * child, count * step * procPerNode, datatype, child, comm, stream);
                }
            } else if (treeRank % step == 0) {
                // Send data to the parent node
                int parent = rank - (step * procPerNode);
                res = ncclSend(recvbuff + count * rank, count * step * procPerNode, datatype, parent, comm, stream);
                break;  // Exit the loop after sending to the parent
            }
            step *= 2;
        }
    }
    
    return res;
}

#endif // GATHER_H