#ifndef SCATTER_H
#define SCATTER_H

#include <mpi.h>

// Function declarations
ncclResult_t scatter_primitive(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t scatter_tree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);

// Function definitions



ncclResult_t scatter_primitive(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ncclResult_t res;
    if (rank == root)
    {
        res = ncclGroupStart();
        for (int i = 0; i < size; i++)
        {
            if (i == rank)
            {
                cudaMemcpy(recvbuff, sendbuff + (count * i), count * sizeof(double), cudaMemcpyDeviceToDevice);
            }
            else
            {
                res = ncclSend(sendbuff + (count * i), count, datatype, i, comm, stream);
            }
        }
        res = ncclGroupEnd();
    }
    else
    {
        res = ncclGroupStart();
        res = ncclRecv(recvbuff, count, datatype, root, comm, stream);
        res = ncclGroupEnd();
    }
    
    return res;
}



ncclResult_t scatter_tree(double* sendbuff, double* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    ncclResult_t res;

    

    int rankdiv = 1;
    while(rankdiv < size/2){
        rankdiv = rankdiv * 2;
    }

    double* tempbuff;
    cudaMalloc(&tempbuff, count*size* sizeof(double));
    cudaMemcpy(tempbuff, sendbuff, count*size* sizeof(double), cudaMemcpyDeviceToDevice);
    while (rankdiv >= 1)
    {
        res = ncclGroupStart();
        if (rank % rankdiv == 0)
        {
            if (rank % (rankdiv * 2) == 0)
            {
                int sendto = rank + rankdiv;
                // std::cout << "rank: " << rank << " sendto: " << sendto <<std::endl;
                res = ncclSend(tempbuff + (count * sendto), count * rankdiv , datatype, sendto, comm, stream);
                
            }
            else
            {
                int recvfrom = rank - rankdiv;
                if(recvfrom < size){
                    // std::cout << "rank: " << rank << " recvfrom: " << recvfrom <<std::endl;
                    res = ncclRecv(tempbuff + (count*rank), count * rankdiv , datatype, recvfrom, comm, stream);
                }
            }

        }
        res = ncclGroupEnd();
        rankdiv = rankdiv / 2;
        
    }

    cudaMemcpy(recvbuff , tempbuff + (count * rank), count * sizeof(double), cudaMemcpyDeviceToDevice);
    // std::cout << "rank: " << rank <<std::endl;
    cudaFree(tempbuff);
    return res;
}

#endif // SCATTER_H
