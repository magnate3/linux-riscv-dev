// export NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml
// export NCCL_DEBUG=INFO
// export NCCL_DEBUG_SUBSYS=GRAPH
//
// g++ -g -Wall nccl-topo.cc -lnccl -lcudart  -L/usr/local/cuda/lib64/ -I /usr/local/cuda/include  -o topo
// export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

#include <stdlib.h>
#include <cstdio>
#include <nccl.h>
#include <cuda_runtime.h>

int main(int argc, char **argv)
{
    int nDevices = 0;
    ncclUniqueId id;
    ncclComm_t comm;

    setenv("NCCL_DEBUG", "INFO", 0);
    setenv("NCCL_DEBUG_SUBSYS", "GRAPH", 0);
    setenv("NCCL_TOPO_DUMP_FILE", "/tmp/nccl_topo.xml", 0);
    setenv("NCCL_TOPO_FILE", "topo_dump.xml", 0);

    cudaGetDeviceCount(&nDevices);
    printf("Detected %d GPUs\n", nDevices);

    // For each pair of GPUs, print link type
    for (int i = 0; i < nDevices; ++i) {
        for (int j = 0; j < nDevices; ++j) {
            int canAccessPeer = 0;

            cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
            printf("GPU %d <-> GPU %d: %s\n", i, j, 
                (canAccessPeer ? "Peer Access Supported" : "No Direct Peer Access"));
        }
    }
    // NCCL Initialization (topology discovery)
    printf("calling ncclGetUniqueId ...\n");
    ncclGetUniqueId(&id);

    printf("calling ncclCommInitRank ...\n");
    ncclCommInitRank(&comm, nDevices, id, 0);
    // NCCL will now internally discover and optimize the topology

    printf("NCCL Topology Discovery Complete; topology written to %s\n",
	   getenv("NCCL_TOPO_DUMP_FILE"));
    ncclCommDestroy(comm);
    return 0;
}
