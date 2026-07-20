#ifndef __NCCL_DBG_H__
#define  __NCCL_DBG_H__
#include <cuda_runtime.h>
#include <nccl.h>
#include <atomic>
#include <thread>
#include <vector>
#include "comm.h"
#include "transport.h"
#define NCCLCOMM_DUMP(comm)                         \
    do {                                            \
   printf("ncclCommDump by comm: rank %d comm %p commHash %lud comm channel %d \n", (comm)->rank, (comm), (comm)->commHash,(comm)->nChannels); \
    } while (0)

#if 0
static ncclResult_t SaveProxy(int peer, struct ncclChannel* channel) {
  if (peer < 0) return ncclSuccess;


  int proxyRecv = 1;
  struct ncclPeer* peerComm = *(channel->peers+peer);
  struct ncclConnector* connector =  proxyRecv ? &peerComm->recv : &peerComm->send;
  if (connector->transportComm == NULL) {
    printf("[%d] Error no transport for %s peer %d on channel %d\n", connector->comm->rank,
        proxyRecv ? "recv" : "send", peer, channel->id);
    return ncclInternalError;
  }
  return ncclSuccess;
}
#endif
const char *ncclSocketToStringTest(union ncclSocketAddress *addr, char *buf, const int numericHostForm ) ;
ncclResult_t ncclProxy(struct ncclComm* comm,int proxyRank) ;
//static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType);
ncclResult_t selectTransport(int type , struct ncclComm* comm,  int channelId, int peer, int connIndex);
ncclResult_t test_selectTransport(struct ncclComm* comm,   int connIndex) ;
void runring(int tid, int nthreads, struct ncclComm *comm);
void runTreeUpDown(int tid, int nthreads, struct ncclComm *comm);
ncclResult_t ncclTransportTest(struct ncclComm* comm);
#endif
