
#include "nccl_dbg.h"

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
const char *ncclSocketToStringTest(union ncclSocketAddress *addr, char *buf, const int numericHostForm = 1) {
  if (buf == NULL || addr == NULL) return NULL;
  struct sockaddr *saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) { buf[0]='\0'; return buf; }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void) getnameinfo(saddr, sizeof(union ncclSocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, flag);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}
ncclResult_t ncclProxy(struct ncclComm* comm,int proxyRank) 
{
    struct ncclProxyState* sharedProxyState = comm->proxyState;
    if(NULL== sharedProxyState)
        return ncclSuccess;
    int tpProxyRank = comm->topParentRanks[proxyRank];
    struct ncclSocket* sock;
    union ncclSocketAddress *addr = NULL;
    char buff[512];
#if 0
    proxyConn->sameProcess = ((comm->peerInfo[proxyRank].hostHash == comm->peerInfo[comm->rank].hostHash) &&
                            (comm->peerInfo[proxyRank].pidHash == comm->peerInfo[comm->rank].pidHash)) ? 1 : 0;
    // Keep one connection per local rank
    proxyConn->connection = NULL;
    proxyConn->tpRank = tpProxyRank;
    proxyConn->rank = proxyRank;
    if (sharedProxyState->peerSocks == NULL) {
      NCCLCHECK(ncclCalloc(&sharedProxyState->peerSocks, comm->sharedRes->tpNLocalRanks));
      NCCLCHECK(ncclCalloc(&sharedProxyState->proxyOps, comm->sharedRes->tpNLocalRanks));
      NCCLCHECK(ncclCalloc(&sharedProxyState->sharedDevMems, comm->sharedRes->tpNLocalRanks));
      for (int i = 0; i < comm->sharedRes->tpNLocalRanks; ++i) {
        //NCCLCHECK(ncclSocketSetFd(-1, &sharedProxyState->peerSocks[i]));
      }
    }
    proxyConn->tpLocalRank = comm->sharedRes->tpRankToLocalRank[proxyConn->tpRank];
#endif
  sock = sharedProxyState->peerSocks;
  for(int i =0; i < comm->nRanks; ++i)
  {
      addr = sharedProxyState->peerAddresses + i;
      printf("socket addr %s \n",ncclSocketToStringTest(addr,buff));
  }
  return ncclSuccess;
}
ncclResult_t ncclTransportTest(struct ncclComm* comm) {
  // Free collNet resources
  for (int r=0; r<comm->nChannels; r++) {
    struct ncclChannel* channel = comm->channels+r;
    struct ncclChannelPeer* peer = channel->peers[comm->nRanks];
    if (peer) {
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* send = peer->send + b;
          if (send->transportResources && send->transportComm)
              printf("conn->transportComm is %p,netTransport.send %p, netTransport.recv %p \n ",send->transportComm,  &netTransport.send , &netTransport.recv);
        }
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          struct ncclConnector* recv = peer->recv + b;
          if (recv->transportResources && recv->transportComm) 
          printf("conn->transportComm is  %p,netTransport.send %p, netTransport.recv %p \n ",recv->transportComm,  &netTransport.send , &netTransport.recv);
        }
    }
  }
  return ncclSuccess;
}
extern struct ncclTransport* ncclTransports[NTRANSPORTS+1];
//static ncclResult_t selectTransport(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclConnect* connect, int channelId, int peer, int connIndex, int* transportType) {
ncclResult_t selectTransport(int type , struct ncclComm* comm,  int channelId, int peer, int connIndex) {
  //int type = 1;
  struct ncclPeerInfo* myInfo = comm->peerInfo+comm->rank;
  struct ncclPeerInfo* peerInfo = comm->peerInfo+peer;
  struct ncclConnector* connector = (type == 1) ? comm->channels[channelId].peers[peer]->send + connIndex :
                                                  comm->channels[channelId].peers[peer]->recv + connIndex;
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports[t];
    struct ncclTransportComm* transportComm = type==1 ? &transport->send : &transport->recv;
    if(connector->transportComm == transportComm)
         printf("transport found for rank %d[%lx] -> rank %d[%lx], ncclTransportComm type== send ?  %d, name:  %s \n", myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId,type,transport->name);
    if(TRANSPORT_NET == t)
    {
	//struct ncclProxyState* sharedProxyState = comm->proxyState;
	//int tpProxyRank = comm->topParentRanks[proxyRank];
        printf("Channel %02d/%d : %d[%d] -> %d[%d] [send] via NET/%s \n", channelId, connIndex, myInfo->rank, myInfo->nvmlDev, peerInfo->rank, peerInfo->nvmlDev, comm->ncclNet->name);
    }
  }
  //printf("equal netTransport ? %d \n",connector->transportComm == (type? &netTransport.send : &netTransport.recv));
  return ncclSuccess;
}
ncclResult_t test_selectTransport(struct ncclComm* comm,  int connIndex)
{
    for (int i=1; i<comm->nRanks; i++) {
        int recvPeer = (comm->rank - i + comm->nRanks) % comm->nRanks;
        int sendPeer = (comm->rank + i) % comm->nRanks;
        uint64_t recvMask = comm->connectRecv[recvPeer];
        uint64_t sendMask = comm->connectSend[sendPeer];

        // Data[i] contains all ncclConnect information for all send and receive connections with a given send and recv peer
        // This data is packed in the array based on the number of sendChannels and recvChannels connected with these peers
        // The first N entries contain recvData, connection information for recv connections
        // The next M entries contain sendData, connection information for send connections
        // It's not guaranteed that each entry of data has the same number of total or send/recv specific connections
#if 0
        for (int c=0; c<MAXCHANNELS; c++) {
          if (recvMask & (1UL<<c)) {
            selectTransport(0,comm,  c, recvPeer, connIndex);
          }
        }
        for (int c=0; c<MAXCHANNELS; c++) {
          if (sendMask & (1UL<<c)) {
            selectTransport(1,comm,  c, sendPeer, connIndex);
          }
        }
#else
        for (int c=0; c<comm->nChannels; c++) {
            selectTransport(0,comm,  c, recvPeer, connIndex);
        }
        for (int c=0; c<comm->nChannels; c++) {
            selectTransport(1,comm,  c, sendPeer, connIndex);
        }
#endif
    }

  return ncclSuccess;
}
void runring(int tid, int nthreads, struct ncclComm *comm)
{
     struct ncclChannel * channel  = comm->channels + tid;
     ncclRing *ring = &channel->ring;
     const int nranks = comm->nRanks;
     int ringIx = ring->index;
     printf("runRing: TID %d: RingIx %d, nranks %d\n", tid, ringIx, nranks);
}
void runTreeUpDown(int tid, int nthreads, struct ncclComm *comm)
{
     struct ncclChannel * channel  = comm->channels + tid;
     ncclTree *tree = &channel->tree;
}
