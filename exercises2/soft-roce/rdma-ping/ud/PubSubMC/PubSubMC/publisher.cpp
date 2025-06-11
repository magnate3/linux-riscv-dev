#include <stdlib.h>
#include <stdio.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>

#include </usr/include/infiniband/verbs.h>
#include </usr/include/rdma/rdma_cma.h>
//#include <chrono>

#include "instrument.h"

using namespace std;

void PrintUsage();
static int get_addr(char *dst, struct sockaddr *addr);

//Construction and Destruction
int RDMAPublisherInit();
int RDMACreateQP();
void CleanUpPubContext();

//CM Event Channel Handlers
int OnAddressResolved(struct rdma_cm_event *event);
int OnMulticastJoin(struct rdma_cm_event *event);

//QP Operations
ibv_send_wr* create_SEND_WQE(void* buffer, size_t bufferlen, ibv_mr* bufferMemoryRegion);
ibv_mr *create_MEMORY_REGION(void* buffer, size_t bufferlen);
int post_SEND_WQE(ibv_send_wr* ll_wqe);


// Completion Queue Event Handlers
void OnReceiveUpdate();

void *MonitorCMEventChannel(void* data);
void *SubscriptionMonitor(void* data);

int PollCQ();

typedef struct CQContext
{
	int id;
} CQContext_t;

typedef struct QPContext
{
	int id;
} QPContext_t;

typedef struct CMContext
{
	int id;
} CMContext_t;

static 	char	*mcast_string;
static	char 	*local_string;

struct PubContext {

	/*
	 * Variable Used for Address Resolution.
	 */
	struct sockaddr_in6			local_in;
	struct sockaddr_in6			mcast_in;
	struct sockaddr				*local_addr;
	struct sockaddr				*mcast_addr;

	//Contexts - Defined by the Local Applicaiton for tracking state (Can be NULL)
	CMContext_t 				*CMContext;
	QPContext_t				*QPContext;

	//Connection Manager Constructs
	struct 	rdma_event_channel		*CMEventChannel;
	struct 	rdma_cm_id			*CMId;

	//QP Context Structures
	struct 	ibv_cq                		*CompletionQueue;	/* Completion Queue Handle */
	struct 	ibv_comp_channel		*CompletionChannel;	/* Completion Channel */
	struct 	ibv_ah 				*AddressHandle;		/* Address Handle */
	uint32_t 				RemoteQpn;
	uint32_t 				RemoteQkey;

	//Protection Domain
	struct 	ibv_pd              		*ProtectionDomain;     /* Protection Domain Handle */

	//Memory Regions
	struct 	ibv_mr                  	*MemoryRegion;         /* Memory Region Handle */

};
static struct PubContext g_PubContext;

/*
 * Signals
 */
static bool SIG_SUBSCRIPTIONESTABLISHED = false;
static bool SIG_KILLCMMONITOR = false;

//Memory Region
ibv_mr* mr_instrument;

//Data Pointed To By Memory Region Struct
instrument_t			instrument;

int main(int argc,char *argv[], char *envp[])
{
	//Create the Instrument
	instrument.Symbol[0] = 'N';
    instrument.Symbol[1] = 'V';
    instrument.Symbol[2] = 'D';
    instrument.Symbol[3] = 'A';
    instrument.Symbol[4] = '\0';
    instrument.Value = 1.0;

	int op;
	while ((op = getopt(argc, argv, "l:m:")) != -1)
	{
		switch (op)
		{
		case 'l':
			local_string = optarg;
			break;
		case 'm':
			mcast_string = optarg;
			break;
		default:
			PrintUsage();
			return -1;
		}
	}
    int timeToRun = 900;
	g_PubContext.local_addr = (struct sockaddr *)&g_PubContext.local_in;
	g_PubContext.mcast_addr = (struct sockaddr *)&g_PubContext.mcast_in;


	fprintf(stdout, "********  ********  ********  ********\n");
	fprintf(stdout,"MARKET DATA PUBLISHER\n");
	fprintf(stderr, "Local IPoIB Address:      %s\n", local_string);
	fprintf(stderr, "Multicast Group:      %s\n", mcast_string);
	fprintf(stdout, "********  ********  ********  ********\n\n");

	if(RDMAPublisherInit() != 0)
	{
		fprintf(stderr, "Exiting - Failed Create the RDMA Channel.\n");
		return 0;
	}

	/*
	 * Start Monitoring the CM Event Channel
	 */
	int data = 5;
	pthread_t 				CMEventChannelMonitorThread;
	fprintf(stderr, "Starting CM Event Monitor ...\n");
	pthread_create(&CMEventChannelMonitorThread, NULL, MonitorCMEventChannel, (void*) &data);

	//Wait for the Signal that the Channel is Established
	while(!SIG_SUBSCRIPTIONESTABLISHED) { sleep(5); }

	//Register the Memory Region
	mr_instrument = create_MEMORY_REGION(&instrument, sizeof(instrument_t));

	//Create a Send WQE - Containing the Address
	ibv_send_wr* sWQE = create_SEND_WQE(&instrument, sizeof(instrument_t), mr_instrument);

    //chrono::time_point<chrono::system_clock> start;
    //chrono::duration<double> delta;

    //start = chrono::system_clock::now();
    do {
		//update the instrument
		instrument.Value++;

		fprintf(stderr, "Sending Update\n");
        fprintf(stderr, "NVDA Stock Price @ %f\n", instrument.Value);
		//Post the Receive WQE
		post_SEND_WQE(sWQE);

		PollCQ();

		sleep(5);
       // delta = chrono::system_clock::now() - start;
    //} while (delta.count() <= timeToRun);
} while (1);

	CleanUpPubContext();

	return 0;
}

/*
 * Returns Number of Completions Received
 */
int PollCQ()
{
	struct ibv_wc wc;
	int ret = 0;

	fprintf(stderr, "Waiting for CQE\n");
	do {
		ret = ibv_poll_cq(g_PubContext.CompletionQueue, 10, &wc);
	} while(ret == 0);
	fprintf(stderr, "Received %u CQE Elements\n", ret);
	fprintf(stderr, "WRID(%llu)\tStatus(%u)\n", wc.wr_id, wc.status);
	return ret;
}

void PrintUsage()
{
	fprintf(stdout, "usage: sub [ -l ip ] [-m mcast_ip] \n");
	//printf("\t[-l ip] - bind to the local interface associated with this IPoIB Address.\n");
	//printf("\t[-m ip] - bind to the local interface associated with this IPoIB Address.\n");
}

void OnReceiveUpdate()
{
	fprintf(stdout, "Received an Update (%s,%f)\n", instrument.Symbol, instrument.Value);
	return;
}

static int get_addr(char *dst, struct sockaddr *addr)
{
	struct addrinfo *res;
	int ret;
	ret = getaddrinfo(dst, NULL, NULL, &res);
	if (ret)
	{
		fprintf(stderr, "getaddrinfo failed - invalid hostname or IP address\n");
		return -1;
	}
	memcpy(addr, res->ai_addr, res->ai_addrlen);
	freeaddrinfo(res);
	return ret;
}

/*
 * Monitors the Communication Managers Event Channel.
 *
 * The CM event channel will notify the program when joins or multicast error happen.
 */
void *MonitorCMEventChannel(void* data)
{
	struct rdma_cm_event *event;
	int ret = 0;

	do {
		ret = rdma_get_cm_event(g_PubContext.CMEventChannel, &event);
		if(ret != 0)
		{
			fprintf(stderr, "ERROR - MonitorCMEventChannel: Non-Zero Return Code from rdma_get_cm_event.\n");
		}

		switch(event->event)
		{
		case RDMA_CM_EVENT_ADDR_RESOLVED:
			fprintf(stderr, "Received RDMA_CM_EVENT_ADDR_RESOLVED Event\n");
			OnAddressResolved(event);
			break;
		case RDMA_CM_EVENT_MULTICAST_JOIN:
			fprintf(stderr, "Received RDMA_CM_EVENT_MULTICAST_JOIN Event\n");
			OnMulticastJoin(event);
			break;

		//TODO: Error Cases
		case RDMA_CM_EVENT_ADDR_ERROR:
			fprintf(stdout, "Address Resolution Error: event: %s, error: %d\n", rdma_event_str(event->event), event->status);
			break;
		case RDMA_CM_EVENT_ROUTE_ERROR:
			fprintf(stdout, "Route Error: event: %s, error: %d\n", rdma_event_str(event->event), event->status);
			break;
		case RDMA_CM_EVENT_MULTICAST_ERROR:
			fprintf(stdout, "Multicast Error: event: %s, error: %d\n", rdma_event_str(event->event), event->status);
			break;
		default:
			break;

		}

		rdma_ack_cm_event(event);
	} while(!SIG_KILLCMMONITOR);

	return 0;
}

/*
 * Initilization
 *
 * Create the CM Event Channel, the Connection Identifier, Bind the application to a local address
 */
int RDMAPublisherInit()
{
	int ret = 0;
	g_PubContext.CMEventChannel = NULL;
	g_PubContext.CMContext = (CMContext_t*) malloc(sizeof(CMContext_t));
	g_PubContext.CMContext->id = 1;

	// Open a Channel to the Communication Manager used to receive async events from the CM.
	g_PubContext.CMEventChannel = rdma_create_event_channel();
	if(!g_PubContext.CMEventChannel)
	{
		fprintf(stderr, "Failed to Open CM Event Channel");
		CleanUpPubContext();
		return -1;
	}

	ret = rdma_create_id(g_PubContext.CMEventChannel, &g_PubContext.CMId, g_PubContext.CMContext, RDMA_PS_UDP);
	if(ret != 0)
	{
		fprintf(stderr, "ERROR RDMAPublisherInit: Failed to Create CM ID");
		CleanUpPubContext();
		return -1;
	}

	if(get_addr(local_string,(struct sockaddr*)&g_PubContext.local_in) != 0)
	{
		fprintf(stderr, "ERROR RDMAPublisherInit: Failed to Resolve Local Address\n");
		CleanUpPubContext();
		return -1;
	}

	if(get_addr(mcast_string,(struct sockaddr*)&g_PubContext.mcast_in) != 0)
	{
		fprintf(stderr, "ERROR RDMAPublisherInit: Failed to Resolve Multicast Address Address\n");
		CleanUpPubContext();
		return -1;
	}

	//Print Out the Resolved Address Information
	fprintf(stdout, "*** *** *** Local Address *** *** *** ***\n");
	char str[INET6_ADDRSTRLEN];
	inet_ntop(AF_INET6, &g_PubContext.local_in.sin6_addr, str, INET6_ADDRSTRLEN);
	fprintf(stdout, "Address: %s\n", str);
	fprintf(stdout, "*** *** *** *** *** *** *** ***\n");
	
	fprintf(stdout, "*** *** *** Multicast Address *** *** *** ***\n");
	inet_ntop(AF_INET6, &g_PubContext.mcast_in.sin6_addr, str, INET6_ADDRSTRLEN);
	fprintf(stdout, "Address: %s\n", str);
	fprintf(stdout, "*** *** *** *** *** *** *** ***\n");

	ret = rdma_bind_addr(g_PubContext.CMId, g_PubContext.local_addr);
	if(ret != 0 )
	{
		fprintf(stderr, "ERROR RDMAPublisherInit: Couldn't bind to local address.\n");
	}

	ret = rdma_resolve_addr(g_PubContext.CMId, 
				(struct sockaddr*)&g_PubContext.local_in, 
				(struct sockaddr*)&g_PubContext.mcast_in, 
				2000);
	if(ret != 0 )
	{
		fprintf(stderr, "ERROR RDMAPublisherInit: Couldn't resolve local address and or mcast address.\n");
	}

	return 0;
}

int OnAddressResolved(struct rdma_cm_event *event)
{
	int ret;

	/*
	 * Get the CM Id from the Event
	 */
	g_PubContext.CMId = event->id;

	/*
	 * Create the QP
	 */
	ret = RDMACreateQP();
	if(ret != 0)
	{
		fprintf(stderr, "ERROR OnAddressResolved - Couldn't Create QP\n");
		return -1;
	}

	ret = rdma_join_multicast(g_PubContext.CMId, g_PubContext.mcast_addr, NULL);
	if(ret != 0)
	{
		fprintf(stderr, "ERROR OnAddressResolved - RDMA MC Join Failed %d\n", ret);
		return -1;
	}

	return 0;
}

int OnMulticastJoin(struct rdma_cm_event *event)
{
	rdma_ud_param *param;
	param = &event->param.ud;

	char buf[40];

	inet_ntop(AF_INET6, param->ah_attr.grh.dgid.raw, buf, 40);

	g_PubContext.RemoteQpn = param->qp_num;
	g_PubContext.RemoteQkey = param->qkey;
	g_PubContext.AddressHandle = ibv_create_ah(g_PubContext.ProtectionDomain, &param->ah_attr);
	if (!g_PubContext.AddressHandle)
	{
		fprintf(stderr, "ERROR OnMulticastJoin - Failed to create the Address Handle\n");
		return -1;
	}

	fprintf(stderr, "Joined Multicast Group QPN(%d) QKey(%d)\n", g_PubContext.RemoteQpn, g_PubContext.RemoteQkey);
	SIG_SUBSCRIPTIONESTABLISHED = true;

	return 0;
}

int RDMACreateQP()
{
	int ret;
	struct ibv_qp_init_attr qp_init_attr;
	CQContext_t *cqcontext = (CQContext_t*)malloc(sizeof(CQContext_t));
	QPContext_t	*qpcontext = (QPContext_t*)malloc(sizeof(QPContext_t));

	cqcontext->id = 1;
	qpcontext->id = 1;

	//Create a Protection Domain
	g_PubContext.ProtectionDomain = ibv_alloc_pd(g_PubContext.CMId->verbs);
	if(!g_PubContext.ProtectionDomain)
	{
		fprintf(stderr, "ERROR - RDMACreateQP: Couldn't allocate protection domain\n");
		return -1;
	}

	/*
	 * Create a Completion Channel - Used to Handle CQE in a callback model.
	 */
	g_PubContext.CompletionChannel = ibv_create_comp_channel(g_PubContext.CMId->verbs);
	if(!g_PubContext.CompletionChannel)
	{
		fprintf(stderr, "ERROR - RDMACreateQP: Coun'dn't Create a Completion CHannel\n");
		return -1;
	}

	/*Create a completion Queue */
	g_PubContext.CompletionQueue = ibv_create_cq(g_PubContext.CMId->verbs, 10, &cqcontext, g_PubContext.CompletionChannel, 1);
	if(!g_PubContext.CompletionQueue)
	{
		fprintf(stderr, "ERROR - RDMACreateQP: Couldn't create completion queue\n");
		return -1;
	}

	/* create the Queue Pair */
	memset(&qp_init_attr, 0, sizeof(qp_init_attr));

	qp_init_attr.qp_type = IBV_QPT_UD;
	qp_init_attr.sq_sig_all = 1;
	qp_init_attr.send_cq = g_PubContext.CompletionQueue;
	qp_init_attr.recv_cq = g_PubContext.CompletionQueue;
	qp_init_attr.cap.max_send_wr = 1;
	qp_init_attr.cap.max_recv_wr = 1;
	qp_init_attr.cap.max_send_sge = 1;
	qp_init_attr.cap.max_recv_sge = 1;
	qp_init_attr.qp_context = &qpcontext;


	ret = rdma_create_qp(g_PubContext.CMId, g_PubContext.ProtectionDomain, &qp_init_attr);
	if(ret != 0)
	{
		fprintf(stderr, "ERROR - RDMACreateQP: Couldn't Create Queue Pair Error(%d)\n", ret);
		return -1;
	}
	return 0;
}

ibv_mr *create_MEMORY_REGION(void* buffer, size_t bufferlen)
{
	ibv_mr* tmpmr = (ibv_mr*)malloc(sizeof(ibv_mr));
	int mr_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
	tmpmr = ibv_reg_mr(g_PubContext.ProtectionDomain, buffer, bufferlen, mr_flags);
	if(!tmpmr)
	{
		fprintf(stderr, "ERROR - create_MEMORY_REGION: Couldn't Register memory region\n");
		return NULL;
	}

	fprintf(stdout, "Memory Region was registered with addr=%p, lkey=0x%x, rkey=0x%x, flags=0x%x\n",
			buffer, tmpmr->lkey, tmpmr->rkey, mr_flags);

	return tmpmr;
}

/*
 * Post a Send WQE
 */
int post_SEND_WQE(ibv_send_wr* ll_wqe)
{
	int ret;
	struct ibv_send_wr *bad_wqe = NULL;

	ret = ibv_post_send(g_PubContext.CMId->qp, ll_wqe, &bad_wqe);
	if(ret != 0)
	{
		fprintf(stderr, "ERROR post_SEND_WQE - Couldn't Post Send WQE\n");
		return -1;
	}
	return 0;
}

ibv_send_wr* create_SEND_WQE(void* buffer, size_t bufferlen, ibv_mr* bufferMemoryRegion)
{
	struct ibv_send_wr *wqe;
	struct ibv_sge *sge;

	wqe = (ibv_send_wr *)malloc(sizeof(ibv_send_wr));
	sge = (ibv_sge *)malloc(sizeof(ibv_sge));

	memset(wqe, 0, sizeof(ibv_send_wr));
	memset(sge, 0, sizeof(ibv_sge));

	wqe->wr_id = 1;
	wqe->next = NULL;
	wqe->opcode = IBV_WR_SEND;
	wqe->sg_list = sge;
	wqe->num_sge = 1;
	wqe->send_flags = IBV_SEND_SIGNALED;

	wqe->wr.ud.ah = g_PubContext.AddressHandle;
	wqe->wr.ud.remote_qpn = g_PubContext.RemoteQpn;
	wqe->wr.ud.remote_qkey = g_PubContext.RemoteQkey;

	sge->addr = (uintptr_t)buffer;
	sge->length = bufferlen;
	sge->lkey = bufferMemoryRegion->lkey;

	return wqe;
}



void CleanUpPubContext()
{
	if(g_PubContext.CMEventChannel != NULL)
	{
		rdma_destroy_event_channel(g_PubContext.CMEventChannel);
	}

	if(g_PubContext.CMId != NULL)
	{
		if(rdma_destroy_id(g_PubContext.CMId) != 0)
		{
			//fprintf(stderr, "CleanUpCMContext: Failed to destroy Connection Manager Id\n");
		}
	}
}
