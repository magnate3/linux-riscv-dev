#include <stdio.h>
#include <stdint.h>
#include <rdma/rdma_cma.h>
#include <math.h>
#include <inttypes.h>

struct qp_attr {
	uint64_t gid_global_interface_id;// Store the gid fields separately because I
	uint64_t gid_global_subnet_prefix; // don't like unions. Needed for RoCE only

	int lid;// A queue pair is identified by the local id (lid)
	int qpn;// of the device port and its queue pair number (qpn)
	int psn;
};


#define IBV_MTU_SIZE IBV_MTU_1024
#define S_QPA sizeof(struct qp_attr)
#ifndef IB_PHYS_PORT
#define IB_PHYS_PORT 2
#endif

#ifndef GID_INDEX 
#define GID_INDEX 1
#endif


#ifndef SL
#define SL 0 
#endif

#define GET_MIN(A, B) ((A) < (B) ? (A) : (B))
#define GET_MAX(A, B) ((A) < (B) ? (B) : (A))

#define DEFAULT_TIMEOUT 	14
#define RC_TRANSPORT 0
#define UC_TRANSPORT 1
#define UD_TRANSPORT 2

#define READ_OPERATOR 2
#define WRITE_OPERATOR 1
#define WRITE_IMM_OPERATOR 3
#define SEND_OPERATOR 0


int reconnect_qp(struct ibv_qp *qp, int my_psn, struct qp_attr *dest, int send_transport, uint32_t newtimeout, int sl);
uint32_t get_current_my_psn(struct ibv_qp *qp, int isClient);
int modify_qp(struct ibv_qp *qp, int sl, struct qp_attr *dest);
