#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <rdma/rdma_cma.h>

#include "share.h"

// Transition connected QP indexed qp_i through RTR and RTS stages
int reconnect_qp(struct ibv_qp *qp, int my_psn, struct qp_attr *dest, int send_transport, uint32_t newtimeout, int sl)
{
	struct ibv_qp_attr conn_attr1 = {
		.qp_state= IBV_QPS_RESET,
	};
	if (ibv_modify_qp(qp, &conn_attr1,
				IBV_QP_STATE)) {
		fprintf(stderr, "Failed to modify conn. QP to RESET\n");
		return -1;
	}

	struct ibv_qp_attr conn_attr0 = {
		.qp_state= IBV_QPS_INIT,
		.pkey_index= 0,
		.port_num= IB_PHYS_PORT,
		.qp_access_flags= IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ
	};
	if (ibv_modify_qp(qp, &conn_attr0,
				IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
		fprintf(stderr, "Failed to modify conn. QP to INIT\n");
		return -1;
	}

	struct ibv_qp_attr conn_attr = {
		.qp_state= IBV_QPS_RTR,
		.path_mtu= IBV_MTU_SIZE,
		.dest_qp_num= dest->qpn,
		.rq_psn= dest->psn,
		.ah_attr= {
			.is_global= 1,
			.dlid= 0,
			.sl= sl,
			.src_path_bits= 0,
			.port_num= IB_PHYS_PORT
		}
	};

	conn_attr.ah_attr.grh.dgid.global.interface_id = 
		dest->gid_global_interface_id;
	conn_attr.ah_attr.grh.dgid.global.subnet_prefix = 
		dest->gid_global_subnet_prefix;

	conn_attr.ah_attr.grh.sgid_index = GID_INDEX;
	conn_attr.ah_attr.grh.hop_limit = 1;

	int rtr_flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
		| IBV_QP_RQ_PSN;
	if(!send_transport) {
		conn_attr.max_dest_rd_atomic = 16;
		conn_attr.min_rnr_timer = 12;
		rtr_flags |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
	}
	if (ibv_modify_qp(qp, &conn_attr, rtr_flags)) {
		fprintf(stderr, "Failed to modify QP to RTR\n");
		return 1;
	}

	memset(&conn_attr, 0, sizeof(conn_attr));
	conn_attr.qp_state    = IBV_QPS_RTS;
	conn_attr.sq_psn    = my_psn;
	int rts_flags = IBV_QP_STATE | IBV_QP_SQ_PSN;
	int new_timeout = (int)ceil(log2( ((double)newtimeout/4.096)));
	if(!send_transport) {
		fprintf(stderr,"new_timeout %d, %"PRIu32"\n", new_timeout*2, newtimeout);
		conn_attr.timeout = GET_MIN(new_timeout*2, 14);
		conn_attr.retry_cnt = 1;
		conn_attr.rnr_retry = 1;
		conn_attr.max_rd_atomic = 16;
		rts_flags |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
			IBV_QP_MAX_QP_RD_ATOMIC;
	}
	if (ibv_modify_qp(qp, &conn_attr, rts_flags)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}

	return 0;
}

uint32_t get_current_my_psn(struct ibv_qp *qp, int isClient){
	struct ibv_qp_attr attr;
	struct ibv_qp_init_attr init_attr;

	if (ibv_query_qp(qp, &attr,
				   IBV_QP_RQ_PSN, &init_attr)) {
		fprintf(stderr, "Failed to query QP state\n");
		return -1;
	}
	if(!isClient){
		printf("remote psn %d\n", attr.rq_psn);
		//printf("current_psn %d\n", attr.sq_psn);
		//printf("dest qp_num  %d\n", attr.dest_qp_num);
	}
	return attr.rq_psn;	
	
	
	//	if(attr.qp_state == IBV_QPS_ERR)
	//		return -1;
			//printf("qp_state is in err\n");
	//	if(attr.qp_state == IBV_QPS_RTS){
	/*		if(isClient)	
				return attr.sq_psn;
			else
				return attr.rq_psn;
	*///	}
			//printf("qp_state is in rts _state\n");
	return -1;
}

int modify_qp(struct ibv_qp *qp, int sl, struct qp_attr *dest){
		struct ibv_qp_attr conn_attr = {
		.qp_state= IBV_QPS_RTS,
		.ah_attr= {
			.is_global= 1,
			.dlid= 0,
			.sl= sl,
			.src_path_bits= 0,
			.port_num= IB_PHYS_PORT
		}
	};
	conn_attr.ah_attr.grh.dgid.global.interface_id = 
		dest->gid_global_interface_id;
	conn_attr.ah_attr.grh.dgid.global.subnet_prefix = 
		dest->gid_global_subnet_prefix;

	conn_attr.ah_attr.grh.sgid_index = GID_INDEX;
	conn_attr.ah_attr.grh.hop_limit = 1;

	int rtr_flags = IBV_QP_STATE |  IBV_QP_AV ;
	if (ibv_modify_qp(qp, &conn_attr, rtr_flags)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}
	
}

