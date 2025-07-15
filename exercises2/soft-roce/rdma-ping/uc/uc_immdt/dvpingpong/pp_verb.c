#include "pp_verb.h"

#define DBG_RC 0
int pp_create_cq_qp_verb(const struct pp_context *ppctx,
			 struct pp_verb_cq_qp *ppv)
{
	int ret;

	struct ibv_cq_init_attr_ex cq_init_attr_ex = {
		.cqe = (1 << PP_MAX_LOG_CQ_SIZE) - 1,
		.cq_context = NULL,
		.channel = NULL,
		.comp_vector = 0,
	};
	 cq_init_attr_ex.wc_flags =
		       IBV_WC_EX_WITH_BYTE_LEN | IBV_WC_EX_WITH_IMM | IBV_WC_EX_WITH_QP_NUM |
		             IBV_WC_EX_WITH_SRC_QP |
			           IBV_WC_EX_WITH_COMPLETION_TIMESTAMP; 
	ppv->cq_ex = ibv_create_cq_ex(ppctx->ibctx, &cq_init_attr_ex);
	if (!ppv->cq_ex) {
		ERR("ibv_create_cq_ex() failed");
		return errno;
	}

	struct ibv_qp_init_attr_ex init_attr = {
		.sq_sig_all = 1,
		.send_cq = ibv_cq_ex_to_cq(ppv->cq_ex),
		.recv_cq = ibv_cq_ex_to_cq(ppv->cq_ex),
		.cap = ppctx->cap,
#if DBG_RC
		.qp_type = IBV_QPT_RC,
#else
		.qp_type = IBV_QPT_UC,
#endif
		.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS,

		.pd = ppctx->pd,
	};
	//init_attr.send_ops_flags |= IBV_QP_EX_WITH_RDMA_WRITE_WITH_IMM;
	ppv->qp = ibv_create_qp_ex(ppctx->ibctx, &init_attr);
	if (!ppv->qp) {
		ERR("ibv_create_qp_ex() failed");
		ret = errno;
		goto fail_create_qp;
	}

	struct ibv_qp_attr attr = {
		.qp_state = IBV_QPS_INIT,
		.pkey_index = 0,
		.port_num = ppctx->port_num,
		.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ,
	};
	ret = ibv_modify_qp(ppv->qp, &attr,
			    IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	if (ret) {
		perror("ibv_modify_qp");
		goto fail_modify_qp;
	}

	return 0;

fail_modify_qp:
	ibv_destroy_qp(ppv->qp);
fail_create_qp:
	ibv_destroy_cq(ibv_cq_ex_to_cq(ppv->cq_ex));

	return ret;
}

void pp_destroy_cq_qp_verb(struct pp_verb_cq_qp *ppv)
{
	ibv_destroy_qp(ppv->qp);
	ibv_destroy_cq(ibv_cq_ex_to_cq(ppv->cq_ex));
}

int pp_move2rts_verb(struct pp_context *ppc, struct ibv_qp *qp,
		     int my_sgid_idx, uint32_t my_sq_psn,
		     struct pp_exchange_info *peer)
{
#if DBG_RC
	int ret;

	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= IBV_MTU_1024,
		.dest_qp_num		= peer->qpn,
		.rq_psn			= peer->psn,
		.max_dest_rd_atomic	= 1,
		.min_rnr_timer		= 12,
		.ah_attr		= {
			.is_global	= 0,
			.src_path_bits	= 0,
			.port_num	= ppc->port_num,
		}
	};

	if (ppc->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
		attr.ah_attr.is_global = 1;
		attr.ah_attr.grh.hop_limit = 64;
		attr.ah_attr.grh.sgid_index = my_sgid_idx;
		attr.ah_attr.grh.dgid = peer->gid;
	attr.timeout	    = 12;
	attr.retry_cnt	    = 3;
	attr.retry_cnt	    = 3;
	} else {
		attr.ah_attr.dlid = peer->lid;
		printf("=DEBUG:%s:%d: attr.ah_attr.dlid %d\n", __func__, __LINE__, attr.ah_attr.dlid);
	}
	ret = ibv_modify_qp(qp, &attr,
			    IBV_QP_STATE              |
			    IBV_QP_AV                 |
			    IBV_QP_PATH_MTU           |
			    IBV_QP_DEST_QPN           |
			    IBV_QP_RQ_PSN             |
			    IBV_QP_MAX_DEST_RD_ATOMIC |
			    IBV_QP_MIN_RNR_TIMER);
	if (ret) {
		ERR("Failed to modify to RTR: ret %d errno %d", ret, errno);
		return ret;
	}

	attr.qp_state	    = IBV_QPS_RTS;
	attr.timeout	    = 12;
	attr.retry_cnt	    = 3;
	attr.rnr_retry	    = 3;
	attr.sq_psn	    = my_sq_psn;
	attr.max_rd_atomic  = 1;
	ret = ibv_modify_qp(qp, &attr,
			    IBV_QP_STATE              |
			    IBV_QP_TIMEOUT            |
			    IBV_QP_RETRY_CNT          |
			    IBV_QP_RNR_RETRY          |
			    IBV_QP_SQ_PSN             |
			    IBV_QP_MAX_QP_RD_ATOMIC);
	if (ret) {
		ERR("Failed to modify to RTR: ret %d errno %d", ret, errno);
		return ret;
	}

#else
	struct ibv_qp_attr attr = {
		.qp_state		= IBV_QPS_RTR,
		.path_mtu		= IBV_MTU_4096,
		.dest_qp_num		= peer->qpn,
		.rq_psn			= peer->psn,
		.max_dest_rd_atomic	= 1,
		//.min_rnr_timer		= 12,
		.ah_attr		= {
			.dlid = peer->lid,
			.is_global	= 0,
			.src_path_bits	= 0,
			.port_num	= ppc->port_num,
		}
	};

	if (ppc->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
		attr.ah_attr.is_global = 1;
		attr.ah_attr.grh.hop_limit = 64;
		attr.ah_attr.grh.sgid_index = my_sgid_idx;
		attr.ah_attr.grh.dgid = peer->gid;
	//attr.timeout	    = 12;
	//bbbb.timeout	    = 12;
	//attr.retry_cnt	    = 3;
	//attr.retry_cnt	    = 3;
	} else {
		attr.ah_attr.dlid = peer->lid;
		printf("=DEBUG:%s:%d: attr.ah_attr.dlid %d\n", __func__, __LINE__, attr.ah_attr.dlid);
	}
	if (ibv_modify_qp(qp, &attr,
			  IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
				  IBV_QP_DEST_QPN | IBV_QP_RQ_PSN)) {
		fprintf(stderr, "Failed to modify QP to RTR\n");
		return 1;
	}

	attr.qp_state = IBV_QPS_RTS;
	attr.sq_psn = my_sq_psn;
	if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
		fprintf(stderr, "Failed to modify QP to RTS\n");
		return 1;
	}

	return 0;
#endif
	INFO("Local qp 0x%x(%d) moved to RTS state\n", qp->qp_num, qp->qp_num);
	return 0;
}

void prepare_recv_wr_verb(struct pp_verb_ctx *ppv, struct ibv_recv_wr wrr[],
			  struct ibv_sge sglists[], int max_wr_num,
			  uint64_t wr_id)
{
	int i;

	for (i = 0; i < max_wr_num; i++) {
		memset(ppv->ppc.mrbuf[i], 0, ppv->ppc.mrbuflen);

		sglists[i].lkey = ppv->ppc.mr[i]->lkey;
		sglists[i].addr = (uint64_t)ppv->ppc.mrbuf[i];
		sglists[i].length = ppv->ppc.mrbuflen;

		if (i < max_wr_num - 1)
			wrr[i].next = &wrr[i+1];
		else
			wrr[i].next = NULL;

		wrr[i].wr_id = wr_id + i;
		wrr[i].sg_list = &sglists[i];
		wrr[i].num_sge = 1;
	}
}

void prepare_send_wr_verb(struct pp_verb_ctx *ppv, struct ibv_send_wr wrs[],
			  struct ibv_sge sglists[], struct pp_exchange_info *peer,
			  int max_wr_num, uint64_t wr_id, int opcode, bool initbuf)
{
	int i;

	for (i = 0; i < max_wr_num; i++) {
		if (initbuf) {
			mem_string(ppv->ppc.mrbuf[i], ppv->ppc.mrbuflen);
			*ppv->ppc.mrbuf[i] = i % 16 + '0';
		}

		sglists[i].lkey = ppv->ppc.mr[i]->lkey;
		sglists[i].addr = (uint64_t)ppv->ppc.mrbuf[i];
		sglists[i].length = ppv->ppc.mrbuflen;

		if (i < max_wr_num - 1)
			wrs[i].next = &wrs[i+1];
		else
			wrs[i].next = NULL;

		wrs[i].wr_id = wr_id + i;
		wrs[i].sg_list = &sglists[i];
		wrs[i].num_sge = 1;
		wrs[i].imm_data = 0x10203040 + i;
		wrs[i].opcode = opcode;
		wrs[i].send_flags = IBV_SEND_SIGNALED;

		if (wrs[i].opcode == IBV_WR_RDMA_WRITE_WITH_IMM) {
			wrs[i].wr.rdma.remote_addr = (uint64_t)peer->addr[i];
			wrs[i].wr.rdma.rkey = peer->mrkey[i];
		}
	}
}

int poll_cq_verb(struct pp_verb_ctx *ppv, int max_wr_num, bool for_recv)
{
	struct ibv_wc wcs[PP_MAX_WR];
	int cq_recved = 0, cqn, i;

	do {
		do {
			cqn = ibv_poll_cq(ibv_cq_ex_to_cq(ppv->cqqp.cq_ex),
					  max_wr_num - cq_recved, wcs);
			usleep(1000 * 10);
		} while (cqn == 0);

		if (cqn < 0) {
			ERR("ibv_poll_cq failed, %d/%d\n", cq_recved, max_wr_num);
			return -1;
		}

		for (i = 0; i < cqn; i++) {
			if (wcs[i].status != IBV_WC_SUCCESS) {
				ERR("Failed status %s(%x) for wr_id 0x%x\n",
				    ibv_wc_status_str(wcs[i].status), wcs[i].status,
				    (int)wcs[i].wr_id);
				return -1;
			}
		}
		if (for_recv)
			for (i = 0; i < cqn; i++)
				dump_msg_short(cq_recved + i, &ppv->ppc);
		else
			INFO("Polled %d/%d CQEs for post_send..\n", cq_recved + cqn, max_wr_num);

		cq_recved += cqn;
	} while (cq_recved < max_wr_num);

	return 0;
}
