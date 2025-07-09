/*
 * Copyright (c) 2005 Mellanox Technologies. All rights reserved.
 *
 */

#include "rdma_mt.h"


static void UNUSED dump_qp_init_attr(struct ibv_qp_init_attr *qp_init_attr)
{
	DEBUG("\nDumping ibv_qp_init_attr:\n");
	DEBUG("\t    qp_init_attr.cap.max_send_wr  :%x\n", qp_init_attr->cap.max_send_wr);
	DEBUG("\t    qp_init_attr.cap.max_send_wr  :%x\n", qp_init_attr->cap.max_send_wr);
	DEBUG("\t    qp_init_attr.cap.max_send_sge :%x\n", qp_init_attr->cap.max_send_sge);
	DEBUG("\t    qp_init_attr.cap.max_recv_sge :%x\n", qp_init_attr->cap.max_recv_sge);
	DEBUG("\t    qp_init_attr.send_cq          :%p\n", qp_init_attr->send_cq);
	DEBUG("\t    qp_init_attr.recv_cq          :%p\n", qp_init_attr->recv_cq);
	DEBUG("\t    qp_init_attr.qp_type          :%x\n", qp_init_attr->qp_type);
	DEBUG("\t    qp_init_attr.sq_sig_all       :%x\n", qp_init_attr->sq_sig_all);
	DEBUG("\t    qp_init_attr.srq              :%p\n", qp_init_attr->srq);
}


static void UNUSED dump_qp_attr(struct thread_context_t *t_ctx)
{
	struct ibv_qp_attr *qp_attr;

	qp_attr = &t_ctx->qp_attr;

	DEBUG("\nibv_qp_attr:\n");
	DEBUG("\tqp_attr.pkey_index          :0x%x\n", qp_attr->pkey_index);
	DEBUG("\tqp_attr.port_num            :0x%x\n", qp_attr->port_num);
	DEBUG("\tqp_attr.sq_psn              :0x%x\n", qp_attr->sq_psn);
	if (t_ctx->qp_type == IBV_QPT_UD) {
		DEBUG("\n\tqp_type == IBV_QPT_UD:\n");
		DEBUG("\tqp_attr.qkey                :0x%x\n", qp_attr->qkey);
	}

	if ((t_ctx->qp_type == IBV_QPT_UC) || (t_ctx->qp_type == IBV_QPT_RC)) {
		DEBUG("\n\tqp_type == IBV_QPT_UC | IBV_QPT_RC:\n");
		DEBUG("\tqp_attr.qp_access_flags     :0x%x\n", qp_attr->qp_access_flags);
		DEBUG("\tqp_attr.qkey                :0x%x\n", qp_attr->qkey);
		DEBUG("\tqp_attr.path_mtu            :0x%x\n", qp_attr->path_mtu);
		DEBUG("\tqp_attr.rq_psn              :0x%x\n", qp_attr->rq_psn);
		DEBUG("\tqp_attr.dest_qp_num         :0x%x\n", qp_attr->dest_qp_num);

		DEBUG("\n\tqp_type == IBV_QPT_UC | IBV_QPT_RC:\n");
		DEBUG("\t\tqp_attr.ah_attr.is_global      :0x%x\n", qp_attr->ah_attr.is_global);
		DEBUG("\t\tqp_attr.ah_attr.dlid           :0x%x\n", qp_attr->ah_attr.dlid);
		DEBUG("\t\tqp_attr.ah_attr.sl             :0x%x\n", qp_attr->ah_attr.sl);
		DEBUG("\t\tqp_attr.ah_attr.src_path_bits  :0x%x\n", qp_attr->ah_attr.src_path_bits);
		DEBUG("\t\tqp_attr.ah_attr.port_num       :0x%x\n", qp_attr->ah_attr.port_num);

		if (t_ctx->qp_type == IBV_QPT_RC) {
			DEBUG("\n\tqp_type == IBV_QPT_RC:\n");
			DEBUG("\tqp_attr.timeout             :0x%x\n", qp_attr->timeout);
			DEBUG("\tqp_attr.retry_cnt           :0x%x\n", qp_attr->retry_cnt);
			DEBUG("\tqp_attr.rnr_retry           :0x%x\n", qp_attr->rnr_retry);
			DEBUG("\tqp_attr.min_rnr_timer       :0x%x\n", qp_attr->min_rnr_timer);
			DEBUG("\tqp_attr.max_rd_atomic       :0x%x\n", qp_attr->max_rd_atomic);
			DEBUG("\tqp_attr.max_dest_rd_atomic  :0x%x\n", qp_attr->max_dest_rd_atomic);
		}
	}

	return;
}


static int modify_qp_to_init(struct thread_context_t *t_ctx)

{
	struct ibv_qp_attr qp_attr;
	enum ibv_qp_attr_mask attr_mask = 0;

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));

	/* modify QP state to init */
	qp_attr.qp_state   = IBV_QPS_INIT;
	qp_attr.pkey_index = t_ctx->qp_attr.pkey_index;
	qp_attr.port_num   = t_ctx->qp_attr.port_num;
	attr_mask          = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;

	if (t_ctx->qp_type == IBV_QPT_UD) {
		qp_attr.qkey = t_ctx->qp_attr.qkey;
		attr_mask |= IBV_QP_QKEY;
	}

	if ((t_ctx->qp_type == IBV_QPT_UC) || (t_ctx->qp_type == IBV_QPT_RC)) {
		qp_attr.qp_access_flags = t_ctx->qp_attr.qp_access_flags; /// need to check
		attr_mask |= IBV_QP_ACCESS_FLAGS;
	}

	return ibv_modify_qp(t_ctx->qp, &qp_attr, attr_mask);
}


static int modify_qp_to_rtr(struct thread_context_t *t_ctx)
{
	struct ibv_qp_attr qp_attr;
	enum ibv_qp_attr_mask attr_mask = 0;

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));

	/* modify QP state to RTR */
	qp_attr.qp_state = IBV_QPS_RTR;
	attr_mask = IBV_QP_STATE;

	if ((t_ctx->qp_type == IBV_QPT_UC) || (t_ctx->qp_type == IBV_QPT_RC)) {
		qp_attr.path_mtu    = t_ctx->qp_attr.path_mtu;    /// need to init the parameter
		qp_attr.dest_qp_num = t_ctx->qp_attr.dest_qp_num; /// need to init the parameter
		qp_attr.rq_psn      = t_ctx->qp_attr.rq_psn;          /// need to init the parameter
		
		qp_attr.ah_attr     = t_ctx->qp_attr.ah_attr;     /// need to init the parameter
		attr_mask |= IBV_QP_RQ_PSN | IBV_QP_DEST_QPN | IBV_QP_AV | IBV_QP_PATH_MTU;
	}

	if (t_ctx->qp_type == IBV_QPT_RC) {
		qp_attr.max_dest_rd_atomic = t_ctx->qp_attr.max_dest_rd_atomic;  /// need double-check
		qp_attr.min_rnr_timer      = t_ctx->qp_attr.min_rnr_timer;       /// need double-check
		attr_mask |= IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
	}

	return ibv_modify_qp(t_ctx->qp, &qp_attr, attr_mask);
}


static int modify_qp_to_rts(struct thread_context_t *t_ctx)
{
	struct ibv_qp_attr qp_attr;
	enum ibv_qp_attr_mask attr_mask = 0;

	memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));

	/* modify QP state to RTS */
	qp_attr.qp_state = IBV_QPS_RTS;
	qp_attr.sq_psn   = t_ctx->qp_attr.sq_psn;
	attr_mask        = IBV_QP_STATE | IBV_QP_SQ_PSN;

	if (t_ctx->qp_type == IBV_QPT_RC) {
		qp_attr.timeout       = t_ctx->qp_attr.timeout;
		qp_attr.retry_cnt     = t_ctx->qp_attr.retry_cnt;
		qp_attr.rnr_retry     = t_ctx->qp_attr.rnr_retry;
		qp_attr.max_rd_atomic = t_ctx->qp_attr.max_rd_atomic;
		attr_mask |= IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC;
	}

	if (Debug > 3) {
		dump_qp_attr(t_ctx);
	}
	return ibv_modify_qp(t_ctx->qp, &qp_attr, attr_mask);
}


static int create_av_for_ud(struct thread_context_t *t_ctx)
{
	struct rdma_resource_t* rdma_resource;
	struct user_param_t *user_param;
	struct ibv_ah_attr ah_attr;
	
	rdma_resource  = t_ctx->rdma_resource;
	user_param     = &(rdma_resource->user_param);

	memset(&ah_attr, 0, sizeof(struct ibv_ah_attr));
	ah_attr.port_num      = user_param->ib_port;
	ah_attr.dlid          = t_ctx->remote_lid;
	ah_attr.static_rate   = DEF_STATIC_RATE;
	ah_attr.src_path_bits = 0;
	ah_attr.sl            = DEF_SL;

	// Todo: GRH and Multicast.
#if 0
	if ((MASK_IS_SET(test_rdma_resource_p->test_flags, IS_SUPPORTED_GRH)) ||
	     (MASK_IS_SET(test_rdma_resource_p->test_flags, IS_SUPPORTED_MULTICAST))) {
		ah_attr_p->is_global = 1;

		/* fill the GRH */
		ah_attr_p->grh.dgid = dest_gid;
		ah_attr_p->grh.flow_label = DEF_FLOW_LABEL;
		ah_attr_p->grh.sgid_index = sgid_index;
		ah_attr_p->grh.hop_limit = DEF_HOP_LIMIT;
		ah_attr_p->grh.traffic_class = DEF_TRAFFIC_CLASS;
#endif

	t_ctx->ud_av_hdl = ibv_create_ah(rdma_resource->pd, &ah_attr);
	if (!t_ctx->ud_av_hdl) {
		ERROR("Failed to create address vector for UD.\n");
		return 1;
	}

	return 0;
}


static int set_qp_attr(struct thread_context_t *t_ctx)
{
	struct rdma_resource_t* rdma_resource;
	struct user_param_t *user_param;
	enum ibv_qp_type qp_type;
	struct ibv_qp_attr *qp_attr;
	struct ibv_ah_attr *ah_attr;

	qp_attr       = &(t_ctx->qp_attr);
	ah_attr       = &(qp_attr->ah_attr);
	rdma_resource = t_ctx->rdma_resource;
	user_param    = &(rdma_resource->user_param);
	qp_type       = t_ctx->qp_type;

	qp_attr->pkey_index = DEF_PKEY_IX;
	qp_attr->port_num   = user_param->ib_port;
	qp_attr->sq_psn     = t_ctx->psn;

	if (qp_type == IBV_QPT_UD)
		qp_attr->qkey = t_ctx->qkey;

	if ((qp_type == IBV_QPT_UC) || (qp_type == IBV_QPT_RC)) {
		qp_attr->qp_access_flags = 
			IBV_ACCESS_LOCAL_WRITE |
			IBV_ACCESS_REMOTE_WRITE |
			IBV_ACCESS_REMOTE_READ |
			IBV_ACCESS_REMOTE_ATOMIC |
			IBV_ACCESS_MW_BIND;

		// set ibv_ah_attr
		memset(ah_attr, 0, sizeof(struct ibv_ah_attr));
		ah_attr->port_num      = user_param->ib_port;
		ah_attr->dlid          = t_ctx->remote_lid;
		ah_attr->static_rate   = DEF_STATIC_RATE;
		ah_attr->src_path_bits = 0;
		ah_attr->sl            = user_param->sl;

		qp_attr->path_mtu               = (uint8_t)user_param->path_mtu; /// ???
		qp_attr->rq_psn                 = t_ctx->remote_psn;
		qp_attr->dest_qp_num            = t_ctx->remote_qpn;
		if (qp_type == IBV_QPT_RC) {
			qp_attr->timeout            = user_param->qp_timeout;
			qp_attr->retry_cnt          = user_param->qp_retry_count;
			qp_attr->rnr_retry          = user_param->qp_rnr_retry;
			qp_attr->min_rnr_timer      = user_param->qp_rnr_timer;
			qp_attr->max_rd_atomic      = 128;
			qp_attr->max_dest_rd_atomic = 16; // Todo: why it should be <= 16?
		}
	}
	return 0;
}


int connect_qp(struct thread_context_t *t_ctx)
{
	struct rdma_resource_t* rdma_resource;
	/// struct ib_qp_sync_info_t qp_sync_info_local, qp_sync_info_remote;
	int rc;

	rdma_resource = t_ctx->rdma_resource;

	rc = set_qp_attr(t_ctx);
	if (rc) {
		ERROR("set_qp_attr failed.\n");
		return rc;
	}

	rc = modify_qp_to_init(t_ctx);
	if (rc) {
		ERROR("ibv_modify_qp to INIT failed, rc=%d.\n", rc);
		return rc;
	}

	rc = modify_qp_to_rtr(t_ctx);
	if (rc) {
		ERROR("ibv_modify_qp to RTR failed, rc=%d.\n", rc);
		return rc;
	}

	rc = modify_qp_to_rts(t_ctx);
	if (rc) {
		ERROR("ibv_modify_qp to RTS failed, rc=%d.\n", rc);
		return rc;
	}

	if (t_ctx->qp_type == IBV_QPT_UD) {
		rc = create_av_for_ud(t_ctx);
		if (rc != 0) {
			return rc;
		}
	}

	return 0;
}


static UNUSED int disconnect_qp(struct thread_context_t *t_ctx)
{
	if (t_ctx->ud_av_hdl != NULL) {
		ibv_destroy_ah(t_ctx->ud_av_hdl);
	}
	return 0;
}


static int ib_device_init(struct rdma_resource_t *rdma_resource)
{
	int i;
	int rc;
	int num_devices;
	struct ibv_device *ib_dev = NULL;
	struct user_param_t *user_param = &(rdma_resource->user_param);

	// get device names in the system
	rdma_resource->dev_list = ibv_get_device_list(&num_devices);
	if (!rdma_resource->dev_list) {
		ERROR("failed to get IB devices list.\n");
		goto failure_2;
	}
	if (!num_devices) {
		ERROR("found %d device(s).\n", num_devices);
		goto failure_1;
	}

	for (i = 0; i < num_devices; i ++) {
		if (!strcmp(ibv_get_device_name(rdma_resource->dev_list[i]), user_param->hca_id)) {
			ib_dev = rdma_resource->dev_list[i];
			break;
		}
	}

	if (!ib_dev) {
		ERROR("IB device %s wasn't found.\n", user_param->hca_id);
		goto failure_1;
	}

	// get device handle.
	rdma_resource->ib_ctx = ibv_open_device(ib_dev);
	if (!rdma_resource->ib_ctx) {
		ERROR("failed to open device %s.\n", user_param->hca_id);
		goto failure_1;
	}

	// query port properties
	if (ibv_query_port(rdma_resource->ib_ctx, user_param->ib_port, &rdma_resource->port_attr)) {
		ERROR("ibv_query_port on port failed.\n");
		goto failure_0;
	}

    if (rdma_resource->port_attr.state != IBV_PORT_ACTIVE) {
        ERROR(" Port#%d of %s is not active.\n", user_param->ib_port, user_param->hca_id);
        goto failure_0;
    }

	// Todo: create async event thread.
	if (0) {
		rc = pthread_create(&(rdma_resource->async_event_thread.thread),
			0, async_event_thread, &(rdma_resource->async_event_thread));
	}

	// allocate Protection Domain
	rdma_resource->pd = ibv_alloc_pd(rdma_resource->ib_ctx);
	if (!rdma_resource->pd) {
		ERROR("ibv_alloc_pd failed.\n");
		goto failure_0;
	}

	DEBUG("IB device initializing successfully.\n");
	return 0;
failure_0:
	ibv_close_device(rdma_resource->ib_ctx);

failure_1:
	ibv_free_device_list(rdma_resource->dev_list);
	
failure_2:
	return 1;
}


static int ib_device_destroy(struct rdma_resource_t *rdma_resource)
{
	int ret = 0;
	
	if (rdma_resource->pd) {
		ibv_dealloc_pd(rdma_resource->pd);
	}

	if (rdma_resource->ib_ctx) {
		ibv_close_device(rdma_resource->ib_ctx);
	}

	if (rdma_resource->dev_list)
		ibv_free_device_list(rdma_resource->dev_list);

	return ret;
}


#if 0
static UNUSED int create_srqs(struct rdma_resource_t* rdma_resource)
{
	int i, j;
	struct ibv_srq_init_attr srq_init_attr;
	struct user_param_t *user_param = &(rdma_resource->user_param);

	rdma_resource->srq_hdl_arr = (struct ibv_srq**)malloc(user_param->num_of_srq * sizeof(struct ibv_srq*));
	if (rdma_resource->srq_hdl_arr == NULL) {
		INFO("failed to alloc memory by malloc.");
		return 1;
	}

	memset(&srq_init_attr, 0, sizeof(srq_init_attr));
	srq_init_attr.srq_context  = 0;
	srq_init_attr.attr.max_sge = DEFAULT_SRQ_SGE_SIZE;	// Todo: need further clarify
	srq_init_attr.attr.max_wr  = user_param->srq_size;

	for (i = 0; i < user_param->num_of_srq; i++) {
		rdma_resource->srq_hdl_arr[i] = ibv_create_srq(rdma_resource->pd, &srq_init_attr);
		if (rdma_resource->srq_hdl_arr[i] == NULL) {
			INFO("failed to create SRQ[%d].\n", i);
			break;
		}
	}

	if (i != user_param->num_of_srq) {
		for (j = 0; j < i; j++) {
			ibv_destroy_srq(rdma_resource->srq_hdl_arr[i]);
		}
		return 1;
	}

	return 0;
}


static int destroy_srqs(struct rdma_resource_t* rdma_resource)
{
	int i;
	struct user_param_t *user_param = &(rdma_resource->user_param);

	for (i = 0; i < user_param->num_of_srq; i++) {
		ibv_destroy_srq(rdma_resource->srq_hdl_arr[i]);
	}

	free(rdma_resource->srq_hdl_arr);
	return 0;
}
#endif

static int create_cq(struct thread_context_t *t_ctx)
{
	struct rdma_resource_t* rdma_resource = t_ctx->rdma_resource;
	struct user_param_t *user_param = &(rdma_resource->user_param);

	t_ctx->send_comp_channel = ibv_create_comp_channel(rdma_resource->ib_ctx);
	if (t_ctx->send_comp_channel == NULL) {
		ERROR("Failed to create completion channel.\n");
		return 1;
	}

	t_ctx->send_cq = ibv_create_cq(rdma_resource->ib_ctx, user_param->cq_size, NULL, t_ctx->send_comp_channel, 0);
	if (t_ctx->send_cq == NULL) {
		ibv_destroy_comp_channel(t_ctx->send_comp_channel);
		ERROR("Failed to create CQ with %u entries.\n", user_param->cq_size);
		return 1;
	}

	ibv_req_notify_cq(t_ctx->send_cq, 0);
#if 0
	rc = pthread_create(&(cq_ctx->thread), NULL, cq_poll_thread, cq_ctx);
	if (rc != 0) {
		ibv_destroy_cq(cq_ctx->cq_hdlr);
		ibv_destroy_comp_channel(cq_ctx->comp_channel);
		break;
	}
#endif

	t_ctx->recv_comp_channel = ibv_create_comp_channel(rdma_resource->ib_ctx);
	if (t_ctx->recv_comp_channel == NULL) {
		ERROR("Failed to create completion channel.\n");
		return 1;
	}

	t_ctx->recv_cq = ibv_create_cq(rdma_resource->ib_ctx, user_param->cq_size, NULL, t_ctx->recv_comp_channel, 0);
	if (t_ctx->recv_cq == NULL) {
		ibv_destroy_comp_channel(t_ctx->recv_comp_channel);
		ERROR("Failed to create CQ with %u entries.\n", user_param->cq_size);
		return 1;
	}

	ibv_req_notify_cq(t_ctx->recv_cq, 0);
#if 0
	rc = pthread_create(&(cq_ctx->thread), NULL, cq_poll_thread, cq_ctx);
	if (rc != 0) {
		ibv_destroy_cq(cq_ctx->cq_hdlr);
		ibv_destroy_comp_channel(cq_ctx->comp_channel);
		break;
	}
#endif

	/// Error Handling.
	return 0;
}


static void destroy_cq(struct thread_context_t *t_ctx)
{
	ibv_destroy_cq(t_ctx->send_cq);
	ibv_destroy_cq(t_ctx->recv_cq);
	ibv_destroy_comp_channel(t_ctx->send_comp_channel);
	ibv_destroy_comp_channel(t_ctx->recv_comp_channel);
	return;
}


int create_qp(struct thread_context_t *t_ctx)
{
	struct rdma_resource_t* rdma_resource = t_ctx->rdma_resource;
	struct user_param_t *user_param = &(rdma_resource->user_param);
	struct ibv_qp_init_attr qp_init_attr;

	if (create_cq(t_ctx)) {
		ERROR("Failed to create CQ.\n");
		return 1;
	}

	memset(&qp_init_attr, 0, sizeof(qp_init_attr));
	qp_init_attr.cap.max_send_wr  = user_param->num_of_oust;
	qp_init_attr.cap.max_recv_wr  = user_param->num_of_oust;
	qp_init_attr.cap.max_send_sge = user_param->max_send_sge;
	qp_init_attr.cap.max_recv_sge = user_param->max_recv_sge;
	qp_init_attr.send_cq          = t_ctx->send_cq;
	qp_init_attr.recv_cq          = t_ctx->recv_cq;
	qp_init_attr.qp_type          = user_param->qp_type; /// the qp_type of daemon.
	qp_init_attr.sq_sig_all       = user_param->sq_sig_all; /// sq_sig_all of daemon.
	qp_init_attr.qp_context       = 0; /// How to set.
#if 0
	qp_init_attr.srq          = t_ctx->srq;
#endif

	dump_qp_init_attr(&qp_init_attr);
	t_ctx->qp = ibv_create_qp(rdma_resource->pd, &qp_init_attr);
	if (t_ctx->qp == NULL) {
		ERROR("ibv_create_qp failed.\n");
		return 1;
	}

#if 0
	t_ctx->sq_max_inline = qp_init_attr.cap.max_inline_data;
#endif

	return 0;
}


void destroy_qp(struct thread_context_t *t_ctx)
{
	ibv_destroy_qp(t_ctx->qp);
	destroy_cq(t_ctx);
	return;
}


int create_rdma_buf_pool(struct thread_context_t *t_ctx)
{
	int64_t i;
	uint32_t mr_access;
	struct rdma_buf_t *rdma_buf = NULL;
	struct rdma_resource_t* rdma_resource;

	rdma_resource = t_ctx->rdma_resource;
	
	// register memory for locally accessing; alloc additional 8 bytes for atomic operations.
	t_ctx->buff = (void*)malloc(DEF_MR_SIZE + 8);
	if (t_ctx->buff == NULL) {
		ERROR("Failed to allocate memory for local access.\n");
		return 1;
	}

	// make sure buffer is 8-bytes aligned for atomic operation
	t_ctx->buff_aligned = (void*)((uint64_t)(t_ctx->buff) & (~0x7ll));
	mr_access =
		IBV_ACCESS_LOCAL_WRITE |
		IBV_ACCESS_REMOTE_READ |
		IBV_ACCESS_REMOTE_WRITE |
		IBV_ACCESS_REMOTE_ATOMIC;

	t_ctx->local_mr = ibv_reg_mr(rdma_resource->pd, t_ctx->buff_aligned, DEF_MR_SIZE, mr_access);
	if (t_ctx->local_mr == NULL) {
		ERROR("Failed to register memory for local access.\n");
		free(t_ctx->buff);
		return 1;
	}

	t_ctx->head = t_ctx->buff_aligned;
	pthread_mutex_init(&(t_ctx->mr_mutex), 0);
	pthread_mutex_init(&(t_ctx->pending_mr_mutex), 0);

	for (i = 0; i < (DEF_MR_SIZE / DEF_BUF_SIZE); i++) {
		rdma_buf = t_ctx->buff_aligned + i * DEF_BUF_SIZE;
		rdma_buf->buf_idx = i;
		rdma_buf->next    = (void*)rdma_buf + DEF_BUF_SIZE;
		rdma_buf->status  = 0x1fffffff;
		rdma_buf->cur     = rdma_buf;
	}

	rdma_buf->next = NULL; // the last buffer;
	t_ctx->tail         = rdma_buf;
	t_ctx->pending_head = NULL;
	t_ctx->pending_tail = NULL;
	
	DEBUG("Totally %d buffer created for RDMA.\n", (DEF_MR_SIZE / DEF_BUF_SIZE));
	return 0;
}


struct rdma_buf_t* get_rdma_buf(struct thread_context_t *t_ctx)
{
	/// int64_t buf_idx;
	struct rdma_buf_t* rdma_buf = NULL;

	pthread_mutex_lock(&t_ctx->mr_mutex);
	if (t_ctx->head == NULL) {
		pthread_mutex_lock(&t_ctx->pending_mr_mutex);
		t_ctx->head = t_ctx->pending_head;
		t_ctx->tail = t_ctx->pending_tail;

		t_ctx->pending_head = NULL;
		t_ctx->pending_tail = NULL;
		pthread_mutex_unlock(&t_ctx->pending_mr_mutex);
	}

	if (t_ctx->head != NULL) {
		rdma_buf = t_ctx->head;
		rdma_buf->status = 0x2fffffff;
		t_ctx->head = rdma_buf->next;
		if (t_ctx->head == NULL) {
			t_ctx->tail = NULL;
		}
	}
	pthread_mutex_unlock(&t_ctx->mr_mutex);

	return rdma_buf;
}


void* get_rdma_payload_buf(struct rdma_buf_t *rdma_buf)
{
	assert(sizeof(struct rdma_buf_t) <= RDMA_BUF_HDR_SIZE);
	return ((void*)rdma_buf + RDMA_BUF_HDR_SIZE);
}


void put_rdma_buf(struct thread_context_t *t_ctx, struct rdma_buf_t *rdma_buf)
{
	pthread_mutex_lock(&t_ctx->pending_mr_mutex);
	if (t_ctx->pending_head == NULL) {
		t_ctx->pending_head = rdma_buf;
		t_ctx->pending_tail = rdma_buf;
	} else {
		rdma_buf->next = NULL;
		t_ctx->pending_tail->next = rdma_buf;
		t_ctx->pending_tail = rdma_buf;
	}
	rdma_buf->cur     = rdma_buf;
	rdma_buf->buf_idx = ((void*)rdma_buf - t_ctx->buff_aligned) / DEF_BUF_SIZE;
	rdma_buf->status  = 0x1fffffff;
	pthread_mutex_unlock(&t_ctx->pending_mr_mutex);
}


int rdma_resource_init(struct sock_t *sock, struct rdma_resource_t *rdma_resource)
{
	int rc;

	rc = ib_device_init(rdma_resource);
	if (rc) {
		ERROR("failed to initialize IB device.\n");
		return 1;
	}

	rdma_resource->freq_mhz = get_cpu_mhz(0);
	return 0;
}


int rdma_resource_destroy(struct rdma_resource_t *rdma_resource)
{
	int test_result = 0;

	/// disconnect_qps(rdma_resource);
	/// destroy_qps(rdma_resource);
	/// destroy_srqs(rdma_resource);
	/// destroy_cqs(rdma_resource);
	DEBUG("Destroying RDMA resource.\n");
	ib_device_destroy(rdma_resource);
	return test_result;
}
