/*
 * Copyright (c) 2005 Mellanox Technologies. All rights reserved.
 *
 */


#include "rdma_mt.h"
volatile int stop_all_threads = 0;

static void UNUSED dump_sys_env(struct rdma_resource_t *rdma_resource)
{
	int i;
	struct thread_context_t *t_ctx;
	struct user_param_t *user_param = &(rdma_resource->user_param);

	DEBUG("\nSystem rdma_resource information:\n");
	DEBUG("    |Thread idx |QP idx|QP num|QP type|CQ ridx|CQ sidx|SRQ idx|\n");
	for (i = 0; i < user_param->num_of_thread; i++) {
		t_ctx  = &(rdma_resource->client_ctx[i]);
#if 0
		INFO("    |%-10d |%-6d|%-6d|%-7d|%-7d|%-7d|%-7d|\n",
			i, t_ctx->qp_idx, qp_ctx->qp_handle->qp_num, qp_ctx->qp_type, qp_ctx->recv_cq_idx,
			qp_ctx->send_cq_idx, qp_ctx->srq_idx);
#endif
	}
}


static void UNUSED dump_wc(struct ibv_wc *wc)
{
	if (!Debug) {
		return;
	}

	DEBUG("\nibv_wc:\n");
	DEBUG("\twc->wr_id                   :%lx\n", wc->wr_id);
	DEBUG("\twc->status                  :%s\n", ibv_wc_status_str(wc->status));
	DEBUG("\twc->opcode                  :%x\n", wc->opcode);
	DEBUG("\twc->vendor_err              :%x\n", wc->vendor_err);
	DEBUG("\twc->byte_len                :%x\n", wc->byte_len);
	DEBUG("\twc->imm_data                :%x\n", wc->imm_data);
	DEBUG("\twc->qp_num                  :%x\n", wc->qp_num);
	DEBUG("\twc->src_qp                  :%x\n", wc->src_qp);
	DEBUG("\twc->wc_flags                :%x\n", wc->wc_flags);
	DEBUG("\twc->pkey_index              :%x\n", wc->pkey_index);
	DEBUG("\twc->slid                    :%x\n", wc->slid);
	DEBUG("\twc->sl                      :%x\n", wc->sl);
	DEBUG("\twc->dlid_path_bits          :%x\n", wc->dlid_path_bits);

	return;
}


static void UNUSED dump_rdma_buf(struct rdma_buf_t *rdma_buf)
{
	return;
}


void *async_event_thread(void *ptr)
{
	struct ibv_async_event event;
	struct async_event_thread_context_t *aet_ctx = (struct async_event_thread_context_t *)ptr;
	struct ibv_context *ibv_ctx = aet_ctx->ibv_ctx;
	int rc;

	DEBUG("async event thread started");

	while (1) {
		rc = ibv_get_async_event(ibv_ctx, &event);
		if (rc) {
			ERROR("ibv_get_async_event failed");
			exit(1);
		}

		ibv_ack_async_event(&event);
		switch (event.event_type) {
		case IBV_EVENT_QP_FATAL:
		case IBV_EVENT_QP_REQ_ERR:
		case IBV_EVENT_QP_ACCESS_ERR:
		case IBV_EVENT_COMM_EST:
		case IBV_EVENT_SQ_DRAINED:
		case IBV_EVENT_PATH_MIG:
		case IBV_EVENT_PATH_MIG_ERR:
		case IBV_EVENT_QP_LAST_WQE_REACHED:
			DEBUG("Got QP async event. Type: %s(0x%x), QP handle: %p",
				ibv_event_type_str(event.event_type),
				event.event_type, event.element.qp);
			break;

		case IBV_EVENT_CQ_ERR:
			DEBUG("Got CQ async event. Type: %s(0x%x), CQ handle: %p",
				ibv_event_type_str(event.event_type),
				event.event_type, event.element.cq);
			break;

		case IBV_EVENT_SRQ_ERR:	
		case IBV_EVENT_SRQ_LIMIT_REACHED:
			DEBUG("Got SRQ async event. Type: %s(0x%x), SRQ handle: %p",
				ibv_event_type_str(event.event_type),
				event.event_type, event.element.srq);
			break;

		case IBV_EVENT_DEVICE_FATAL:
			DEBUG("Got CA async event. Type: %s(0x%x)",
				ibv_event_type_str(event.event_type), event.event_type);
			break;

		case IBV_EVENT_PORT_ACTIVE:
		case IBV_EVENT_PORT_ERR:
		case IBV_EVENT_LID_CHANGE:
		case IBV_EVENT_PKEY_CHANGE:
		case IBV_EVENT_SM_CHANGE:
			DEBUG("Got Port async event. Type: %s(0x%x), Port number: %d",
				ibv_event_type_str(event.event_type), event.event_type, event.element.port_num);
			break;

		default:
			DEBUG("Got unexpected async event. Type: %s(0x%x)",
				ibv_event_type_str(event.event_type), event.event_type);
		}

		if (event.event_type < MAX_ASYNC_EVENT_VALUE) {
			continue;
#if 0
			if (async_thread_data_p->expected_event_arr[event.event_type]) {
				continue;
			}
#endif
		} else {
			ERROR("The async event value %u is not supported by test", event.event_type);
		}

		return NULL;
	}
}


static int get_thread_wc(struct thread_context_t *t_ctx, struct ibv_wc *wc, int is_send)
{
	struct ibv_cq           *cq;
	struct ibv_comp_channel *comp_channel;
	struct rdma_resource_t *rdma_resource;
	struct user_param_t *user_param;
	void *ectx;
	int rc = 0;

	rdma_resource = t_ctx->rdma_resource;
	user_param    = &(rdma_resource->user_param);

	if (is_send) {
		cq = t_ctx->send_cq;
		comp_channel = t_ctx->send_comp_channel;
	} else {
		cq = t_ctx->recv_cq;
		comp_channel = t_ctx->recv_comp_channel;
	}

	if (user_param->use_event) {
		rc = ibv_get_cq_event(comp_channel, &cq, &ectx);
		if (rc != 0) {
			ERROR("Failed to do ibv_get_cq_event.\n");
			return 1;
		}

		ibv_ack_cq_events(cq, 1);

		rc = ibv_req_notify_cq(cq, 0);
		if (rc != 0) {
			ERROR("Failed to do ibv_get_cq_event");
			return 1;
		}
	}

	do {
		rc = ibv_poll_cq(cq, 1, wc);
		if (rc < 0) {
			ERROR("Failed to poll CQ.\n");
			return 1;
		}
	} while (!user_param->use_event && (rc == 0)); /// need timeout

	return 0;
}


static int rdma_send(struct thread_context_t *t_ctx, struct rdma_req_t *rdma_req) 
{
	int rc = 0;
	struct rdma_resource_t *rdma_resource;
	struct user_param_t *user_param;
	struct rdma_buf_t *rdma_buf;
	struct ibv_wc wc;

	rdma_resource = t_ctx->rdma_resource;
	user_param    = &(rdma_resource->user_param);
	rdma_buf      = rdma_req->rdma_buf;

	rdma_buf->slid = rdma_resource->port_attr.lid;
	rdma_buf->dlid = t_ctx->remote_lid;
	rdma_buf->sqpn = t_ctx->qp->qp_num;
	rdma_buf->dqpn = t_ctx->remote_qpn;

	rc = post_send(t_ctx, rdma_req);
	if (rc) {
		ERROR("Failed to post_send.\n");
		return rc;
	}

	rc = get_thread_wc(t_ctx, &wc, 1);
	if (rc) {
		ERROR("Failed to get wc.\n");
		return rc;
	}

	if (wc.status != IBV_WC_SUCCESS) {
		ERROR("Got bad completion with status: 0x%x, vendor syndrome: 0x%x\n",
			wc.status, wc.vendor_err);
		return 1;
	}

	return 0;
}


static int verify_rdma_buf(struct thread_context_t *t_ctx, struct rdma_buf_t *rdma_buf)
{
	struct rdma_resource_t *rdma_resource;

	rdma_resource = t_ctx->rdma_resource;
	if ((rdma_buf->dlid != rdma_resource->port_attr.lid) || (rdma_buf->dqpn != t_ctx->qp->qp_num)) {
		ERROR("Failed to verify received rdma_buf, received dlid=%d,dqpn=%d, expected dlid=%d,dqpn=%d.\n",
			rdma_buf->dlid, rdma_buf->dqpn, rdma_resource->port_attr.lid, t_ctx->qp->qp_num);
		return 1;
	}

	return 0;
}


static int rdma_receive(struct thread_context_t *t_ctx, struct rdma_buf_t **rdma_buf) 
{
	int rc = 0;
	struct ibv_wc wc;

	rc = get_thread_wc(t_ctx, &wc, 0);
	if (rc) {
		ERROR("Failed to get wc.\n");
		return rc;
	}

	if (wc.status != IBV_WC_SUCCESS) {
		ERROR("Got bad completion with status: 0x%x, vendor syndrome: 0x%x.\n", wc.status, wc.vendor_err);
		return 1;
	}

	*rdma_buf = (struct rdma_buf_t*)wc.wr_id;
	rc = verify_rdma_buf(t_ctx, *rdma_buf);
	if (rc) {
		ERROR("Failed to verify rdma_buf header.\n");
		return rc;
	}

	return 0;
}


/// need to improve the structure of the function. When to alloc and when to free?
static int do_rdma_transaction(struct thread_context_t *t_ctx, int iter) 
{
	int rc;
	struct rdma_resource_t *rdma_resource;
	struct user_param_t *user_param;
	struct rdma_req_t rdma_req;
	struct rdma_buf_t *rdma_recv_buf;
	int *rdma_send_payload;
	int *rdma_recv_payload;

	rdma_resource     = t_ctx->rdma_resource;
	user_param        = &(rdma_resource->user_param);
	rdma_req.rdma_buf = get_rdma_buf(t_ctx);
	if (rdma_req.rdma_buf == NULL) {
		rc = 1;
		ERROR("Failed to get RDMA buffer.\n");
		return rc;
	}
	
	rdma_req.num_of_oust = 1;
	if (t_ctx->is_requestor) {
		rdma_send_payload  = get_rdma_payload_buf(rdma_req.rdma_buf);
		*rdma_send_payload = iter;
		rdma_req.data_size = RDMA_BUF_HDR_SIZE + sizeof(int);
		rdma_req.opcode    = user_param->opcode;

		rc = rdma_send(t_ctx, &rdma_req);
		if (rc) {
			ERROR("Failed to do_rdma_send.\n");
			return rc;
		}

		t_ctx->t_b[iter] = get_cycles();
		if (user_param->direction == 0) {
			rc = rdma_receive(t_ctx, &rdma_recv_buf);
			if (rc) {
				ERROR("Failed to do_rdma_recv.\n");
				return rc;
			}

			rdma_recv_payload = get_rdma_payload_buf(rdma_recv_buf);
			DEBUG("Get response: %d.\n", *rdma_recv_payload);
			if (*rdma_recv_payload != iter) {
				ERROR("Receive wrong iter=%d, expected iter=%d.\n", *rdma_recv_payload, iter);
				return 1;
			}

			put_rdma_buf(t_ctx, rdma_recv_buf);
		}
	} else {
		rc = rdma_receive(t_ctx, &rdma_recv_buf);
		if (rc) {
			ERROR("Failed to do_rdma_recv.\n");
			return rc;
		}

		t_ctx->t_b[iter] = get_cycles();

		if (user_param->direction == 0) {
			rdma_recv_payload  = get_rdma_payload_buf(rdma_recv_buf);
			rdma_send_payload  = get_rdma_payload_buf(rdma_req.rdma_buf);
			
			*rdma_send_payload = *rdma_recv_payload;
			DEBUG("Receive data: %d.\n", *rdma_recv_payload);
			rdma_req.data_size = RDMA_BUF_HDR_SIZE + sizeof(int);
			rdma_req.opcode    = user_param->opcode;
			rc = rdma_send(t_ctx, &rdma_req);
			if (rc) {
				ERROR("Failed to do_rdma_send.\n");
				return rc;
			}
		}

		put_rdma_buf(t_ctx, rdma_recv_buf);
	}

	DEBUG("do_rdma_transaction ... end\n");

	put_rdma_buf(t_ctx, rdma_req.rdma_buf);
	return 0;
}


static int UNUSED check_opcode_supported(struct thread_context_t *t_ctx, enum ibv_wr_opcode opcode)
{
	int ret = 1;

	DEBUG("\nDoing check_opcode_supported, t_ctx=%p, opcode=%x, t_ctx->qp_type=%x.\n",
		t_ctx, opcode, t_ctx->qp_type);

	switch (opcode) {
	case IBV_WR_ATOMIC_FETCH_AND_ADD:
	case IBV_WR_ATOMIC_CMP_AND_SWP:
	case IBV_WR_RDMA_READ:
		if (t_ctx->qp_type != IBV_QPT_RC) {
			ret = 0;
		}
		break;
	case IBV_WR_RDMA_WRITE:
	case IBV_WR_RDMA_WRITE_WITH_IMM:
		if ((t_ctx->qp_type != IBV_QPT_RC) && (t_ctx->qp_type != IBV_QPT_UC)) {
			ret = 0;
		}
		break;

	case IBV_WR_SEND:
	case IBV_WR_SEND_WITH_IMM:
		// supported by all ts types
		break;
	default:
		ERROR("Unsupported operation, opcode: %d", opcode);
		ret = 0;
	}

	return ret;
}


static void* rdma_thread(void *ptr) 
{
	int i, j, rc;
	struct rdma_resource_t  *rdma_resource;
	struct user_param_t     *user_param;
	struct thread_context_t *t_ctx;
	struct rdma_req_t       rdma_req;
	double                  lat;
	
	t_ctx              = (struct thread_context_t*)ptr;
	rdma_resource      = t_ctx->rdma_resource;
	user_param         = &(rdma_resource->user_param);
	t_ctx->thread_id   = pthread_self();
	t_ctx->num_of_iter = user_param->num_of_iter;

	if (create_rdma_buf_pool(t_ctx)) {
		ERROR("Failed to create MR pool.\n");
		return NULL;
	}

	{
		uint32_t qp_type;

		if (user_param->server_ip != NULL) {
			qp_type = htonl(user_param->qp_type);
		}

		sock_c2d(&(t_ctx->sock), sizeof(qp_type), &qp_type);

		if (user_param->server_ip == NULL) {
			user_param->qp_type = ntohl(qp_type);
		}

		t_ctx->qp_type = user_param->qp_type; /// redesign
	}

	if (create_qp(t_ctx)) {
		ERROR("Failed to create QP.\n");
		return NULL;
	}

	{
		struct thread_sync_info_t {
			uint32_t qp_num;
			uint32_t direction;
			uint32_t opcode;
			uint32_t qkey;
			uint32_t psn;
			uint32_t num_of_iter;
			uint16_t lid;
		} ATTR_PACKED;
		struct thread_sync_info_t local_info;
		struct thread_sync_info_t remote_info;

		local_info.lid         = htons(rdma_resource->port_attr.lid);
		local_info.qp_num      = htonl(t_ctx->qp->qp_num);
		local_info.direction   = htonl(user_param->direction);
		local_info.opcode      = htonl(user_param->opcode); /// enum ibv_wr_opcode
		local_info.qkey        = htonl(0);
		local_info.psn         = htonl(0);
		local_info.num_of_iter = htonl(t_ctx->num_of_iter);

		rc = sock_sync_data(&(t_ctx->sock), sizeof(local_info), &local_info, &remote_info);
		if (rc) {
			ERROR("failed to sync data.\n");
			return NULL;
		}

		t_ctx->remote_lid      = ntohs(remote_info.lid);
		t_ctx->remote_qpn      = ntohl(remote_info.qp_num);
		t_ctx->remote_qkey     = ntohl(remote_info.qkey);
		t_ctx->remote_psn      = ntohl(remote_info.psn);
		if (user_param->server_ip == NULL) {
			user_param->direction = ntohl(remote_info.direction);
			user_param->opcode    = ntohl(remote_info.opcode);
			t_ctx->num_of_iter    = ntohl(remote_info.num_of_iter);

			if (user_param->direction == 0 || user_param->direction == 1) {
				t_ctx->is_requestor = 0;
			} else if (user_param->direction == 2) {
				t_ctx->is_requestor = 1;
			}
		} else {
			if (user_param->direction == 0 || user_param->direction == 1) {
				t_ctx->is_requestor = 1;
			} else if (user_param->direction == 2) {
				t_ctx->is_requestor = 0;
			}
		}

	}

	t_ctx->t_a = (cycles_t*)malloc(t_ctx->num_of_iter * sizeof(cycles_t));
	if (t_ctx->t_a == NULL) {
		ERROR("Failed to allocate memory.\n");
		return NULL;
	}

	t_ctx->t_b = (cycles_t*)malloc(t_ctx->num_of_iter * sizeof(cycles_t));
	if (t_ctx->t_b == NULL) {
		free(t_ctx->t_a);
		ERROR("Failed to allocate memory.\n");
		return NULL;
	}

	t_ctx->t_c = (cycles_t*)malloc(t_ctx->num_of_iter * sizeof(cycles_t));
	if (t_ctx->t_c == NULL) {
		free(t_ctx->t_b);
		free(t_ctx->t_a);
		ERROR("Failed to allocate memory.\n");
		return NULL;
	}

	for (i = 0; i < LAT_LEVEL; i++) {
		t_ctx->lat[i] = 0;
	}

	if (connect_qp(t_ctx)) {
		ERROR("Failed to connect QP.\n");
		return NULL;
	}

	for(i = 0; i < user_param->num_of_oust; i++) {
		rdma_req.rdma_buf    = get_rdma_buf(t_ctx);
		rdma_req.num_of_oust = 1;
		rdma_req.data_size   = DEF_BUF_SIZE;

		rc = post_receive(t_ctx, &rdma_req);
		if (rc) {
			ERROR("Failed to post_receive, i:%d.\n", i);
			return NULL;
		}
	}

	sock_sync_ready(&t_ctx->sock);
	for (i = 0; i < t_ctx->num_of_iter; i++) {
		t_ctx->t_a[i] = get_cycles();
		DEBUG("do_rdma_transaction, t_ctx->num_of_iter=%d, i=%d.\n", t_ctx->num_of_iter, i);
		rc = do_rdma_transaction(t_ctx, i);
		if (rc) {
			ERROR("Failed to do_rdma_transaction, i:%d.\n", i);
			return NULL;
		}
		
		t_ctx->t_c[i] = get_cycles();

		if (user_param->direction == 0 || (!t_ctx->is_requestor)) {
			rdma_req.rdma_buf = get_rdma_buf(t_ctx);

			if (rdma_req.rdma_buf == NULL) {
				ERROR("Failed to get RDMA buffer.\n");
				return NULL; /// Memory Leak and remove hung RX buffers
			}
			rdma_req.num_of_oust = 1;
			post_receive(t_ctx, &rdma_req);
		}

		if (user_param->interval) {
			usleep(user_param->interval);
		}
	}
	
	/// Memory leak, release the hung RX rdma_buf;
	destroy_qp(t_ctx);

	t_ctx->min_lat = 0x7fffffff;
	t_ctx->max_lat = 0;
	for (i = 0; i < t_ctx->num_of_iter; i++) {
		lat = (t_ctx->t_c[i] - t_ctx->t_a[i]) / rdma_resource->freq_mhz;

		if (lat < t_ctx->min_lat) {
			t_ctx->min_lat          = lat;
			t_ctx->min_lat_iter_num = i;
		}

		if (lat > t_ctx->max_lat) {
			t_ctx->max_lat          = lat;
			t_ctx->max_lat_iter_num = i;
		}

		for (j = 0; j < LAT_LEVEL; j++) {
			if (j < 7) {
				if (lat < (1 + j)) {
					t_ctx->lat[j]++;
					break;
				}
			} else {
				if (lat < (1 << (j - 4))) {
					t_ctx->lat[j]++;
					break;
				}
			}
		}

		if (j == LAT_LEVEL) {
			t_ctx->lat[LAT_LEVEL - 1]++;
		}
	}

	free(t_ctx->t_a);
	free(t_ctx->t_b);
	free(t_ctx->t_c);
	if (!user_param->server_ip) {
		/// sock_close_multi(&(t_ctx->sock), sock_bind); // how to close sock_fd.
		free(t_ctx); /// Need to improve.
	}

	INFO("RDMA testing thread successfully exited.\n");
	return NULL;
}


int start_rdma_threads(struct sock_t *sock, struct rdma_resource_t *rdma_resource,
	struct sock_bind_t *sock_bind)
{
	int i, j;
	int rc;
	int ret = 0;
	unsigned long exit_code;
	struct thread_context_t *t_ctx;
	struct user_param_t *user_param = &(rdma_resource->user_param);
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	if (user_param->server_ip) {
		rdma_resource->client_ctx = 
			(struct thread_context_t*)malloc(sizeof(struct thread_context_t) * user_param->num_of_thread);
	} else {
		pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
	}

	i = 0;

	while (1) {
		if (user_param->server_ip) {
			t_ctx = &rdma_resource->client_ctx[i];
		} else {
			t_ctx = (struct thread_context_t*)malloc(sizeof(struct thread_context_t));
		}

		t_ctx->rdma_resource = rdma_resource;
		rc = sock_connect_multi(sock_bind, &(t_ctx->sock));
		if (rc) {
			ERROR("Failed to open connection between the 2 sides.\n");
			goto failure_1;
		}

		rc = pthread_create(&(t_ctx->thread), &attr, rdma_thread, t_ctx);
		if (rc != 0) {
			ERROR("Failed to create thread.\n");
			break;
		}

		i++;
		if (i >= user_param->num_of_thread && (user_param->server_ip)) {
			break;
		}
	}

	if (user_param->server_ip) {
		if (i != user_param->num_of_thread) {
			for (j = 0; j < i; j++) {
				t_ctx = &rdma_resource->client_ctx[i];
				pthread_cancel(t_ctx->thread);
				pthread_join(t_ctx->thread, (void *)&exit_code);
			}

			ret = 1;
		}

		for (i = 0; i < user_param->num_of_thread; i++) {
			t_ctx = &rdma_resource->client_ctx[i];

			rc = pthread_join(t_ctx->thread, (void *)&exit_code);
			if ((rc != 0) || (exit_code != 0)) {
				ERROR("Failed to wait for thread[%d] termination.\n", i);
				ret = 1;
			} else {
				INFO("Thread[%d] finished with return value %lu\n", i, exit_code);

				if (t_ctx->min_lat < rdma_resource->min_lat) {
					rdma_resource->min_lat          = t_ctx->min_lat;
					rdma_resource->min_lat_iter_num = t_ctx->min_lat_iter_num;
				}

				if (t_ctx->max_lat > rdma_resource->max_lat) {
					rdma_resource->max_lat          = t_ctx->max_lat;
					rdma_resource->max_lat_iter_num = t_ctx->max_lat_iter_num;
				}

				for (j = 0; j < LAT_LEVEL; j++) {
					rdma_resource->lat[j] += t_ctx->lat[j];
				}
			}

			sock_close_multi(&(t_ctx->sock), sock_bind);
		}
		free(rdma_resource->client_ctx);

		INFO("Got min lat %f when sending NO.%u packet.\n", rdma_resource->min_lat, rdma_resource->min_lat_iter_num);
		INFO("Got max lat %f when sending NO.%u packet.\n", rdma_resource->max_lat, rdma_resource->max_lat_iter_num);
		for (i = 0; i < LAT_LEVEL; i++) {
			if (i < 7) {
				INFO("The number of Lat < %4dus is %3d \n", (1 + i), rdma_resource->lat[i]);
			} else {
				INFO("The number of Lat < %4dus is %3d \n", (1 << (i - 4)), rdma_resource->lat[i]);
			}
		}
	}

failure_1:
	return ret;
}

