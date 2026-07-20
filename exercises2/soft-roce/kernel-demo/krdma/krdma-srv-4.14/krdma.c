/*
 * RDMA support for KVM software distributed memory
 *
 * This feature allows us to run multiple KVM instances on different machines
 * sharing the same address space.
 *
 * Copyright (C) 2019, Trusted Cloud Group, Shanghai Jiao Tong University.
 *
 * Authors:
 *   Yubin Chen <binsschen@sjtu.edu.cn>
 *   Zhuocheng Ding <tcbbd@sjtu.edu.cn>
 *   Jin Zhang <jzhang3002@sjtu.edu.cn>
 *   Xingguo Jia <jiaxg1998@sjtu.edu.cn>
 *
 * This work is licensed under the terms of the GNU GPL, version 2.  See
 * the COPYING file in the top-level directory.
 */

#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/inet.h>
#include <linux/proc_fs.h>
#include <linux/kthread.h>
#include <linux/sched/signal.h>
// #include <linux/kvm_host.h>

#include "krdma.h"
#define CLINET_IP "10.11.11.82"
#define SERVER_IP "10.11.11.251"
static bool dbg = 1;
#define krdma_debug(x, ...) do { 				\
	if (unlikely(dbg)) printk(KERN_WARNING "%s(): %d "	\
		x, __func__, __LINE__, ##__VA_ARGS__);	\
} while (0)

#define krdma_err(x, ...) do { 					\
	printk(KERN_ERR "%s(): %d "		\
		x, __func__, __LINE__, ##__VA_ARGS__);	\
} while (0)

////////////////////////////////////////////////////////////////////
///////////////Connection Management Functions//////////////////////
////////////////////////////////////////////////////////////////////

/* Allocate cb, freed by caller */
static int __krdma_create_cb(struct krdma_cb **cbp, enum krdma_role role);
static int __krdma_free_cb(struct krdma_cb *cb);

/* Allocate cb->cm_id, freed by caller */
// remote host & port
static int __krdma_bound_dev_remote(struct krdma_cb *cb, const char *host, const char *port);
// local host & port
static int __krdma_bound_dev_local(struct krdma_cb *cb, const char *host, const char *port);

/* Allocate cb->pd, cb->cq, cb->qp, mr, freed by caller */
static int krdma_init_cb(struct krdma_cb *cb);
/* Deallocate cb->cm_id, cb->pd, cb->cq, cb->qp, mr */
int krdma_release_cb(struct krdma_cb *cb);

/* Call rdma_cm, allocate nothing */
static int __krdma_connect(struct krdma_cb *cb);
static int __krdma_listen(struct krdma_cb *cb);
static int __krdma_accept(struct krdma_cb *cb);

/* wait for rdma_cm RDMA_CM_EVENT_CONNECT_REQUEST */
static struct krdma_cb *__krdma_wait_for_connect_request(struct krdma_cb *listen_cb);

static int krdma_set_addr(struct sockaddr_in *addr, const char *host, const char *port) {
	int ret;
	long portdec;

	if (!addr)
		return 0;

	memset(addr, 0, sizeof(struct sockaddr_in));
	addr->sin_family = AF_INET;
	ret = kstrtol(port, 10, &portdec);
	if (ret < 0) {
		krdma_err("kstrtol error %d", ret);
		return ret;
	}
	addr->sin_addr.s_addr = in_aton(host);
	addr->sin_port = htons(portdec);
	return 0;
}

static int krdma_cma_event_handler(struct rdma_cm_id *cm_id,
		struct rdma_cm_event *event)
{
	int ret;
	struct krdma_cb *cb = cm_id->context;
	struct krdma_cb *conn_cb = NULL;

	switch (event->event) {
	case RDMA_CM_EVENT_ADDR_RESOLVED:
		krdma_debug("%s: RDMA_CM_EVENT_ADDR_RESOLVED, cm_id %p\n",
				__func__, cm_id);
		cb->state = KRDMA_ADDR_RESOLVED;
		break;

	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		krdma_debug("%s: RDMA_CM_EVENT_ROUTE_RESOLVED, cm_id %p\n",
				__func__, cm_id);
		cb->state = KRDMA_ROUTE_RESOLVED;
		break;

	case RDMA_CM_EVENT_ROUTE_ERROR:
		krdma_debug("%s: RDMA_CM_EVENT_ROUTE_ERROR, cm_id %p, error %d\n",
				__func__, cm_id, event->status);
		break;

	case RDMA_CM_EVENT_CONNECT_REQUEST:
		krdma_debug("%s: RDMA_CM_EVENT_CONNECT_REQUEST, cm_id %p\n",
				__func__, cm_id);
		/* create a new cb */
		ret = __krdma_create_cb(&conn_cb, KRDMA_ACCEPT_CONN);
		if (!ret) {
			conn_cb->cm_id = cm_id;
			cm_id->context = conn_cb;
			list_add_tail(&conn_cb->list, &cb->ready_conn);
		} else {
			krdma_err("__krdma_create_cb fail, ret %d\n", ret);
			cb->state = KRDMA_ERROR;
		}
		break;

	case RDMA_CM_EVENT_ESTABLISHED:
		krdma_debug("%s: RDMA_CM_EVENT_ESTABLISHED, cm_id %p\n",
				__func__, cm_id);
		cb->state = KRDMA_CONNECTED;
		break;

	case RDMA_CM_EVENT_DISCONNECTED:
		krdma_debug(KERN_ERR "%s: RDMA_CM_EVENT_DISCONNECTED, cm_id %p\n",
				__func__, cm_id);
		cb->state = KRDMA_DISCONNECTED;
		break;

	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
	case RDMA_CM_EVENT_REJECTED:
		krdma_err("RDMA_CM_EVENT %d, cm_id %p\n", event->event, cm_id);
		cb->state = KRDMA_CONNECT_REJECTED;
		break;
	default:
		krdma_debug("%s: unknown event %d, cm_id %p\n",
				__func__, event->event, cm_id);
	}
	complete(&cb->cm_done);
	return 0;
}

/* MR for RDMA send recv */
static int __krdma_setup_mr_sr(struct krdma_cb *cb) {
	int i;
	krdma_send_trans_t *send_trans_buf;
	krdma_recv_trans_t *recv_trans_buf;

	BUG_ON(cb->read_write);

	send_trans_buf = kzalloc(RDMA_SEND_BUF_SIZE *
			sizeof(krdma_send_trans_t), GFP_KERNEL);
	recv_trans_buf = kzalloc(RDMA_RECV_BUF_SIZE *
			sizeof(krdma_recv_trans_t), GFP_KERNEL);
	if (!(send_trans_buf && recv_trans_buf)) {
		krdma_err("kzalloc send/recv_trans_buf failed\n");
		goto exit;
	}
	cb->mr.sr_mr.send_trans_buf = send_trans_buf;
	cb->mr.sr_mr.recv_trans_buf = recv_trans_buf;

	for (i = 0; i < RDMA_SEND_BUF_SIZE; i++) {
		send_trans_buf[i].send_buf = ib_dma_alloc_coherent(cb->pd->device,
				RDMA_SEND_BUF_LEN, &send_trans_buf[i].send_dma_addr,
				GFP_KERNEL | GFP_DMA);
		if (!send_trans_buf[i].send_buf) {
			krdma_err("ib_dma_alloc_coherent send_buf failed\n");
			goto out_free_bufs;
		}

		send_trans_buf[i].send_sge.lkey = cb->pd->local_dma_lkey;
		/* .length is set at runtime. */
		send_trans_buf[i].send_sge.addr = send_trans_buf[i].send_dma_addr;
		send_trans_buf[i].sq_wr.next = NULL;
		/* .wr_id is set at runtime. */
		send_trans_buf[i].sq_wr.sg_list = &send_trans_buf[i].send_sge;
		send_trans_buf[i].sq_wr.num_sge = 1;
		send_trans_buf[i].sq_wr.opcode = IB_WR_SEND_WITH_IMM;
		send_trans_buf[i].sq_wr.send_flags = IB_SEND_SIGNALED;
		/* .ex.imm_data is set at runtime. */
	}

	for (i = 0; i < RDMA_RECV_BUF_SIZE; i++) {
		recv_trans_buf[i].recv_buf = ib_dma_alloc_coherent(cb->pd->device,
				RDMA_RECV_BUF_LEN, &recv_trans_buf[i].recv_dma_addr,
				GFP_KERNEL | GFP_DMA);
		if (!recv_trans_buf[i].recv_buf) {
			krdma_err("ib_dma_alloc_coherent recv_buf failed\n");
			goto out_free_bufs;
		}

		recv_trans_buf[i].recv_sge.lkey = cb->pd->local_dma_lkey;
		recv_trans_buf[i].recv_sge.length = RDMA_RECV_BUF_LEN;
		recv_trans_buf[i].recv_sge.addr = recv_trans_buf[i].recv_dma_addr;
		recv_trans_buf[i].rq_wr.next = NULL;
		recv_trans_buf[i].rq_wr.wr_id = i;
		recv_trans_buf[i].rq_wr.sg_list = &recv_trans_buf[i].recv_sge;
		recv_trans_buf[i].rq_wr.num_sge = 1;
	}

	return 0;

out_free_bufs:
	for (i = 0; i < RDMA_SEND_BUF_SIZE; i++) {
		if (send_trans_buf[i].send_buf) {
			ib_dma_free_coherent(cb->pd->device, RDMA_SEND_BUF_LEN,
					send_trans_buf[i].send_buf,
					send_trans_buf[i].send_dma_addr);
		}
	}
	for (i = 0; i < RDMA_RECV_BUF_SIZE; i++) {
		if (recv_trans_buf[i].recv_buf) {
			ib_dma_free_coherent(cb->pd->device, RDMA_RECV_BUF_LEN,
					recv_trans_buf[i].recv_buf,
					recv_trans_buf[i].recv_dma_addr);
		}
	}

exit:
	kfree(send_trans_buf);
	kfree(recv_trans_buf);
	return -ENOMEM;
}

static uint64_t __krdma_virt_to_dma(
	struct krdma_cb *cb, void *addr, size_t length, bool physical_allocation) {
	struct ib_device *ibd = cb->pd->device;

	return physical_allocation ?
		(uint64_t) phys_to_dma(ibd->dma_device, (phys_addr_t) virt_to_phys(addr))
	:	(uint64_t) ib_dma_map_single(ibd, addr, length, DMA_BIDIRECTIONAL);
}

/* MR for RDMA read write */
static int __krdma_setup_mr_rw(struct krdma_cb *cb, bool physical_allocation) {
	krdma_rw_info_t *local_info;
	krdma_rw_info_t *remote_info;
	struct ib_port_attr port_attr;
	void *buf;
	int ret;

	BUG_ON(!cb->read_write);

	local_info = kzalloc(sizeof(krdma_rw_info_t), GFP_KERNEL);
	if (!local_info) {
		krdma_err("local_info alloc failed.\n");
		goto exit;
	}
	cb->mr.rw_mr.local_info = local_info;

	remote_info = kzalloc(sizeof(krdma_rw_info_t), GFP_KERNEL);
	if (!remote_info) {
		krdma_err("remote_info alloc failed.\n");
		goto out_free_local_info;
	}
	cb->mr.rw_mr.remote_info = remote_info;

	buf = kmalloc(RDMA_RDWR_BUF_LEN, GFP_KERNEL);
	if (!buf) {
		krdma_err("buf alloc failed.\n");
		goto out_free_remote_info;
	}
	local_info->buf = buf;

	local_info->length = RDMA_RDWR_BUF_LEN;
	local_info->addr = __krdma_virt_to_dma(cb, local_info->buf, local_info->length, 1);
	local_info->rkey = cb->pd->unsafe_global_rkey;
	local_info->qp_num = cb->qp->qp_num;
	
	ret = ib_query_port(cb->cm_id->device, cb->cm_id->port_num, &port_attr);
	if (ret) {
		krdma_err("ib_query_port failed.\n");
		goto out_free_buf;
	}

	local_info->lid = port_attr.lid;
	return 0;

out_free_buf:
	kfree(buf);
	cb->mr.rw_mr.local_info->buf = NULL;

out_free_remote_info:
	kfree(remote_info);
	cb->mr.rw_mr.remote_info = NULL;

out_free_local_info:
	kfree(local_info);
	cb->mr.rw_mr.local_info = NULL;
exit:
	return -ENOMEM;
}

static int krdma_setup_mr(struct krdma_cb *cb) {
	return cb->read_write ? __krdma_setup_mr_rw(cb, true) : __krdma_setup_mr_sr(cb);
}

static int __krdma_free_mr_sr(struct krdma_cb *cb)
{
	int i;
	krdma_send_trans_t *send_trans_buf = cb->mr.sr_mr.send_trans_buf;
	krdma_recv_trans_t *recv_trans_buf = cb->mr.sr_mr.recv_trans_buf;
	BUG_ON(cb->read_write);

	for (i = 0; i < RDMA_SEND_BUF_SIZE; i++) {
		if (send_trans_buf[i].send_buf) {
			ib_dma_free_coherent(cb->pd->device, RDMA_SEND_BUF_LEN,
					send_trans_buf[i].send_buf,
					send_trans_buf[i].send_dma_addr);
			send_trans_buf[i].send_buf = NULL;
		}
	}
	for (i = 0; i < RDMA_RECV_BUF_SIZE; i++) {
		if (recv_trans_buf[i].recv_buf) {
			ib_dma_free_coherent(cb->pd->device, RDMA_RECV_BUF_LEN,
					recv_trans_buf[i].recv_buf,
					recv_trans_buf[i].recv_dma_addr);
		}
		recv_trans_buf[i].recv_buf = NULL;
	}
	kfree(send_trans_buf);
	kfree(recv_trans_buf);

	return 0;
}

static int __krdma_free_mr_rw(struct krdma_cb *cb) {
	int ret = 0;

	BUG_ON(!cb->read_write);

	kfree(cb->mr.rw_mr.local_info->buf);
	kfree(cb->mr.rw_mr.remote_info);
	cb->mr.rw_mr.remote_info = NULL;
	kfree(cb->mr.rw_mr.local_info);
	cb->mr.rw_mr.local_info = NULL;

	return ret;
}

static int krdma_free_mr(struct krdma_cb *cb) {
	return cb->read_write ? __krdma_free_mr_rw(cb) : __krdma_free_mr_sr(cb);
}


static int krdma_post_recv(struct krdma_cb *cb);

static int krdma_connect_single(const char *host, const char *port,
		struct krdma_cb *cb)
{
	int ret;

	if (host == NULL || port == NULL || cb == NULL)
		return -EINVAL;

	ret = __krdma_bound_dev_remote(cb, host, port);
	if (ret)
		return ret;

	ret = krdma_init_cb(cb);
	if (ret < 0)
		goto out_release_cm_id;

	ret = __krdma_connect(cb);
	if (ret < 0)
		goto out_release_cb;
	return 0;

out_release_cm_id:
	rdma_destroy_id(cb->cm_id);
	cb->cm_id = NULL;
	return ret;

out_release_cb:
	krdma_release_cb(cb);
	return ret;
}

int krdma_connect(const char *host, const char *port, struct krdma_cb **conn_cb)
{
	int ret;
	struct krdma_cb *cb;

	if (host == NULL || port == NULL || conn_cb == NULL)
		return -EINVAL;

	ret = __krdma_create_cb(&cb, KRDMA_CLIENT_CONN);
	if (ret) {
		krdma_err("__krdma_create_cb fail, ret %d\n", ret);
		return ret;
	}
	cb->read_write = false;

retry:
	ret = krdma_connect_single(host, port, cb);
	if (ret == 0) {
		/*
		 * If multiple clients desire to connect to remote servers, only one of
		 * them can call this function. Others must be blocked even if conn_cb
		 * has not been set here. Then double checking whether conn_cb is
		 * non-NULL can ensure the correctness of lazy connection.
		 */
		smp_mb();
		*conn_cb = cb;
		krdma_debug("%p krdma_connect succeed\n", cb);
		return 0;
	}
	if (ret == -CLIENT_RETRY &&
				++cb->retry_count < RDMA_CONNECT_RETRY_MAX) {
		krdma_err("krdma_connect_single failed, retry_count %d, " \
				"reconnecting...\n", cb->retry_count);
		msleep(1000);
		goto retry;
	} else {
		__krdma_free_cb(cb);
		*conn_cb = NULL;
		krdma_err("krdma_connect_single failed, ret: %d\n", ret);
		return ret;
	}
}

int krdma_listen(const char *host, const char *port, struct krdma_cb **listen_cb)
{
	int ret;
	struct krdma_cb *cb;

	if (host == NULL || port == NULL || listen_cb == NULL)
		return -EINVAL;

	ret = __krdma_create_cb(listen_cb, KRDMA_LISTEN_CONN);
	if (ret) {
		krdma_err("__krdma_create_cb fail, ret %d\n", ret);
		return ret;
	}
	cb = *listen_cb;
	cb->read_write = false;

	ret = __krdma_bound_dev_local(cb, host, port);
	if (ret < 0)
		goto out_free_cb;
	
	ret = __krdma_listen(cb);
	if (ret < 0)
		goto out_release_cm_id;
	return 0;

out_release_cm_id:
	rdma_destroy_id(cb->cm_id);
	cb->cm_id = NULL;
	
out_free_cb:
	__krdma_free_cb(cb);
	*listen_cb = NULL;
	return ret;
}

static struct krdma_cb *__krdma_wait_for_connect_request(struct krdma_cb *listen_cb) {
	struct krdma_cb *cb;

	while (list_empty(&listen_cb->ready_conn)) {
		wait_for_completion_interruptible(&listen_cb->cm_done);
		if (listen_cb->state == KRDMA_ERROR) {
			krdma_err("rdma_listen cancel\n");
			goto exit;
		}
		if (kthread_should_stop())
			goto exit;
	}

	/* Pick a ready connnection. */
	cb = list_first_entry(&listen_cb->ready_conn, struct krdma_cb, list);
	list_del(&cb->list);
	list_add_tail(&cb->list, &listen_cb->active_conn);
	return cb;

exit:
	return NULL;
}

int krdma_accept(struct krdma_cb *listen_cb, struct krdma_cb **accept_cb)
{
	int ret = 0;
	struct krdma_cb *cb;

	if (listen_cb == NULL) {
		krdma_err("null listen_socket\n");
		ret = -EINVAL;
		goto exit;
	}

	if (accept_cb == NULL) {
		krdma_err("null accept_cb\n");
		ret = -EINVAL;
		goto exit;
	}

	cb = __krdma_wait_for_connect_request(listen_cb);
	if (cb == NULL) {
		krdma_err("__krdma_wait_for_connect_request failed.\n");
		ret = -STATE_ERROR;
		goto exit;
	}

	*accept_cb = cb;
	cb->read_write = false;

	krdma_debug("get connection, cm_id %p\n", cb->cm_id);

	ret = krdma_init_cb(cb);
	if (ret < 0)
		goto out_free_cb;

	ret = __krdma_accept(cb);
	if (ret < 0)
		goto out_release_cb;

	return 0;

out_release_cb:
	krdma_release_cb(cb);

out_free_cb:
	__krdma_free_cb(cb);
	*accept_cb = NULL;

exit:
	return ret;
}

int krdma_release_cb(struct krdma_cb *cb)
{
	struct krdma_cb *entry = NULL;
	struct krdma_cb *this = NULL;

	if (cb == NULL)
		return -EINVAL;

	if (!cb->cm_id)
		return -EINVAL;

	rdma_disconnect(cb->cm_id);
	if (cb->cm_id->qp)
		rdma_destroy_qp(cb->cm_id);

	krdma_free_mr(cb);
	if (cb->send_cq)
		ib_destroy_cq(cb->send_cq);
	if (cb->recv_cq)
		ib_destroy_cq(cb->recv_cq);

	if (cb->pd)
		ib_dealloc_pd(cb->pd);

	rdma_destroy_id(cb->cm_id);
	cb->cm_id = NULL;

	if (cb->role == KRDMA_LISTEN_CONN) {
		list_for_each_entry_safe(entry, this, &cb->ready_conn, list) {
			krdma_release_cb(entry);
			list_del(&entry->list);
		}
		list_for_each_entry_safe(entry, this, &cb->active_conn, list) {
			krdma_release_cb(entry);
			list_del(&entry->list);
		}
	}

	return 0;
}

static int __krdma_create_cb(struct krdma_cb **cbp, enum krdma_role role)
{
	struct krdma_cb *cb;

	cb = kzalloc(sizeof(*cb), GFP_KERNEL);
	if (!cb)
		return -ENOMEM;
	init_completion(&cb->cm_done);

	cb->role = role;
	if (cb->role == KRDMA_LISTEN_CONN) {
		INIT_LIST_HEAD(&cb->ready_conn);
		INIT_LIST_HEAD(&cb->active_conn);
	}
	mutex_init(&cb->slock);
	mutex_init(&cb->rlock);

	if (cbp)
		*cbp = cb;
	return 0;
}

static int __krdma_free_cb(struct krdma_cb *cb) {
	kfree(cb);
	return 0;
}

/*
 * Call rdma_resolve_route for dev detection
 */
static int __krdma_bound_dev_remote(struct krdma_cb *cb, const char *host, const char *port) {
	int ret;
	struct sockaddr_in addr;
	
	/* Create cm_id */
	cb->cm_id = rdma_create_id(&init_net, krdma_cma_event_handler, cb,
					RDMA_PS_TCP, IB_QPT_RC);
	if (IS_ERR(cb->cm_id)) {
		ret = PTR_ERR(cb->cm_id);
		krdma_err("rdma_create_id error %d\n", ret);
		goto exit;
	}
	krdma_debug("created cm_id %p\n", cb->cm_id);

	/* Resolve address */
	ret = krdma_set_addr(&addr, host, port);
	if (ret < 0)
		goto free_cm_id;
	ret = rdma_resolve_addr(cb->cm_id, NULL,
			(struct sockaddr *)&addr, RDMA_RESOLVE_TIMEOUT);
	if (ret) {
		krdma_err("rdma_resolve_addr failed, ret %d\n", ret);
		goto free_cm_id;
	}
	wait_for_completion(&cb->cm_done);
	if (cb->state != KRDMA_ADDR_RESOLVED) {
		ret = -STATE_ERROR;
		krdma_err("rdma_resolve_route state error, ret %d\n", ret);
		goto free_cm_id;
	}
	krdma_debug("rdma_resolve_addr succeed, device[%s] port_num[%u]\n",
			cb->cm_id->device->name, cb->cm_id->port_num);

	/* Resolve route. */
	ret = rdma_resolve_route(cb->cm_id, RDMA_RESOLVE_TIMEOUT);
	if (ret) {
		krdma_err("rdma_resolve_route failed, ret %d\n", ret);
		goto free_cm_id;
	}
	wait_for_completion(&cb->cm_done);
	if (cb->state != KRDMA_ROUTE_RESOLVED) {
		ret = -STATE_ERROR;
		krdma_err("rdma_resolve_route state error, ret %d\n", ret);
		goto free_cm_id;
	}
	krdma_debug("rdma_resolve_route succeed, cm_id %p\n", cb->cm_id);

	return 0;

free_cm_id:
	rdma_destroy_id(cb->cm_id);
	cb->cm_id = NULL;
exit:
	return ret;
}

/*
 * Call rdma_bind for ib dev detection
 */
static int __krdma_bound_dev_local(struct krdma_cb *cb, const char *host, const char *port) {
	int ret;
	struct sockaddr_in addr;

	/* Create cm_id. */
	cb->cm_id = rdma_create_id(&init_net, krdma_cma_event_handler, cb,
			RDMA_PS_TCP, IB_QPT_RC);
	if (IS_ERR(cb->cm_id)) {
		ret = PTR_ERR(cb->cm_id);
		krdma_err("rdma_create_id error %d\n", ret);
		goto exit;
	}
	krdma_debug("created cm_id %p\n", cb->cm_id);

	/* Bind address. */
	ret = krdma_set_addr(&addr, host, port);
	if (ret < 0)
		goto free_cm_id;
	ret = rdma_bind_addr(cb->cm_id, (struct sockaddr *)&addr);
	if (ret) {
		krdma_err("rdma_bind_addr failed, ret %d\n", ret);
		goto free_cm_id;
	}
	krdma_debug("rdma_bind_addr succeed, device[%s] port_num[%u]\n",
			cb->cm_id->device->name, cb->cm_id->port_num);

	return 0;

free_cm_id:
	rdma_destroy_id(cb->cm_id);
	cb->cm_id = NULL;
exit:
	return ret;
}

/* 
 * Called after __krdma_bound_dev_{local, remote}.
 * Allocate pd, cq, qp, mr, freed by caller
 */
static int krdma_init_cb(struct krdma_cb *cb) {
	int ret;
	struct ib_cq_init_attr cq_attr;
	struct ib_qp_init_attr qp_init_attr;

	/* Create Protection Domain. */
	cb->pd = ib_alloc_pd(cb->cm_id->device, IB_PD_UNSAFE_GLOBAL_RKEY);
	if (IS_ERR(cb->pd)) {
		ret = PTR_ERR(cb->pd);
		krdma_err("ib_alloc_pd failed\n");
		goto exit;
	}
	krdma_debug("ib_alloc_pd succeed, cm_id %p\n", cb->cm_id);

	/* Create send Completion Queue. */
	memset(&cq_attr, 0, sizeof(cq_attr));
	cq_attr.cqe = RDMA_CQ_QUEUE_DEPTH;
	cq_attr.comp_vector = 0;
	cb->send_cq = ib_create_cq(cb->cm_id->device, NULL, NULL, cb, &cq_attr);
	if (IS_ERR(cb->send_cq)) {
		ret = PTR_ERR(cb->send_cq);
		krdma_err("ib_create_cq failed, ret%d\n", ret);
		goto free_pd;
	}

	/* Create recv Completion Queue. */
	memset(&cq_attr, 0, sizeof(cq_attr));
	cq_attr.cqe = RDMA_CQ_QUEUE_DEPTH;
	cq_attr.comp_vector = 0;
	cb->recv_cq = ib_create_cq(cb->cm_id->device, NULL, NULL, cb, &cq_attr);
	if (IS_ERR(cb->recv_cq)) {
		ret = PTR_ERR(cb->recv_cq);
		krdma_err("ib_create_cq failed, ret%d\n", ret);
		goto free_send_cq;
	}

	krdma_debug("ib_create_cq succeed, cm_id %p\n", cb->cm_id);

	/* Create Queue Pair. */
	memset(&qp_init_attr, 0, sizeof(qp_init_attr));
	qp_init_attr.cap.max_send_wr = RDMA_SEND_QUEUE_DEPTH;
	qp_init_attr.cap.max_recv_wr = RDMA_RECV_QUEUE_DEPTH;
	qp_init_attr.cap.max_recv_sge = 1;
	qp_init_attr.cap.max_send_sge = 1;
	/* Mlx doesn't support inline sends for kernel QPs (yet) */
	qp_init_attr.cap.max_inline_data = 0;
	qp_init_attr.qp_type = IB_QPT_RC;
	qp_init_attr.send_cq = cb->send_cq;
	qp_init_attr.recv_cq = cb->recv_cq;
	qp_init_attr.sq_sig_type = IB_SIGNAL_REQ_WR;
	ret = rdma_create_qp(cb->cm_id, cb->pd, &qp_init_attr);
	if (ret) {
		krdma_err("rdma_create_qp failed, ret %d\n", ret);
		goto free_recv_cq;
	}
	cb->qp = cb->cm_id->qp;
	krdma_debug("ib_create_qp succeed, cm_id %p\n", cb->cm_id);

	/* Setup buffers. */
	ret = krdma_setup_mr(cb);
	if (ret) {
		krdma_err("krdma_setup_mr failed, ret %d\n", ret);
		goto free_qp;
	}

	mutex_lock(&cb->rlock);
	ret = krdma_post_recv(cb);
	if (ret) {
		krdma_err("krdma_post_recv failed, ret %d\n", ret);
		mutex_unlock(&cb->rlock);
		goto free_buffers;
	}
	mutex_unlock(&cb->rlock);
	return 0;

free_buffers:
	krdma_free_mr(cb);
free_qp:
	rdma_destroy_qp(cb->cm_id);
free_recv_cq:
	ib_destroy_cq(cb->recv_cq);
free_send_cq:
	ib_destroy_cq(cb->send_cq);
free_pd:
	ib_dealloc_pd(cb->pd);
exit:
	return ret;
}

static int __krdma_connect(struct krdma_cb *cb) {
	int ret;
	struct rdma_conn_param conn_param;

	/* Connect to remote. */
	memset(&conn_param, 0, sizeof(conn_param));
	/*
	 * The maximum number of times that a data transfer operation
	 * should be retried on the connection when an error occurs. This setting controls
	 * the number of times to retry send, RDMA, and atomic operations when timeouts
	 * occur.
	 */
	conn_param.retry_count = 7;
	/*
	 * The maximum number of times that a send operation from the
	 * remote peer should be retried on a connection after receiving a receiver not
	 * ready (RNR) error.
	 */
	conn_param.rnr_retry_count = 7;

	ret = rdma_connect(cb->cm_id, &conn_param);
	if (ret) {
		krdma_err("rdma_connect failed, ret %d\n", ret);
		return ret;
	}
	wait_for_completion(&cb->cm_done);
	if (cb->state != KRDMA_CONNECTED) {
		krdma_err("wait for KRDMA_CONNECTED state, but get %d\n", cb->state);
		if (cb->state == KRDMA_CONNECT_REJECTED)
			ret = -CLIENT_RETRY;
		else
			ret = -CLIENT_EXIT;

		return ret;
	}
	krdma_debug("krdma_connect_single succeed, cm_id %p\n", cb->cm_id);
	return 0;
}

static int __krdma_listen(struct krdma_cb *cb) {
	int ret;

	ret = rdma_listen(cb->cm_id, 3);
	if (ret) {
		krdma_err("rdma_listen failed: %d\n", ret);
		return ret;
	}
	krdma_debug("rdma_listen start...\n");
	return 0;
}

static int __krdma_accept(struct krdma_cb *cb) {
	int ret;
	struct rdma_conn_param conn_param;

	/* Accept */
	memset(&conn_param, 0, sizeof conn_param);
	conn_param.retry_count = conn_param.rnr_retry_count = 7;

	ret = rdma_accept(cb->cm_id, &conn_param);
	if (ret) {
		krdma_err("rdma_accept error: %d\n", ret);
		goto exit;
	}
	wait_for_completion(&cb->cm_done);
	if (cb->state != KRDMA_CONNECTED) {
		krdma_err("wait for KRDMA_CONNECTED state, but get %d\n", cb->state);
		goto exit;
	}

	krdma_debug("new connection accepted with the following attributes:\n"
		"local: %pI4:%d\nremote: %pI4:%d\n",
		&((struct sockaddr_in *)&cb->cm_id->route.addr.src_addr)->sin_addr.s_addr,
		ntohs(((struct sockaddr_in *)&cb->cm_id->route.addr.src_addr)->sin_port),
		&((struct sockaddr_in *)&cb->cm_id->route.addr.dst_addr)->sin_addr.s_addr,
		ntohs(((struct sockaddr_in *)&cb->cm_id->route.addr.dst_addr)->sin_port));

exit:
	return ret;
}

////////////////////////////////////////////////////////////////////
//////////////////////SEND/RECV Functions///////////////////////////
////////////////////////////////////////////////////////////////////

static int krdma_post_recv(struct krdma_cb *cb);
/* @return wr_id of wc if polling succeed. */
static int krdma_poll(struct krdma_cb *cb, imm_t *imm, size_t *length,
		bool block, krdma_poll_type_t type)
{
	struct ib_wc wc;
	int ret = 0;
	int retry_cnt = 0;
	struct ib_cq *cq;
#ifdef DYNAMIC_POLLING_INTERVAL
	uint32_t usec_sleep = 1;
#endif

	might_sleep();

	switch (type) {
	case KRDMA_SEND:
		cq = cb->send_cq;
		break;
	case KRDMA_RECV:
		cq = cb->recv_cq;
		break;
	default:
		return -EINVAL;
	}

repoll:
	switch (cb->state) {
		case KRDMA_ERROR:
			return -STATE_ERROR;
		case KRDMA_DISCONNECTED:
			return -EPIPE;
		default:
			break;
			/* Okay by default. */
	}
	/* Spin waiting for send/recv completion */
	while ((ret = ib_poll_cq(cq, 1, &wc) == 1)) {
		if (wc.status != IB_WC_SUCCESS) {
			if (wc.status == IB_WC_WR_FLUSH_ERR)
				continue;
			krdma_err("wc.status: %s wr.id %llu\n", ib_wc_status_msg(wc.status), wc.wr_id);
			return -STATE_ERROR;
		}

		if (imm && (wc.wc_flags & IB_WC_WITH_IMM)) {
			*imm = ntohl(wc.ex.imm_data);
		}
		if (length && (wc.opcode == IB_WC_RECV)) {
			*length = wc.byte_len;
		}
		switch (wc.opcode) {
		case IB_WC_SEND:
			BUG_ON(type != KRDMA_SEND);
			krdma_debug("cb %p send completion, wr_id 0x%llx retry %d times\n", cb, wc.wr_id, retry_cnt);
			break;
		case IB_WC_RECV:
			BUG_ON(type != KRDMA_RECV);
			krdma_debug("cb %p recv completion, wr_id 0x%llx\n", cb, wc.wr_id);
			if ((ret = (krdma_post_recv(cb))) < 0) {
				return -STATE_ERROR;
			}
			break;
		case IB_WC_RDMA_READ:
			BUG_ON(type != KRDMA_READ);
			krdma_debug("cb %p read completion, wr_id 0x%llx retry %d times\n",
				cb, wc.wr_id, retry_cnt);
			break;
		case IB_WC_RDMA_WRITE:
			BUG_ON(type != KRDMA_WRITE);
			krdma_debug("cb %p write completion, wr_id 0x%llx retry %d times\n",
				cb, wc.wr_id, retry_cnt);
			/*if ((ret = (krdma_post_recv(cb))) < 0) {
				return -STATE_ERROR;
			}*/
			break;
		default:
			krdma_err("Unexpected opcode %u\n", wc.opcode);
			BUG();
		}
		return wc.wr_id;
	}
	if (ret == 0) {
		/*
		 * Occasionally, we will take CPU for too long and cause RCU to stuck,
		 * because interrupt and preemption are disabled in ib_poll_cq and thus
		 * the window for preemption is small. For this reason, we voluntarily
		 * call schedule() here.
		 */
		if (block) {
			retry_cnt++;
			/*
			 * Most send requests complete in 20~60 polls (At least for local
			 * loop back.
			 */
			if (retry_cnt > 128) {
#ifdef DYNAMIC_POLLING_INTERVAL
				/* A TCP-like Additive Increase and Multiplicative Decrease rule. */
				usec_sleep = (usec_sleep + 1) > 1000 ? 1000 : (usec_sleep + 1);
				usleep_range(usec_sleep, usec_sleep);
#else
				schedule();
#endif
			}
			if (retry_cnt >= 10000 && retry_cnt % 10000 == 0) {
				/* Issue warning per ~10s */
				krdma_err("cb %p waiting for send too LONG!\n", cb);
			}
			goto repoll;
		}
		else {
			return -EAGAIN;
		}
	}
	return ret;
}

static bool search_send_buf(struct krdma_cb *cb, uint16_t txid,
		krdma_send_trans_t **trans, enum krdma_trans_state state)
{
	int i;
	krdma_send_trans_t *send_trans_buf = cb->mr.sr_mr.send_trans_buf;

	for (i = 0; i < RDMA_SEND_BUF_SIZE; i++) {
		if (send_trans_buf[i].state == state && send_trans_buf[i].txid
				== txid) {
			*trans = &send_trans_buf[i];
			return true;
		}
	}
	return false;
}

static bool search_recv_buf(struct krdma_cb *cb, uint16_t txid,
		krdma_recv_trans_t **trans, enum krdma_trans_state state)
{
	int i;
	krdma_recv_trans_t *recv_trans_buf = cb->mr.sr_mr.recv_trans_buf;

	for (i = 0; i < RDMA_RECV_BUF_SIZE; i++) {
		if (recv_trans_buf[i].state == state &&
				(recv_trans_buf[i].txid == txid || txid == 0xFF)) {
			*trans = &recv_trans_buf[i];
			return true;
		}
	}
	return false;
}

static int search_empty_send_buf(struct krdma_cb *cb, krdma_send_trans_t **trans)
{
	int i;
	krdma_send_trans_t *send_trans_buf = cb->mr.sr_mr.send_trans_buf;

	/* TODO: Is this necessary? */
	static int last_schedule = 0;

	last_schedule = (last_schedule + 1) % RDMA_SEND_BUF_SIZE;

	for (i = last_schedule; i < last_schedule + RDMA_SEND_BUF_SIZE; i++) {
		if (send_trans_buf[i % RDMA_SEND_BUF_SIZE].state == INVALID) {
			*trans = &send_trans_buf[i % RDMA_SEND_BUF_SIZE];
			last_schedule = i % RDMA_SEND_BUF_SIZE;
			return i % RDMA_SEND_BUF_SIZE;
		}
	}
	/* Buffer overflow */
	BUG();
}

static int search_empty_recv_buf(struct krdma_cb *cb, krdma_recv_trans_t **trans)
{
	int i;
	krdma_recv_trans_t *recv_trans_buf = cb->mr.sr_mr.recv_trans_buf;

	for (i = 0; i < RDMA_RECV_BUF_SIZE; i++) {
		if (recv_trans_buf[i].state == INVALID) {
			*trans = &recv_trans_buf[i];
			return i;
		}
	}
	return -ENOENT;
}

static void build_posted_send_trans(const struct krdma_cb *cb, uint16_t txid,
		krdma_send_trans_t *trans) {
	trans->txid = txid;
	trans->state = POSTED;
}

static void build_posted_recv_trans(const struct krdma_cb *cb,
		krdma_recv_trans_t *trans) {
	trans->state = POSTED;
}

/* trans->length should be correctly set before calling this function. */
static uint16_t get_trans_txid(krdma_recv_trans_t *trans) {
	tx_add_t tx_add;

	memcpy(&tx_add, &trans->imm, sizeof(imm_t));
	memcpy(((char *)&tx_add) + sizeof(imm_t), trans->recv_buf + (trans->length
		- (sizeof(tx_add_t) - sizeof(imm_t))),
		sizeof(tx_add_t) - sizeof(imm_t));

	return tx_add.txid;
}

static void build_polled_recv_trans(const struct krdma_cb *cb, imm_t imm,
		size_t length, krdma_recv_trans_t *trans) {
	trans->imm = imm;
	trans->length = length;

	trans->txid = get_trans_txid(trans);
	trans->state = POLLED;
}

static void build_krdma_send_output(const struct krdma_cb *cb,
		krdma_send_trans_t *trans)
{
	trans->state = INVALID;
}

/*
 * Content + additional(without first 32-bit data, which is store in imm)
 *
 * imm:
 * |<---1st part of tx_add, i.e., the first 32 bit--->|
 * payload: cb->recv_buf:
 * |<------------------------sz=wc.byte_len------------------------------->|
 * |<----real data (sz=ret_val of send/recv)--->|<---2nd part of tx_add--->|
 */
/* rdma transaction->krdma interfaces. */
static size_t build_krdma_recv_output(struct krdma_cb *cb,
		krdma_recv_trans_t *trans, char *buffer, tx_add_t *tx_add)
{
	size_t real_length = trans->length - (sizeof(tx_add_t) - sizeof(imm_t));

	memcpy(tx_add, &trans->imm, sizeof(imm_t));
	memcpy(((char *)tx_add) + sizeof(imm_t), trans->recv_buf + real_length,
			sizeof(tx_add_t) - sizeof(imm_t));
	memcpy(buffer, trans->recv_buf, real_length);

	trans->state = INVALID;

	return real_length;
}

static int krdma_post_recv(struct krdma_cb *cb)
{
	int ret = 0;
	int slot;
	krdma_recv_trans_t *recv_trans;
	struct ib_recv_wr *bad_wr;
	krdma_recv_trans_t *recv_trans_buf = cb->mr.sr_mr.recv_trans_buf;

	while ((slot = search_empty_recv_buf(cb, &recv_trans)) != -ENOENT) {
		build_posted_recv_trans(cb, recv_trans);
		ret = ib_post_recv(cb->qp, &recv_trans_buf[slot].rq_wr, &bad_wr);
		if (ret) {
			krdma_err("ib_post_recv error, ret %d\n", ret);
			return -STATE_ERROR;
		}
	}

	return ret;
}

/*
 * @param tx_add the txid field should be set as input parameter, 0xFF denotes
 * acceptance all receiving requests.
 * wr_id means which slot is used for transmission.
 */
int krdma_receive(struct krdma_cb *cb, char *buffer)
{
	int ret;
	size_t len;
	imm_t imm;
	/* The desired txid. */
	tx_add_t tx_add = {
		.txid = 0xFF,
	};
	uint16_t txid = tx_add.txid;
	uint16_t recv_txid;
	uint32_t usec_sleep = 0;
	krdma_recv_trans_t *recv_trans;
	int retry_cnt = 0;
	unsigned long flag = SOCK_NONBLOCK;
	krdma_recv_trans_t *recv_trans_buf = cb->mr.sr_mr.recv_trans_buf;

	BUILD_BUG_ON(sizeof(tx_add_t) < sizeof(imm_t));
	BUG_ON(cb->read_write);

	krdma_debug("%s: cb %p receive 0x%x\n", __func__, cb, tx_add.txid);

	mutex_lock(&cb->rlock);

repoll:
	/* Search in the buffer. */
	if (search_recv_buf(cb, txid, &recv_trans, POLLED)) {
		ret = build_krdma_recv_output(cb, recv_trans, buffer, &tx_add);
		mutex_unlock(&cb->rlock);
		krdma_debug("%s: cb %p find 0x%x in buffer\n", __func__, cb, tx_add.txid);

		return ret;
	}

	/* Not found in the buffer. */
	ret = krdma_poll(cb, &imm, &len, false, KRDMA_RECV);
	if (ret < 0) {
		if (ret == -EAGAIN) {
			retry_cnt++;
			/*
			 * Besides the incrment of concurrency (only one task runnable each
			 * time) of krdma_receive, unlock here is also used to avoid some
			 * kinds (not clearly learnt) of deadlock.
			 */
			mutex_unlock(&cb->rlock);
			usec_sleep = (usec_sleep + 1) > 1000 ? 1000 : (usec_sleep + 1);
			usleep_range(usec_sleep, usec_sleep);
			if ((flag & SOCK_NONBLOCK) && retry_cnt > 128) {
				return -EAGAIN;
			}
			mutex_lock(&cb->rlock);
			goto repoll;
		}
		mutex_unlock(&cb->rlock);
		krdma_err("krdma_poll error, ret %d\n", ret);
		return ret;
	}
	usec_sleep = 0;

	recv_trans = &recv_trans_buf[ret];
	if (unlikely(recv_trans->state != POSTED)) {
		mutex_unlock(&cb->rlock);
		BUG();
	}
	build_polled_recv_trans(cb, imm, len, recv_trans);
	recv_txid = recv_trans->txid;

	/* Not my transaction. */
	if (txid != 0xFF && recv_txid != txid) {
		krdma_debug("%s: cb %p wish 0x%x is 0x%x\n", __func__, cb, txid, recv_txid);
		goto repoll;
	}
	else {
		/* My transaction ! */
		build_krdma_recv_output(cb, recv_trans, buffer, &tx_add);
		krdma_debug("%s: cb %p find my tx 0x%x\n", __func__, cb, tx_add.txid);
	}

	mutex_unlock(&cb->rlock);
	krdma_debug("%s: cb %p received 0x%x\n", __func__, cb, txid);
	return ret >= 0 ? len - (sizeof(tx_add_t) - sizeof(imm_t)) : ret;
}

/* wr_id of send means txid. */
int krdma_send(struct krdma_cb *cb, const char *buffer, size_t length)
{
	int ret = 0;
	struct ib_send_wr *bad_wr;
	imm_t imm;
	krdma_send_trans_t *send_trans;
	tx_add_t tx_add = {
		.txid = 0xFF,
	};
	uint16_t txid = tx_add.txid;
	size_t recv_length;
	int slot;
	krdma_send_trans_t *send_trans_buf = cb->mr.sr_mr.send_trans_buf;

	BUG_ON(cb->read_write);
	mutex_lock(&cb->slock);

	slot = search_empty_send_buf(cb, &send_trans);
	build_posted_send_trans(cb, txid, send_trans);
	krdma_debug("%s: cb %p send 0x%x length %lu\n", __func__, cb, send_trans->txid, length);

	send_trans_buf[slot].send_sge.length = length + (sizeof(tx_add_t) - sizeof(imm_t));
	send_trans_buf[slot].sq_wr.wr_id = tx_add.txid;
	send_trans_buf[slot].sq_wr.ex.imm_data = htonl(*(const uint32_t*) &tx_add);
	memcpy(send_trans_buf[slot].send_buf + length, (((const char *) &tx_add) + sizeof(imm_t)),
			sizeof(tx_add_t) - sizeof(imm_t));
	memcpy(send_trans_buf[slot].send_buf, buffer, length);

	ret = ib_post_send(cb->qp, &send_trans_buf[slot].sq_wr, &bad_wr);
	if (ret) {
		mutex_unlock(&cb->slock);
		krdma_err("ib_post_send failed, ret %d\n", ret);
		return ret;
	}

	if (unlikely(search_send_buf(cb, txid, &send_trans, POLLED))) {
		mutex_unlock(&cb->slock);
		BUG();
	}

	ret = krdma_poll(cb, &imm, &recv_length, true, KRDMA_SEND);
	if (ret < 0) {
		mutex_unlock(&cb->slock);
		krdma_err("krdma_poll error, ret %d\n", ret);
		return ret;
	}

	/* Not my transaction, which should not happen. */
	if (unlikely(ret != txid)) {
		mutex_unlock(&cb->slock);
		BUG();
	}
	/* My transaction! */
	else {
		bool searched;
		searched = search_send_buf(cb, txid, &send_trans, POSTED);
		if (unlikely(!searched)) {
			mutex_unlock(&cb->slock);
			BUG();
		}
		build_krdma_send_output(cb, send_trans);
	}
	mutex_unlock(&cb->slock);

	krdma_debug("%s: cb %p sent 0x%x\n", __func__, cb, txid);

	return ret >= 0 ? length : ret;
}

////////////////////////////////////////////////////////////////////
//////////////////RDMA READ/WRITE Functions/////////////////////////
////////////////////////////////////////////////////////////////////

static void dump_rw_info(krdma_rw_info_t *info, char *desc) {
	if (!info || !desc) {
		krdma_err("NULL info!\n");
		return;
	}
	krdma_debug("[krdma_rw_info_t]: %s = { addr %llu, rkey %u, qp_num %d, lid %hu }\n",
		desc, info->addr, info->rkey, info->qp_num, info->lid);
}

static const char *server_host = "10.0.16.6";
static const char *server_port = "22421";

static int krdma_exch_info_client(struct krdma_cb *cb) {
	int ret = 0;
	struct ktcp_cb *conn_tcp = NULL;
	krdma_rw_info_t *local_info = cb->mr.rw_mr.local_info;
	krdma_rw_info_t *remote_info = cb->mr.rw_mr.remote_info;

	if (!(local_info && remote_info)) {
		krdma_err("NULL info!\n");
		return -EINVAL;
	}

	ret = ktcp_connect(server_host, server_port, &conn_tcp);
	if (ret < 0) {
		krdma_err("ktcp_connect failed with ret %d.\n", ret);
		goto out_release_tcp;
	}

	ret = ktcp_send(conn_tcp, 
		(const char *) local_info, sizeof(krdma_rw_info_t));
	if (ret < 0 || ret < sizeof(krdma_rw_info_t)) {
		krdma_err("ktcp_send failed with ret %d.\n", ret);
		goto out_release_tcp;
	}

	ret = ktcp_receive(conn_tcp, (char *) remote_info);
	if (ret < 0 || ret < sizeof(krdma_rw_info_t)) {
		krdma_err("ktcp_receive failed with ret %d.\n", ret);
		goto out_release_tcp;
	}

	dump_rw_info(local_info, "Client local_info");
	dump_rw_info(remote_info, "Client remote_info");

out_release_tcp:
	ktcp_release(conn_tcp);
	return ret > 0 ? -ret : ret;
}

static int krdma_exch_info_server(struct krdma_cb *cb) {
	int ret = 0;
	struct ktcp_cb *listen_tcp = NULL;
	struct ktcp_cb *accept_tcp = NULL;
	krdma_rw_info_t *local_info = cb->mr.rw_mr.local_info;
	krdma_rw_info_t *remote_info = cb->mr.rw_mr.remote_info;

	if (!(local_info && remote_info)) {
		krdma_err("NULL info!\n");
		return -EINVAL;
	}

	ret = ktcp_listen(server_host, server_port, &listen_tcp);
	if (ret < 0) {
		krdma_err("ktcp_listen failed with ret %d.\n", ret);
		goto exit;
	}

	ret = ktcp_accept(listen_tcp, &accept_tcp);
	if (ret < 0) {
		krdma_err("ktcp_accept failed with ret %d.\n", ret);
		goto out_release_listen_tcp;
	}

	ret = ktcp_receive(accept_tcp, (char *) remote_info);
	if (ret < 0 || ret < sizeof(krdma_rw_info_t)) {
		krdma_err("ktcp_receive failed with ret %d.\n", ret);
		goto out_release_accept_tcp;
	}

	ret = ktcp_send(accept_tcp, 
		(const char *) local_info, sizeof(krdma_rw_info_t));
	if (ret < 0 || ret < sizeof(krdma_rw_info_t)) {
		krdma_err("ktcp_send failed with ret %d.\n", ret);
		goto out_release_accept_tcp;
	}

	dump_rw_info(local_info, "Server local_info");
	dump_rw_info(remote_info, "Server remote_info");

out_release_accept_tcp:
	ktcp_release(accept_tcp);
out_release_listen_tcp:
	ktcp_release(listen_tcp);
exit:
	return ret > 0 ? -ret : ret;
}

int krdma_rw_init_client(const char *host, const char *port, struct krdma_cb **conn_cb) {
	int ret;
	struct krdma_cb *cb;

	if (host == NULL || port == NULL || conn_cb == NULL)
		return -EINVAL;

	ret = __krdma_create_cb(&cb, KRDMA_CLIENT_CONN);
	if (ret) {
		krdma_err("__krdma_create_cb fail, ret %d\n", ret);
		return ret;
	}
	cb->read_write = true;

retry:
	ret = krdma_connect_single(host, port, cb);
	if (ret == 0) {
		/*
		 * If multiple clients desire to connect to remote servers, only one of
		 * them can call this function. Others must be blocked even if conn_cb
		 * has not been set here. Then double checking whether conn_cb is
		 * non-NULL can ensure the correctness of lazy connection.
		 */
		smp_mb();
		*conn_cb = cb;
		goto rest_client_init;
	}
	if (ret == -CLIENT_RETRY &&
				++cb->retry_count < RDMA_CONNECT_RETRY_MAX) {
		krdma_err("krdma_connect_single failed, retry_count %d, " \
				"reconnecting...\n", cb->retry_count);
		msleep(1000);
		goto retry;
	}
	else {
		goto out_free_cb;
	}

rest_client_init:
	ret = krdma_exch_info_client(cb);
	if (ret < 0) {
		krdma_err("krdma_exch_info_client failed with ret %d\n", ret);
		goto out_release_cb;
	}

	krdma_debug("%p krdma_rw_init_client succeed\n", cb);
	return 0;

out_release_cb:
	krdma_release_cb(cb);
out_free_cb:
	__krdma_free_cb(cb);
	*conn_cb = NULL;
	krdma_err("krdma_rw_init_client failed, ret: %d\n", ret);
	return ret;
}

int krdma_rw_init_server(const char *host, const char *port, struct krdma_cb **cbp) {
	int ret;
	struct krdma_cb *accept_cb;
	struct krdma_cb *listen_cb;
	
	ret = krdma_listen(host, port, &listen_cb);
	if (ret < 0) {
		krdma_err("krdma_listen failed.\n");
		goto exit;
	}
	listen_cb->read_write = true;

	accept_cb = __krdma_wait_for_connect_request(listen_cb);
	if (accept_cb == NULL) {
		krdma_err("__krdma_wait_for_connect_request failed.\n");
		ret = -STATE_ERROR;
		goto out_release_listen_cb;
	}
	accept_cb->read_write = true;
	*cbp = accept_cb;

	krdma_debug("get connection, cm_id %p\n", accept_cb->cm_id);

	ret = krdma_init_cb(accept_cb);
	if (ret < 0)
		goto out_free_accept_cb;

	ret = __krdma_accept(accept_cb);
	if (ret < 0)
		goto out_release_accept_cb;
	
	ret = krdma_exch_info_server(accept_cb);
	if (ret < 0)
		goto out_release_accept_cb;

	krdma_debug("%p krdma_rw_init_server succeed\n", accept_cb);
	rdma_destroy_id(listen_cb->cm_id);
	__krdma_free_cb(listen_cb);
	return 0;

out_release_accept_cb:
	krdma_release_cb(accept_cb);
	
out_free_accept_cb:
	__krdma_free_cb(accept_cb);
	*cbp = NULL;

out_release_listen_cb:
	rdma_destroy_id(listen_cb->cm_id);
	__krdma_free_cb(listen_cb);
exit:
	krdma_err("krdma_rw_init_server failed, ret: %d\n", ret);
	return ret;
}

static void build_krdma_read_output(
	struct krdma_cb *cb, char *buffer, size_t length) {
	BUG_ON(!cb->read_write);
	if (!cb)
		return;
	memcpy(buffer, cb->mr.rw_mr.local_info->buf, length);
}

static void build_krdma_write_input(
	struct krdma_cb *cb, const char *buffer, size_t length) {
	BUG_ON(!cb->read_write);
	if (!cb)
		return;
	memcpy(cb->mr.rw_mr.local_info->buf, buffer, length);
}

int krdma_read(struct krdma_cb *cb, char *buffer, size_t length) {
	struct ib_rdma_wr rdma_wr;
	struct ib_send_wr *bad_wr = NULL;
	struct ib_sge sge[1];
	int ret;
	imm_t imm;
	size_t len;

	BUG_ON(!cb->read_write);

	memset(sge, 0, sizeof(sge));
	memset(&rdma_wr, 0, sizeof(rdma_wr));

	sge[0].addr = cb->mr.rw_mr.local_info->addr;
	sge[0].length = cb->mr.rw_mr.local_info->length;
	sge[0].lkey = cb->pd->local_dma_lkey;

	rdma_wr.remote_addr = (uintptr_t) cb->mr.rw_mr.remote_info->addr;
	rdma_wr.rkey = cb->mr.rw_mr.remote_info->rkey;

	rdma_wr.wr.sg_list = sge;
	rdma_wr.wr.wr_id = 999;
	rdma_wr.wr.opcode = IB_WR_RDMA_READ;
	rdma_wr.wr.send_flags = IB_SEND_SIGNALED;
	rdma_wr.wr.num_sge = 1;
	rdma_wr.wr.next = NULL;

	ret = ib_post_send(cb->qp, &rdma_wr.wr, &bad_wr);
	if (unlikely(ret)) {
		krdma_err("ib_post_send failed.\n");
		return ret;
	}
	krdma_debug("ib_post_send succeed.\n");

	ret = krdma_poll(cb, &imm, &len, true, KRDMA_READ);
	if (unlikely(ret < 0)) {
		krdma_err("krdma_poll failed with ret %d.\n", ret);
		return ret;
	}

	build_krdma_read_output(cb, buffer, length);
	krdma_debug("krdma_read succeed with buffer = %s, length = %lu.\n", 
		buffer, length);
	return 0;
}

int krdma_write(struct krdma_cb *cb, const char *buffer, size_t length) {
	struct ib_rdma_wr rdma_wr;
	struct ib_send_wr *bad_wr = NULL;
	struct ib_sge sge[1];
	int ret;
	imm_t imm = 666;
	size_t len;

	BUG_ON(!cb->read_write);

	memset(sge, 0, sizeof(sge));
	memset(&rdma_wr, 0, sizeof(rdma_wr));

	sge[0].addr = cb->mr.rw_mr.local_info->addr;
	sge[0].length = cb->mr.rw_mr.local_info->length;
	sge[0].lkey = cb->pd->local_dma_lkey;

	rdma_wr.remote_addr = (uintptr_t) cb->mr.rw_mr.remote_info->addr;
	rdma_wr.rkey = cb->mr.rw_mr.remote_info->rkey;

	rdma_wr.wr.sg_list = sge;
	rdma_wr.wr.wr_id = 999;
	rdma_wr.wr.ex.imm_data = imm;
	rdma_wr.wr.opcode = IB_WR_RDMA_WRITE_WITH_IMM;
	rdma_wr.wr.send_flags = IB_SEND_SIGNALED;
	rdma_wr.wr.num_sge = 1;
	rdma_wr.wr.next = NULL;

	build_krdma_write_input(cb, buffer, length);

	ret = ib_post_send(cb->qp, &rdma_wr.wr, &bad_wr);
	if (unlikely(ret)) {
		krdma_err("ib_post_send failed.\n");
		return ret;
	}
	krdma_debug("ib_post_send succeed.\n");

	ret = krdma_poll(cb, &imm, &len, true, KRDMA_WRITE);
	if (unlikely(ret < 0)) {
		krdma_err("krdma_poll failed with ret %d.\n", ret);
		return ret;
	}

	krdma_debug("krdma_write succeed with buffer = %s, length = %lu.\n", 
		buffer, length);
	return 0;
}


////////////////////////////////////////////////////////////////////
/////////////////////////TEST Functions/////////////////////////////
////////////////////////////////////////////////////////////////////


static int sr_client(void *data) {
	struct krdma_cb *cb = NULL;
	int ret = 0;
	int buf = 0xdeadbeef;

	ret = krdma_connect("172.16.0.2", "22223", &cb);
	if (ret) {
		krdma_err("krdma_connect failed.\n");
		return ret;
	}
	krdma_debug("krdma_connect succeed.\n");
	ret = krdma_send(cb, (const char *) &buf, sizeof(buf));
	if (ret) {
		krdma_err("krdma_send failed.\n");
		goto exit;
	}
	krdma_debug("krdma_send succeed.\n");

exit:
	krdma_release_cb(cb);
	__krdma_free_cb(cb);
	return ret;
}

static int sr_server(void *data) {
	struct krdma_cb *listen_cb = NULL;
	struct krdma_cb *accept_cb = NULL;
	int ret = 0;
	int buf = -1;

	ret = krdma_listen(SERVER_IP, "23333", &listen_cb);
	if (ret) {
		krdma_err("krdma_listen failed.\n");
		return ret;
	}
	krdma_debug("krdma_listen succeed.\n");
	while (1) {
		if (kthread_should_stop()) {
			ret = 0;
			goto free_listen_cb;
		}
		ret = krdma_accept(listen_cb, &accept_cb);
		if (ret < 0) {
			krdma_err("krdma_accept failed.\n");
			goto free_listen_cb;
		}
		krdma_debug("krdma_accept succeed.\n");
		ret = krdma_receive(accept_cb, (char *) &buf);
		if (ret < 0) {
			krdma_err("krdma_receive failed.\n");
			goto free_accept_cb;
		}
		krdma_debug("krdma_receive succeed len %d.\n", ret);
	}

free_accept_cb:
	krdma_release_cb(accept_cb);
	__krdma_free_cb(accept_cb);
free_listen_cb:
	rdma_destroy_id(listen_cb->cm_id);
	listen_cb->cm_id = NULL;
	__krdma_free_cb(listen_cb);
	return ret;
}

static int rw_client(void *data) {
	struct krdma_cb *cb = NULL;
	int ret = 0;
	
	char word[20] = "Hello RDMA";
	char res[20];
	ret = krdma_rw_init_client("172.16.0.2", "11232", &cb);
	if (ret < 0)
		goto exit;
	krdma_write(cb, word, sizeof(word));
	krdma_read(cb, res, sizeof(res));
	krdma_release_cb(cb);
	__krdma_free_cb(cb);

exit:
	return ret;
}

static int rw_server(void *data) {
	struct krdma_cb *cb = NULL;
	int ret = 0;
	imm_t imm;
	size_t length;

	ret = krdma_rw_init_server(SERVER_IP, "19932", &cb);
	if (ret < 0)
		goto exit;
	while (!kthread_should_stop()) {
		krdma_poll(cb, &imm, &length, true, 2);
	}
	krdma_release_cb(cb);
	__krdma_free_cb(cb);
exit:
	return ret;
}

static struct task_struct *thread = NULL;

static int server = 1; // server or client?
module_param(server, int, S_IRUGO);

static int rw = 1; // read/write or send/recv?
module_param(rw, int, S_IRUGO);

int __init krdma_init(void) {
	int ret;
	int (*func[4])(void *data) = {
		sr_client,
		sr_server,
		rw_client,
		rw_server
	};

	char *name[4] = {
		"sr_client",
		"sr_server",
		"rw_client",
		"rw_server"
	};

	int choice = server + 2 * rw;

	krdma_err("server %d\n", server);
	krdma_err("read/write %d\n", rw);

	thread = kthread_run(func[choice], NULL, name[choice]);
	if (IS_ERR(thread)) {
		krdma_err("%s start failed.\n", name[choice]);
		ret = PTR_ERR(thread);
		return ret;
	}
    return 0;
}

void __exit krdma_exit(void) {
	int ret;
	send_sig(SIGKILL, thread, 1);
	ret = kthread_stop(thread);
	if (ret < 0) {
		krdma_err("kill thread failed.\n");
	}
}

module_init(krdma_init);
module_exit(krdma_exit);
MODULE_AUTHOR("dirk");
MODULE_DESCRIPTION("RDMA read write example");
MODULE_LICENSE("GPL");
