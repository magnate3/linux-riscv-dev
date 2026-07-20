#include <linux/module.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/stat.h>
#include <linux/inet.h>
#include <linux/wait.h>
#include <linux/string.h>
//#include <linux/jiffies.h>

#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <linux/version.h>
MODULE_AUTHOR("Austin Pohlmann");
MODULE_LICENSE("GPL v2");

#define TEST_FAST_REG_MR 0
static int mode = -1;
module_param(mode, int, S_IRUGO);

const char* cma_event[] = {
	"RDMA_CM_EVENT_ADDR_RESOLVED",
	"RDMA_CM_EVENT_ADDR_ERROR",
	"RDMA_CM_EVENT_ROUTE_RESOLVED",
	"RDMA_CM_EVENT_ROUTE_ERROR",
	"RDMA_CM_EVENT_CONNECT_REQUEST",
	"RDMA_CM_EVENT_CONNECT_RESPONSE",
	"RDMA_CM_EVENT_CONNECT_ERROR",
	"RDMA_CM_EVENT_UNREACHABLE",
	"RDMA_CM_EVENT_REJECTED",
	"RDMA_CM_EVENT_ESTABLISHED",
	"RDMA_CM_EVENT_DISCONNECTED",
	"RDMA_CM_EVENT_DEVICE_REMOVAL",
	"RDMA_CM_EVENT_MULTICAST_JOIN",
	"RDMA_CM_EVENT_MULTICAST_ERROR",
	"RDMA_CM_EVENT_ADDR_CHANGE",
	"RDMA_CM_EVENT_TIMEWAIT_EXIT",
	"RDMA_CM_EVENT_ALT_ROUTE_RESOLVED",
	"RDMA_CM_EVENT_ALT_ROUTE_ERROR",
	"RDMA_CM_EVENT_LOAD_ALT_PATH",
	"RDMA_CM_EVENT_ALT_PATH_LOADED",
};

enum crdma_state {
	WAITING,
	CONNECT_REQUEST,
	CONNECTED,
	ROUTE_RESOLVED,
	ADDR_RESOLVED,
	FRMR_COMPLETE,
	RDMA_WRITE_COMPLETE,
	RDMA_SEND_COMPLETE,
	RDMA_RECV_COMPLETE,
	RDMA_READ_COMPLETE,
	DISCONNECT,
	ERROR
};

struct crdma_cb {
	u8 addr[4];
	uint16_t port;
	struct rdma_cm_id *cmid, *child;
	struct ib_cq *cq;
	struct ib_pd *pd;
	struct ib_qp *qp;
	struct ib_mr *mr;
	struct ib_mr *frmr;
	struct ib_fast_reg_page_list *frpl;
	struct ib_send_wr send_wr;
	struct ib_send_wr fast_wr;
	struct ib_recv_wr recv_wr;
#if  LINUX_VERSION_CODE <  KERNEL_VERSION(5, 0, 1)
	struct ib_send_wr *bad_send;
	struct ib_recv_wr *bad_recv;
#else
	const struct ib_send_wr *bad_send;
	const struct ib_recv_wr *bad_recv;
#endif
	struct ib_sge test_sge[2];	// 0 is recieve, 1 is send
	void *buffs[3];				// 0 is recieve, 1 is send, 2 is the fast reg region
	u64 dma[3];
	//unsigned long times[2];
	uint32_t remote_rkey;
	uint64_t remote_addr;
	uint32_t remote_len;
	wait_queue_head_t wqueue;
};

enum crdma_state state;

static int crdma_cma_event_handler(struct rdma_cm_id *cma_id, struct rdma_cm_event *event) {

	struct crdma_cb *cb = cma_id->context;
	pr_info("%s on cma_id %p\n", cma_event[event->event], cma_id);
	switch (event->event) {
	case RDMA_CM_EVENT_ADDR_RESOLVED:
		state = ADDR_RESOLVED;
		if (rdma_resolve_route(cma_id, 2000)) {
			pr_err( "Failed to resolve route!!!\n");
			wake_up_interruptible(&cb->wqueue);
		}
		break;

	case RDMA_CM_EVENT_ROUTE_RESOLVED:
		state = ROUTE_RESOLVED;
		wake_up_interruptible(&cb->wqueue);
		break;

	case RDMA_CM_EVENT_CONNECT_REQUEST:
		cb->child = cma_id;
		state = CONNECT_REQUEST;
		wake_up_interruptible(&cb->wqueue);
		break;

	case RDMA_CM_EVENT_ESTABLISHED:
		state = CONNECTED;
		wake_up_interruptible(&cb->wqueue);
		break;

	case RDMA_CM_EVENT_ADDR_ERROR:
	case RDMA_CM_EVENT_ROUTE_ERROR:
	case RDMA_CM_EVENT_CONNECT_ERROR:
	case RDMA_CM_EVENT_UNREACHABLE:
	case RDMA_CM_EVENT_REJECTED:
		pr_err( "Event %d had error %d\n", event->event,
		       event->status);
		break;

	case RDMA_CM_EVENT_DISCONNECTED:
		pr_err( "Disconnecting...\n");
		state = DISCONNECT;
		wake_up_interruptible(&cb->wqueue);
		break;

	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		pr_err( "Device removal detected!!!\n");
		break;

	default:
		pr_err( "Unsupported operation!\n");
		break;
	}
	return 0;
}

static void crdma_cq_event_handler(struct ib_cq *cq, void *ctx) {
	struct crdma_cb *cb = ctx;
	struct ib_wc wc;
#if  LINUX_VERSION_CODE <  KERNEL_VERSION(5, 0, 1)
	struct ib_recv_wr *bad_wr;
#else
	const struct ib_recv_wr *bad_wr;
#endif
	int ret;
	ib_req_notify_cq(cb->cq, IB_CQ_NEXT_COMP);
	while ((ret = ib_poll_cq(cb->cq, 1, &wc)) == 1) {
		if (wc.status) {
			if (wc.status == IB_WC_WR_FLUSH_ERR) {
				pr_info("cq flushed\n");
				continue;
			} else {
				pr_err("cq completion failed with "
				       "wr_id %Lx status %d opcode %d vender_err %x\n",
					wc.wr_id, wc.status, wc.opcode, wc.vendor_err);
				goto error;
			}
		}

		switch (wc.opcode) {
		case IB_WC_SEND:
			pr_info("Send completion\n");
			//state = RDMA_SEND_COMPLETE;
			//wake_up_interruptible(&cb->wqueue);
			break;

		case IB_WC_RDMA_WRITE:
			pr_info("Rdma write completion\n");
			state = RDMA_WRITE_COMPLETE;
			wake_up_interruptible(&cb->wqueue);
			break;

		case IB_WC_RDMA_READ:
			pr_info("Rdma read completion\n");
			state = RDMA_READ_COMPLETE;
			wake_up_interruptible(&cb->wqueue);
			break;

		case IB_WC_RECV:
			pr_info("Recv completion\n");
			state = RDMA_RECV_COMPLETE;
			ret = ib_post_recv(cb->qp, &cb->recv_wr, &bad_wr);
			if (ret) {
				pr_err("Post recv error: %d\n",
				       ret);
				goto error;
			}
			wake_up_interruptible(&cb->wqueue);
			break;
#if TEST_FAST_REG_MR
		case IB_WC_FAST_REG_MR:
			pr_info("Fast reg mr complete\n");
			state = FRMR_COMPLETE;
			wake_up_interruptible(&cb->wqueue);
			break;
#endif
		default:
			pr_err("%s:%d Unexpected opcode %d, Shutting down\n",
			       __func__, __LINE__, wc.opcode);
			goto error;
		}
	}
	if (ret) {
		pr_err("poll error %d\n", ret);
		goto error;
	}
	return;
	error:
	state = ERROR;
	wake_up_interruptible(&cb->wqueue);
}

static int crdma_connect(struct crdma_cb *cb){
	struct rdma_conn_param conn_param;
	int ret;

	memset(&conn_param, 0, sizeof conn_param);
	conn_param.responder_resources = 5;
	conn_param.initiator_depth = 5;
	conn_param.retry_count = 10;

	ret = rdma_connect(cb->cmid, &conn_param);
	if (ret) {
		pr_err( "rdma_connect error %d\n", ret);
		return ret;
	}

	wait_event_interruptible(cb->wqueue, state == CONNECTED);
	if (state == ERROR) {
		pr_err( "wait for CONNECTED state %d\n", state);
		return -1;
	}

	pr_info("Successfully allocated resources!!!\n");
	return 0;
}

static int crdma_accept(struct crdma_cb *cb){
	struct rdma_conn_param conn_param;
	int ret;

	pr_info("Accepting connection request\n");

	memset(&conn_param, 0, sizeof conn_param);
	conn_param.responder_resources = 5;
	conn_param.initiator_depth = 5;

	ret = rdma_accept(cb->child, &conn_param);
	if (ret) {
		pr_err( "rdma_accept error: %d\n", ret);
		return ret;
	}
	wait_event_interruptible(cb->wqueue, state >= CONNECTED);
	if (state == ERROR) {
		pr_err( "wait for CONNECTED state %d\n", state);
		return -1;
	}
	return 0;
}

static int crdma_create_qp(struct crdma_cb *cb) {
	struct ib_qp_init_attr init_attr;
	int ret;
	memset(&init_attr, 0, sizeof(init_attr));
	init_attr.cap.max_send_wr = 8;
	init_attr.cap.max_recv_wr = 2;
	init_attr.cap.max_recv_sge = 2;
	init_attr.cap.max_send_sge = 2;
	init_attr.qp_type = IB_QPT_RC;
	init_attr.send_cq = cb->cq;
	init_attr.recv_cq = cb->cq;
	init_attr.sq_sig_type = IB_SIGNAL_REQ_WR;
	if (!mode) {
		ret = rdma_create_qp(cb->child, cb->pd, &init_attr);
		if (!ret)
			cb->qp = cb->child->qp;
	} else {
		ret = rdma_create_qp(cb->cmid, cb->pd, &init_attr);
		if (!ret)
			cb->qp = cb->cmid->qp;
	}
	return ret;
}

static void crdma_free_qp(struct crdma_cb *cb) {
	ib_destroy_qp(cb->qp);
	ib_destroy_cq(cb->cq);
	ib_dealloc_pd(cb->pd);
}

static int crdma_make_qp(struct crdma_cb *cb, struct rdma_cm_id *cmid) {
	int ret;
        struct ib_cq_init_attr attr = {0};
	cb->pd = ib_alloc_pd(cmid->device,0);
	if (IS_ERR(cb->pd)) {
		pr_err( "Failed to allocate pd!!!\n");
		return PTR_ERR(cb->pd);
	}
	pr_info("Created pd %p\n", cb->pd);

#if 0
	cb->cq = ib_create_cq(cmid->device, crdma_cq_event_handler, NULL,
	                      cb, 4, 0);
#else
        cb->cq = ib_create_cq(cmid->device, crdma_cq_event_handler, NULL,
                              cb, &attr);
#endif
	if (IS_ERR(cb->cq)) {
		pr_err( "Failed to create cq!!!\n");
		ret = PTR_ERR(cb->cq);
		goto err1;
	}
	ret = ib_req_notify_cq(cb->cq, IB_CQ_NEXT_COMP);
	if(ret){
		pr_err( "Failed to create cq!!!\n");
		goto err2;
	}
	pr_info("Created cq %p\n", cb->cq);
	ret = crdma_create_qp(cb);
	if (ret) {
		pr_err( "crdma_create_qp failed: %d\n", ret);
		goto err2;
	}
	pr_info("Created qp %p\n", cb->qp);
	return 0;
err2:
	ib_destroy_cq(cb->cq);
err1:
	ib_dealloc_pd(cb->pd);
	return ret;
}

static int crdma_bind(struct crdma_cb *cb) {
	int ret = 0;
	struct sockaddr_in * sock = kzalloc(sizeof(*sock), GFP_KERNEL);
	sock->sin_family = AF_INET;
	sock->sin_port = 1234;
	memcpy((void *)&sock->sin_addr.s_addr, cb->addr, 4);
	pr_info("Socket successfully created!!!\n");
	if (mode) {
		if ((ret = rdma_resolve_addr(cb->cmid, NULL, (struct sockaddr *)sock, 2000))) {
			pr_err( "Failed to resolve address!!!\n");
			goto done;
		}
		wait_event_interruptible(cb->wqueue, state == ROUTE_RESOLVED || state == ADDR_RESOLVED);
		if ((ret = (state == ADDR_RESOLVED))) {
			pr_err( "Failed to resolve address/route!!!\n");
			goto done;
		}
	} else {
		if ((ret = rdma_bind_addr(cb->cmid, (struct sockaddr *)sock))) {
			pr_err( "rdma_bind_addr failed!!!\n");
			goto done;
		}
		pr_info("Address successfully bound to the id!!!\n");
	}
done:
	kfree(sock);
	return ret;
}

static int crdma_fr(struct crdma_cb *cb){
	int ret=0,i;
#if TEST_FAST_REG_MR
	cb->frpl = ib_alloc_fast_reg_page_list(cb->pd->device, 8);
	pr_info("Page list: %p\nMax page list length: %u\n", cb->frpl, cb->frpl->max_page_list_len);
	if(IS_ERR(cb->frpl)){
		pr_err("Failed to allocate the page list!!!\n");
		return PTR_ERR(cb->frpl);
	}
	cb->frmr = ib_alloc_fast_reg_mr(cb->pd, 
					cb->frpl->max_page_list_len);
	if (IS_ERR(cb->frmr)) {
		pr_err("fast_reg_mr failed\n");
		ret = PTR_ERR(cb->frmr);
		goto error0;
	}
#endif
	pr_info("Fast_reg rkey: %lu\n", (long unsigned)cb->frmr->rkey);
	cb->buffs[2] = kmalloc(8*4096, GFP_KERNEL);
	cb->dma[2] = ib_dma_map_single(cb->pd->device, cb->buffs[2], 8*4096, DMA_BIDIRECTIONAL);
	if(ib_dma_mapping_error(cb->pd->device, cb->dma[2])){
		pr_err("Error mapping fast reg buffer\n");
		ret = 1;
		goto error1;
	}
#if TEST_FAST_REG_MR
	for(i=0; i<7; i++){
		cb->frpl->page_list[i] = (cb->dma[2] + i*4096) & PAGE_MASK;
	}
	pr_info("Page mask: %llx\n", PAGE_MASK);
	cb->send_wr.opcode = IB_WR_FAST_REG_MR;
	cb->send_wr.num_sge = 0;
	cb->send_wr.sg_list =NULL;
	cb->send_wr.next = NULL;
	cb->send_wr.send_flags = IB_SEND_SIGNALED;
	cb->send_wr.wr.fast_reg.access_flags = 
		IB_ACCESS_LOCAL_WRITE | IB_ACCESS_REMOTE_READ | IB_ACCESS_REMOTE_WRITE;
	cb->send_wr.wr.fast_reg.page_list = cb->frpl;
	cb->send_wr.wr.fast_reg.rkey = cb->frmr->rkey;
	cb->send_wr.wr.fast_reg.page_shift = PAGE_SHIFT;
	cb->send_wr.wr.fast_reg.page_list_len = 8;
	cb->send_wr.wr.fast_reg.iova_start=cb->dma[2];
	cb->send_wr.wr.fast_reg.length = 8*4096;
#endif
	//cb->times[0] = jiffies;
	state = WAITING;
	if(ib_post_send(cb->qp, &cb->send_wr, &cb->bad_send)){
		pr_err("Failed to post work request to send queue!!!\n");
		goto error2;
	}
	if(wait_event_interruptible(cb->wqueue, state >=FRMR_COMPLETE)){
		pr_info("Interrupted\n");
	}
	pr_info("State after wakikng: %u\n", state);
	if(state >= DISCONNECT)
		goto error2;
	cb->dma[0] = cb->test_sge[0].addr;
	cb->dma[1] = cb->test_sge[1].addr;
	state = WAITING;
	cb->test_sge[0].lkey = cb->frmr->lkey;
	cb->test_sge[1].lkey = cb->frmr->lkey;
	cb->test_sge[0].addr = cb->dma[2];
	cb->test_sge[1].addr = cb->dma[2] + 4096;
	cb->test_sge[0].length = 4*1024;
	cb->test_sge[1].length = 4*1024;
	cb->recv_wr.next = NULL;
	cb->recv_wr.num_sge = 1;
	cb->recv_wr.sg_list = cb->test_sge;
	memcpy(cb->buffs[2] + 4096, &cb->dma[2], sizeof(cb->dma[2]));
	memcpy(cb->buffs[2] + 4096+sizeof(cb->dma[2]), &cb->frmr->rkey, sizeof(cb->frmr->rkey));
	pr_info("Address: %llx\n",(long long unsigned)cb->dma[2]);
	cb->send_wr.next = NULL;
	cb->send_wr.sg_list = &cb->test_sge[1];
	cb->send_wr.num_sge = 1;
	cb->send_wr.opcode = IB_WR_SEND;
	cb->send_wr.send_flags = IB_SEND_SIGNALED;
	if(ib_post_send(cb->qp, &cb->send_wr, &cb->bad_send)){
		pr_err("Failed to post work request to send queue!!!\n");
		goto error2;
	}
	if(wait_event_interruptible(cb->wqueue, state >=RDMA_SEND_COMPLETE)){
		pr_info("Interrupted\n");
	}
	//return 0;
	cb->test_sge[0].addr = cb->dma[0];
	cb->test_sge[1].addr = cb->dma[1];
error2:
	ib_dma_unmap_single(cb->pd->device, cb->dma[2], 8*4*1024, DMA_BIDIRECTIONAL);
error1:
	ib_dereg_mr(cb->frmr);
error0:
#if TEST_FAST_REG_MR
	ib_free_fast_reg_page_list(cb->frpl);
#endif
	return ret;

}
struct ib_mr *ib_get_dma_mr(struct ib_pd *pd, int mr_access_flags)
{
	struct ib_mr *mr;
	int err;

	err = ib_check_mr_access(mr_access_flags);
	if (err)
		return ERR_PTR(err);

#if  LINUX_VERSION_CODE <  KERNEL_VERSION(5, 0, 1)
        mr = pd->device->get_dma_mr(pd, mr_access_flags);
#else
	mr =  pd->device->ops.get_dma_mr(pd, mr_access_flags);
#endif
	if (!IS_ERR(mr)) {
		mr->device  = pd->device;
		mr->pd      = pd;
		mr->uobject = NULL;
		//atomic_inc(&pd->usecnt);
		//atomic_set(&mr->usecnt, 0);
	}

	return mr;
}
static int crdma_mr(struct crdma_cb *cb){
	int ret = 0;
	cb->buffs[0] = kmalloc(4*1024, GFP_KERNEL);	
	cb->buffs[1] = kmalloc(4*1024, GFP_KERNEL);

	cb->test_sge[0].addr = 
		ib_dma_map_single(cb->pd->device, cb->buffs[0], 4*1024, DMA_BIDIRECTIONAL);
	cb->test_sge[0].length = 4*1024;

	cb->test_sge[1].addr = 
		ib_dma_map_single(cb->pd->device, cb->buffs[1], 4*1024, DMA_BIDIRECTIONAL);
	cb->test_sge[1].length = 4*1024;

	if(ib_dma_mapping_error(cb->pd->device, cb->test_sge[0].addr)){
		pr_err("mapping error for buffs[0]\n");
		ret = 1;
		goto error1;
	}
	if(ib_dma_mapping_error(cb->pd->device, cb->test_sge[1].addr)){
		pr_err("mapping error for buffs[1]\n");
		ret = 1;
		goto error1;
	}
//#if TEST_FAST_REG_MR
#if 1
	cb->mr = ib_get_dma_mr(cb->pd, 
		IB_ACCESS_LOCAL_WRITE|IB_ACCESS_REMOTE_READ|IB_ACCESS_REMOTE_WRITE);
#else
        cb->mr = ib_alloc_mr(rdma_d->pd, IB_MR_TYPE_MEM_REG, PAGE_SIZE);
#endif
	if (IS_ERR(cb->mr)) {
		pr_err("get_dma_mr failed\n");
		ret = PTR_ERR(cb->mr);
		goto error1;
	}
	pr_info("Dma rkey: %llu\n", (long long unsigned)cb->mr->rkey);
	cb->test_sge[0].lkey = cb->mr->lkey;
	cb->test_sge[1].lkey = cb->mr->lkey;
	cb->recv_wr.next = NULL;
	cb->recv_wr.num_sge = 1;
	cb->recv_wr.sg_list = cb->test_sge;

	cb->send_wr.next = NULL;
	cb->send_wr.sg_list = &cb->test_sge[1];
	cb->send_wr.num_sge = 1;
	cb->send_wr.opcode = IB_WR_SEND;
	cb->send_wr.send_flags = IB_SEND_SIGNALED;
	return ret;
error1:
	ib_dma_unmap_single(cb->pd->device, cb->test_sge[0].addr, 4*1024, DMA_BIDIRECTIONAL);
	ib_dma_unmap_single(cb->pd->device, cb->test_sge[1].addr, 4*1024, DMA_BIDIRECTIONAL);
	kfree(cb->buffs[0]);
	kfree(cb->buffs[1]);
	return ret;

}
static void crdma_free_mr(struct crdma_cb *cb){
	ib_dma_unmap_single(cb->pd->device, cb->test_sge[0].addr, 4*1024, DMA_BIDIRECTIONAL);
	ib_dma_unmap_single(cb->pd->device, cb->test_sge[1].addr, 4*1024, DMA_BIDIRECTIONAL);
	kfree(cb->buffs[0]);
	kfree(cb->buffs[1]);
	ib_dereg_mr(cb->mr);
	return;
	/*ib_dma_unmap_single(cb->pd->device, cb->test_sge[0].addr, 4*1024, DMA_BIDIRECTIONAL);
	ib_dma_unmap_single(cb->pd->device, cb->test_sge[1].addr, 4*1024, DMA_BIDIRECTIONAL);
	ib_dereg_mr(cb->frmr);
	ib_free_fast_reg_page_list(cb->frpl);*/
}
/*static void crdma_dev_info(struct crdma_cb *cb){
	struct ib_device_attr attr;
	ib_query_device(cb->pd->device,&attr);
	pr_info("Max mr size: %llu\nMax frpl length: %u\n",attr.max_mr_size, attr.max_fast_reg_page_list_len);
}*/
static void server(struct crdma_cb *cb) {
	if (crdma_bind(cb))
		return;

	if (rdma_listen(cb->cmid, 3)) {
		pr_err( "rdma_listen failed!!!\n");
		return;
	}
	pr_info("Listening for a connection...\n");

	wait_event_interruptible(cb->wqueue, state == CONNECT_REQUEST);

	if (crdma_make_qp(cb, cb->child)) {
		pr_err( "Failed to allocate resources!!!\n");
		goto error0;
	}

	if(crdma_mr(cb)){
		pr_err("Failed to setup dma mr\n");
		goto error0;
	}

	if(ib_post_recv(cb->qp, &cb->recv_wr, &cb->bad_recv)){
		pr_err("Failed to post work request to receive queue!!!\n");
		goto error1;
	}

	crdma_accept(cb);
	memcpy(cb->buffs[1],&cb->mr->rkey, sizeof(cb->mr->rkey));
	if(ib_post_send(cb->qp, &cb->send_wr, &cb->bad_send)){
		pr_err("Failed to post work request to send queue!!!\n");
		goto error1;
	}
	pr_info("State before wakikng: %u\n", state);
	if(wait_event_interruptible(cb->wqueue, state >= RDMA_RECV_COMPLETE)){
		pr_info("Interrupted\n");
		goto error1;
	}
	pr_info("State after wakikng: %u\n", state);
	cb->remote_rkey = *(u32 *)(cb->buffs[0]);
	pr_info("Remote rkey: %lu\n", (long unsigned)cb->remote_rkey);

	if(crdma_fr(cb)){
		pr_err("Error trying to run fast reg test\n");
		goto error1;
	}
	if(wait_event_interruptible(cb->wqueue, state ==DISCONNECT)){
		pr_info("Interrupted\n");
	}
error1:
	crdma_free_mr(cb);
error0:
	crdma_free_qp(cb);
	rdma_destroy_id(cb->child);
	return;
}

static void client(struct crdma_cb *cb) {
	if (crdma_bind(cb))
		return;

	if (cb->cmid->device == NULL) {
		pr_err(KERN_ERR "device is NULL\n");
		return;
																		
	}
	if (crdma_make_qp(cb, cb->cmid)) {
		pr_err( "Failed to allocate resources!!!\n");
		goto error0;
	}

	if(crdma_mr(cb)){
		pr_err("Failed to setup dma mr\n");
		goto error0;
	}

	if(ib_post_recv(cb->qp, &cb->recv_wr, &cb->bad_recv)){
		pr_err("Failed to post work request to receive queue!!!\n");
		goto error1;
	}

	if(crdma_connect(cb)){
		pr_err( "Falied to connect client!!!\n");
		goto error1;
	}
	memcpy(cb->buffs[1],&cb->mr->rkey, sizeof(cb->mr->rkey));
	if(ib_post_send(cb->qp, &cb->send_wr, &cb->bad_send)){
		pr_err("Failed to post work request to send queue!!!\n");
		goto error1;
	}
	if(wait_event_interruptible(cb->wqueue, state >= RDMA_RECV_COMPLETE)){
		pr_info("Interrupted\n");
	}
	state = WAITING;
	cb->remote_rkey = *(u32 *)(cb->buffs[0]);
	pr_info("Remote rkey: %lu\n", (long unsigned)cb->remote_rkey);
	if(wait_event_interruptible(cb->wqueue, state >= RDMA_RECV_COMPLETE)){
		pr_info("Interrupted\n");
	}
	cb->remote_rkey = *(u32 *)(cb->buffs[0] + sizeof(cb->dma[2]));
	cb->dma[2] = *(u64 *)(cb->buffs[0]);
	pr_info("Remote rkey: %lu\nAddress: %llx\n", (long unsigned)cb->remote_rkey, (long long unsigned)cb->dma[2]);

	cb->send_wr.opcode = IB_WR_RDMA_WRITE;
#if TEST_FAST_REG_MR
	cb->send_wr.wr.rdma.rkey = cb->remote_rkey;
	cb->send_wr.wr.rdma.remote_addr = cb->dma[2];
#endif
	if(ib_post_send(cb->qp, &cb->send_wr, &cb->bad_send)){
		pr_err("Failed to post work request to send queue!!!\n");
		goto error1;
	}
	if(wait_event_interruptible(cb->wqueue, state ==RDMA_WRITE_COMPLETE)){
		pr_info("Interrupted\n");
	}
error1:
	crdma_free_mr(cb);
error0:
	crdma_free_qp(cb);
	return;
}

static int __init initialize(void) {
	struct crdma_cb *cb;
	pr_info("CRDMA initialize function\n");

	if (mode == -1) {
		pr_err( "ERROR: Mode not specified, 0 for server and 1 for client\n");
		return -1;
	}

	cb = kzalloc(sizeof(*cb), GFP_KERNEL);
	in4_pton("10.11.11.251", -1, cb->addr, -1, NULL);
	init_waitqueue_head(&cb->wqueue);
	cb->cmid = rdma_create_id(&init_net,crdma_cma_event_handler, cb, RDMA_PS_TCP, IB_QPT_RC);
	if (IS_ERR(cb->cmid)) {
		pr_err( "rdma_create_id error %ld\n", PTR_ERR(cb->cmid));
		return -1;
	}
	pr_info("Created cm_id %p\n",  cb->cmid);

	if (mode)
		client(cb);
	else
		server(cb);

	rdma_destroy_id(cb->cmid);
	kfree(cb);
	pr_info("Returning a nonzero number to avoid having to remove the module (work is already done)\n");
	return -1;
}

static void __exit escape(void) {
	pr_info("CRDMA exit function\n\n");
}

module_init(initialize);
module_exit(escape);
