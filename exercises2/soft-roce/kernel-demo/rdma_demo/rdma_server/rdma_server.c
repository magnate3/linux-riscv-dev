#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/inet.h>
#include <linux/slab.h>
#include <linux/mutex.h>
#include <linux/gfp.h>

#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <rdma/rw.h>

enum rdma_struct_flags_bit {
	ADDR_RESOLVED = 0,
	ROUTE_RESOLVED,
	REMOVING = 24,
};

struct rkey_msg {
	u64 remote_key;
	u64 remote_addr;
};

#define RECV_CMD_SIZE	(4)
struct rdma_recv_cmd {
	struct ib_recv_wr recv_wr;
	struct ib_cqe cqe;
};

struct rdma_connection {
	unsigned long state;
	int send_mr_finished;
	int recv_mr_finished;
	struct rdma_cm_id *cm_id;

	struct ib_pd *pd;
	struct ib_cq *cq;
	struct ib_mr *mr;

	char *recv_buf;
	dma_addr_t recv_dma_addr;	// dma addr of recv_buf
	struct ib_sge recv_sgl;
	struct ib_recv_wr recv_mr_wr;
	struct ib_cqe recv_mr_cqe;
	int recv_cmd_pos;
	struct rdma_recv_cmd recv_cmd[RECV_CMD_SIZE];

	struct ib_sge send_sgl;
	struct ib_send_wr sq_wr;
	struct ib_cqe sq_cqe;
	char *send_buf;
	dma_addr_t send_dma_addr;	// dma addr of send_buf

	struct ib_sge rdma_sgl;
	struct ib_rdma_wr rdma_sq_wr;
	struct ib_cqe rdma_sq_cqe;
	char *rdma_buf;
	dma_addr_t rdma_dma_addr;	// dma addr of rdma_buf

	struct ib_reg_wr reg_mr_wr;
	struct ib_cqe reg_mr_cqe;
	u64 remote_key;
	u64 remote_addr;

	struct list_head list;
	struct work_struct disconnect_work;
};

struct rdma_struct {
	unsigned long flags;
	struct sockaddr_storage sin;

	struct rdma_cm_id *cm_id;
	struct rdma_cm_event *event;
	struct rdma_listener *listener;

	struct ib_mr *reg_mr;

	wait_queue_head_t wait;

	struct mutex connection_lock;
	struct list_head connection_list;
};

struct rdma_struct rdma_d;
struct rdma_connection *rdma_cc = NULL;

static int do_alloc_qp(struct rdma_cm_id *cm_id, struct ib_pd *pd, struct ib_cq *cq);
static struct ib_cq *do_alloc_cq(struct rdma_cm_id *cm_id);
static void rdma_recv_done(struct ib_cq *cq, struct ib_wc *wc);
static void rdma_send_done(struct ib_cq *cq, struct ib_wc *wc);
static void rdma_rdma_send_done(struct ib_cq *cq, struct ib_wc *wc);
static void rdma_reg_mr_done(struct ib_cq *cq, struct ib_wc *wc);
static void rdma_recv_mr_done(struct ib_cq *cq, struct ib_wc *wc);
static int recv_mr(struct rdma_connection *rdma_c);
static int recv_data(struct rdma_connection *rdma_c);

static int send_file_show(struct seq_file *m, void *ignored)
{
	return 0;
}

static int send_file_open(struct inode *inode, struct file *file) {
	return single_open(file, send_file_show, inode->i_private);
}

static ssize_t send_file_write(struct file *file, const char __user *ubuf, size_t cnt, loff_t *ppos)
{
	if (rdma_cc)
		recv_data(rdma_cc);
	return cnt;
}

static int send_file_release(struct inode *inode, struct file *file)
{
	return single_release(inode, file);
}

static const struct file_operations send_file_fops = {
	.owner = THIS_MODULE,
	.open = send_file_open,
	.llseek = seq_lseek,
	.read = seq_read,
	.write = send_file_write,
	.release = send_file_release,
};

static void init_requests(struct rdma_connection *rdma_c)
{
	int i;

	// recv request
	rdma_c->recv_sgl.addr = rdma_c->recv_dma_addr;
	rdma_c->recv_sgl.length = PAGE_SIZE;
	rdma_c->recv_sgl.lkey = rdma_c->pd->local_dma_lkey;

	for (i = 0; i < RECV_CMD_SIZE; i++) {
		rdma_c->recv_cmd[i].recv_wr.sg_list = &rdma_c->recv_sgl;
		rdma_c->recv_cmd[i].recv_wr.num_sge = 1;
		rdma_c->recv_cmd[i].cqe.done = rdma_recv_done;
		rdma_c->recv_cmd[i].recv_wr.wr_cqe = &rdma_c->recv_cmd[i].cqe;
	}

	rdma_c->recv_mr_wr.sg_list = &rdma_c->recv_sgl;
	rdma_c->recv_mr_wr.num_sge = 1;
	rdma_c->recv_mr_cqe.done = rdma_recv_mr_done;
	rdma_c->recv_mr_wr.wr_cqe = &rdma_c->recv_mr_cqe;
	// send request
	rdma_c->send_sgl.addr = rdma_c->send_dma_addr;
	rdma_c->send_sgl.length = PAGE_SIZE;
	rdma_c->send_sgl.lkey = rdma_c->pd->local_dma_lkey;

	rdma_c->sq_wr.opcode = IB_WR_SEND;
	rdma_c->sq_wr.send_flags = IB_SEND_SIGNALED;
	rdma_c->sq_wr.sg_list = &rdma_c->send_sgl;
	rdma_c->sq_wr.num_sge = 1;
	rdma_c->sq_wr.wr_cqe = &rdma_c->sq_cqe;
	rdma_c->sq_cqe.done = &rdma_send_done;

	// rdma request
	rdma_c->rdma_sgl.addr = rdma_c->rdma_dma_addr;
	rdma_c->rdma_sq_wr.wr.send_flags = IB_SEND_SIGNALED;
	rdma_c->rdma_sq_wr.wr.sg_list = &rdma_c->rdma_sgl;
	rdma_c->rdma_sq_wr.wr.num_sge = 1;
	rdma_c->rdma_sq_wr.wr.wr_cqe = &rdma_c->rdma_sq_cqe;
	rdma_c->rdma_sq_cqe.done = &rdma_rdma_send_done;

	// reg mr request
	rdma_c->reg_mr_wr.wr.opcode = IB_WR_REG_MR;
	rdma_c->reg_mr_wr.mr = rdma_c->mr;
	rdma_c->reg_mr_wr.wr.wr_cqe = &rdma_c->reg_mr_cqe;
	rdma_c->reg_mr_cqe.done = &rdma_reg_mr_done;
}

static int prepare_buffer(struct rdma_connection *rdma_c)
{
	rdma_c->recv_buf = (char *)__get_free_page(GFP_KERNEL | GFP_DMA);
	if (IS_ERR(rdma_c->recv_buf)) {
		printk(KERN_ERR "alloc recv_buf failed.\n");
		return -ENOMEM;
	}
	rdma_c->send_buf = (char *)__get_free_page(GFP_KERNEL | GFP_DMA);
	if (IS_ERR(rdma_c->send_buf)) {
		printk(KERN_ERR "alloc send_buf failed.\n");
		goto free_recv_buf;
	}
	memset(rdma_c->recv_buf, 0x0, PAGE_SIZE);
	memset(rdma_c->send_buf, 0x0, PAGE_SIZE);
	rdma_c->recv_dma_addr = ib_dma_map_single(rdma_c->pd->device, rdma_c->recv_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	rdma_c->send_dma_addr = ib_dma_map_single(rdma_c->pd->device, rdma_c->send_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	rdma_c->rdma_buf = ib_dma_alloc_coherent(rdma_c->pd->device, PAGE_SIZE, &rdma_c->rdma_dma_addr, GFP_KERNEL);
	if (!rdma_c->rdma_buf || !rdma_c->send_dma_addr || !rdma_c->recv_dma_addr) {
		printk(KERN_ERR "map dma addr failed\n");
		goto free_dma_addr;
	}

	rdma_c->mr = ib_alloc_mr(rdma_c->pd, IB_MR_TYPE_MEM_REG, PAGE_SIZE);
	if (IS_ERR(rdma_c->mr)) {
		printk(KERN_ERR "alloc mr failed.\n");
		goto free_dma_addr;
	}
	
	init_requests(rdma_c);

	return 0;
free_dma_addr:
	if (rdma_c->recv_dma_addr)
		ib_dma_unmap_single(rdma_c->pd->device, (unsigned long)rdma_c->recv_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_c->send_dma_addr)
		ib_dma_unmap_single(rdma_c->pd->device, (unsigned long)rdma_c->send_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_c->rdma_buf)
		ib_dma_free_coherent(rdma_c->pd->device, PAGE_SIZE, rdma_c->rdma_buf, rdma_c->rdma_dma_addr);
	free_page((unsigned long)rdma_c->send_buf);
free_recv_buf:
	free_page((unsigned long)rdma_c->recv_buf);
	return -ENOMEM;
}

static int add_to_connection_list(struct rdma_cm_id *cm_id, struct ib_pd *pd, struct ib_cq *cq)
{
	struct rdma_connection *_new = kzalloc(sizeof(struct rdma_connection), GFP_KERNEL);
	if (_new == NULL)
		return -ENOMEM;

	_new->cm_id = cm_id;
	_new->pd = pd;
	_new->cq = cq;
	cm_id->context = _new;
	INIT_LIST_HEAD(&_new->list);
	if (prepare_buffer(_new)) {
		kfree(_new);
		return -ENOMEM;
	}
	printk(KERN_ERR "new connection 0x%p\n", _new);
	mutex_lock(&rdma_d.connection_lock);
	list_add_tail(&rdma_d.connection_list, &_new->list);
	mutex_unlock(&rdma_d.connection_lock);

	return 0;
}

static int do_accept(struct rdma_cm_id *cm_id, struct rdma_cm_event *event)
{
	struct rdma_conn_param conn_param;
	int err = 0;
	struct ib_pd *pd = NULL;
	struct ib_cq *cq = NULL;
	struct rdma_connection *rdma_c = NULL;

	// alloc pd
	if (cm_id->device == NULL) {
		printk(KERN_ERR "device is NULL\n");
		return -ENOMEM;
	}
	pd = ib_alloc_pd(cm_id->device, 0);
	if (IS_ERR(pd)) {
		printk(KERN_ERR "alloc pd failed.\n");
		err = PTR_ERR(pd);
		goto failed;;
	}
	printk(KERN_ERR "alloc pd\n");
	// create cq
	cq = do_alloc_cq(cm_id);
	if (IS_ERR(cq)) {
		printk(KERN_ERR " alloc cq failed.\n");
		err = PTR_ERR(cq);
		goto failed;
	}
	printk(KERN_ERR "alloc cq\n");
	// create qp
	err = do_alloc_qp(cm_id, pd, cq);
	if (err) {
		printk(KERN_ERR "alloc qp failed. error=%d\n", err);
		goto failed;
	}
	printk(KERN_ERR "alloc qp\n");
	memset(&conn_param, 0x0, sizeof(struct rdma_conn_param));
	conn_param.responder_resources = 1;
	conn_param.initiator_depth = 1;

	if ((err = add_to_connection_list(cm_id, pd, cq)))
		goto failed;

	rdma_c = cm_id->context;
	err = recv_mr(rdma_c);
	if (err) {
		printk(KERN_ERR "post recv failed.\n");
		goto out;
	}
	err = rdma_accept(cm_id, &conn_param);
	if (err) {
		printk(KERN_ERR "accept failed, error=%d.\n", err);
		// we destroy it when doing rmmod
		goto out;
	}

	return 0;
failed:
	if (cm_id->qp && !IS_ERR(cm_id->qp)) {
		ib_destroy_qp(cm_id->qp);
	}
	if (cq && !IS_ERR(cq)) {
		ib_destroy_cq(cq);
	}
	if (pd && !IS_ERR(pd)) {
		ib_dealloc_pd(pd);
	}
	
out:
	return err;
}

static void destroy_buffer(struct rdma_connection *rdma_c)
{
	if (rdma_c->mr)
		ib_dereg_mr(rdma_c->mr);
	if (rdma_c->recv_dma_addr)
		ib_dma_unmap_single(rdma_c->pd->device, (unsigned long)rdma_c->recv_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_c->send_dma_addr)
		ib_dma_unmap_single(rdma_c->pd->device, (unsigned long)rdma_c->send_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_c->rdma_buf)
		ib_dma_free_coherent(rdma_c->pd->device, PAGE_SIZE, rdma_c->rdma_buf, rdma_c->rdma_dma_addr);
	if (rdma_c->send_buf)
		free_page((unsigned long)rdma_c->send_buf);
	if (rdma_c->recv_buf)
		free_page((unsigned long)rdma_c->recv_buf);
}

static void do_disconnect(struct work_struct *work)
{
	struct rdma_connection *rdma_c = container_of(work, struct rdma_connection, disconnect_work);

	mutex_lock(&rdma_d.connection_lock);
	printk(KERN_ERR "do disconnect 0x%p.\n", rdma_c);
	list_del_init(&rdma_c->list);
	BUG_ON(rdma_c->cm_id == NULL);
	rdma_disconnect(rdma_c->cm_id);
	ib_drain_qp(rdma_c->cm_id->qp);
	destroy_buffer(rdma_c);
	ib_destroy_qp(rdma_c->cm_id->qp);
	rdma_destroy_id(rdma_c->cm_id);
	ib_free_cq(rdma_c->cq);
	ib_dealloc_pd(rdma_c->pd);

	kfree(rdma_c);
	printk(KERN_ERR "do disconnect finished.\n");
	mutex_unlock(&rdma_d.connection_lock);
}

static int send_mr(struct rdma_connection *rdma_c)
{
	const struct ib_send_wr *bad_wr = NULL;
	int ret = 0;
	u8 key = 0;
	struct scatterlist sg = {0};

	ib_update_fast_reg_key(rdma_c->mr, ++key);
	rdma_c->reg_mr_wr.key = rdma_c->mr->rkey;
	rdma_c->reg_mr_wr.access = IB_ACCESS_REMOTE_WRITE | IB_ACCESS_REMOTE_READ | IB_ACCESS_LOCAL_WRITE;
	sg_dma_address(&sg) = rdma_c->recv_dma_addr;
	sg_dma_len(&sg) = PAGE_SIZE;
	printk(KERN_ERR "reg_mr key=%d, dma_addr=0x%llx.\n", rdma_c->reg_mr_wr.key, rdma_c->recv_dma_addr);

	ret = ib_map_mr_sg(rdma_c->mr, &sg, 1, NULL, PAGE_SIZE);
	if (ret < 0 || ret > PAGE_SIZE) {
		printk(KERN_ERR "map_mr_sg failed\n");
		return -1;
	}

	ret = ib_post_send(rdma_c->cm_id->qp, &rdma_c->reg_mr_wr.wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post reg_mr_wr failed\n");
		return -2;
	}

	return 0;
}

static int send_rdma_addr(struct rdma_connection *rdma_c)
{
	const struct ib_send_wr *bad_wr;
	int ret;
	struct rkey_msg *msg;

	msg = (struct rkey_msg *)rdma_c->send_buf;
	msg->remote_key = be64_to_cpu(rdma_c->mr->rkey);
	msg->remote_addr = be64_to_cpu((unsigned long)rdma_c->recv_buf);
	ret = ib_post_send(rdma_c->cm_id->qp, &rdma_c->sq_wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post sq_wr failed\n");
		return -2;
	}

	printk(KERN_ERR "%s(): rkey=%ld, raddr=0x%lx.\n", __func__, (unsigned long)rdma_c->mr->rkey, (unsigned long)rdma_c->recv_buf);
	return 0;
}

static int recv_data(struct rdma_connection *rdma_c)
{
	int ret;
	const struct ib_recv_wr *bad_wr;
	struct rdma_cm_id *cm_id = rdma_c->cm_id;

	printk(KERN_ERR "post recv.\n");
	ret = ib_post_recv(cm_id->qp, &rdma_c->recv_cmd[rdma_c->recv_cmd_pos].recv_wr, &bad_wr);
	rdma_c->recv_cmd_pos = (rdma_c->recv_cmd_pos + 1) % RECV_CMD_SIZE;
	if (ret) {
		printk(KERN_ERR "post recv after sending mr failed.\n");
		return -1;
	}

	return 0;
}

static int recv_mr(struct rdma_connection *rdma_c)
{
	int ret;
	const struct ib_recv_wr *bad_wr;
	struct rdma_cm_id *cm_id = rdma_c->cm_id;

	ret = ib_post_recv(cm_id->qp, &rdma_c->recv_mr_wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post recv after sending mr failed.\n");
		return -1;
	}

	return 0;
}

static void rdma_recv_done(struct ib_cq *cq, struct ib_wc *wc)
{

//	printk(KERN_ERR "enter %s().\n", __func__);
	if (likely(wc->status == IB_WC_SUCCESS)) {
		if (rdma_cc)
			recv_data(rdma_cc);
		printk(KERN_ERR "recv data \"%s\".\n",
				rdma_cc->recv_buf);
	}
//	printk(KERN_ERR "exit %s().\n", __func__);
}

static void rdma_recv_mr_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct rdma_connection *rdma_c = container_of(wc->wr_cqe, struct rdma_connection, recv_mr_cqe);
	struct rkey_msg *msg;

	if (likely(wc->status == IB_WC_SUCCESS)) {
		recv_data(rdma_c);
		if (rdma_c->recv_mr_finished == 0) {
			rdma_c->recv_mr_finished = 1;
			msg = (struct rkey_msg *)rdma_c->recv_buf;
			rdma_c->remote_key = cpu_to_be64(msg->remote_key);
			rdma_c->remote_addr = cpu_to_be64(msg->remote_addr);
			printk(KERN_ERR "recv mr finished, rkey=%lld, raddr=0x%llx.\n", rdma_c->remote_key, rdma_c->remote_addr);

			send_mr(rdma_c);
			send_rdma_addr(rdma_c);
			// only for client uses IB_WR_RDMA_READ to read data from server
			memcpy(rdma_c->recv_buf, "abcdefedcba test", 17);
			printk("recv_buf data=\"%s\"\n", rdma_c->recv_buf);
		} else {
			printk(KERN_ERR "recv_mr() recv data finished.\n");
		}
	}
}


static void rdma_rdma_send_done(struct ib_cq *cq, struct ib_wc *wc)
{
	if (likely(wc->status == IB_WC_SUCCESS)) {
		printk(KERN_ERR "rdma send done\n");
	}
}

static void rdma_send_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_connection *rdma_c = cm_id->context;

	if (likely(wc->status == IB_WC_SUCCESS)) {
		if (rdma_c->send_mr_finished == 0) {
			printk(KERN_ERR "send mr finished.\n");
			rdma_c->send_mr_finished = 1;
		} else {
			printk(KERN_ERR "send data finished.\n");
		}
	}
}

static void rdma_reg_mr_done(struct ib_cq *cq, struct ib_wc *wc)
{
	if (likely(wc->status == IB_WC_SUCCESS)) {
		printk(KERN_ERR "reg_mr done\n");
	}
}

/*
static void rdma_cq_event_handler(struct ib_cq *cq, void *ctx)
{
	int ret;
	struct ib_wc wc;
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_connection *rdma_c =cm_id->context;

	printk(KERN_ERR "enter cq_event_handler.\n");
	if (test_bit(REMOVING, &rdma_c->state))
		goto out;
	while ((ret = ib_poll_cq(cq, 1, &wc)) == 1) {
		printk(KERN_ERR "opcode=0x%x, state=%d.\n", wc.opcode, wc.status);
		if (test_bit(REMOVING, &rdma_c->state))
			break;

		switch (wc.opcode) {
			case IB_WC_SEND:
				if (rdma_c->send_mr_finished == 0) {
					printk(KERN_ERR "send mr finished.\n");
					rdma_c->send_mr_finished = 1;
				} else {
					printk(KERN_ERR "send data finished.\n");
				}
				break;
			case IB_WC_RDMA_WRITE:
				break;
			case IB_WC_RDMA_READ:
				break;
			case IB_WC_REG_MR:
				break;
			case IB_WC_RECV:
				printk(KERN_ERR "IB_WC_RECV.\n");
				break;
			default:
				printk(KERN_ERR "unknow opcode=0x%x.\n", wc.opcode);
				break;
		}
	}
out:
	printk(KERN_ERR "exit cq_event_handler.\n");
}
*/

static int rdma_cm_handler(struct rdma_cm_id *cm_id, struct rdma_cm_event *event) {
	int err = 0;
	struct rdma_connection *pos, *next, *rdma_c;

	switch (event->event) {
		case RDMA_CM_EVENT_CONNECT_REQUEST:
			printk(KERN_ERR "event is connect_request.\n");
			err = do_accept(cm_id, event);
			if (err) {
				printk(KERN_ERR "accept failed.\n");
				break;
			}
			break;
		case RDMA_CM_EVENT_ESTABLISHED:
			printk(KERN_ERR "event is ESTABLISHED.\n");
			rdma_c = cm_id->context;
			rdma_cc = rdma_c;
			break;
		case RDMA_CM_EVENT_DISCONNECTED:
			printk(KERN_ERR "event is DISCONNECTED.\n");
			// we cannot destroy cm_id in current context, it will cause deadlock
			mutex_lock(&rdma_d.connection_lock);
			list_for_each_entry_safe(pos, next, &rdma_d.connection_list, list) {
				if (pos->cm_id == cm_id) {
					INIT_WORK(&pos->disconnect_work, do_disconnect);
					schedule_work(&pos->disconnect_work);
					break;
				}
			}
			mutex_unlock(&rdma_d.connection_lock);
			break;
		default:
			printk(KERN_ERR "event is unrecognized.\n");
			break;
	}

	return err;
}

static void init_rdma_struct(struct rdma_struct *rdma_d)
{
	init_waitqueue_head(&rdma_d->wait);
	INIT_LIST_HEAD(&rdma_d->connection_list);
	mutex_init(&rdma_d->connection_lock);
}

static int do_alloc_qp(struct rdma_cm_id *cm_id, struct ib_pd *pd, struct ib_cq *cq)
{
	struct ib_qp_init_attr qp_attr = {0};

	qp_attr.qp_context = cm_id;
//	qp_attr.event_handler = rdma_qp_event_handler;
	qp_attr.send_cq = cq;
	qp_attr.recv_cq = cq;
	qp_attr.qp_type = IB_QPT_RC;

	qp_attr.cap.max_send_wr = 128;
	qp_attr.cap.max_recv_wr = 128;
	qp_attr.cap.max_send_sge = 1;
	qp_attr.cap.max_recv_sge = 1;
	qp_attr.sq_sig_type = IB_SIGNAL_REQ_WR;

	return rdma_create_qp(cm_id, pd, &qp_attr);
}

static struct ib_cq *do_alloc_cq(struct rdma_cm_id *cm_id)
{
	struct ib_cq_init_attr cq_attr = {0};

	cq_attr.cqe = 128 * 2;
	cq_attr.comp_vector = 0;
//	return ib_create_cq(cm_id->device, rdma_cq_event_handler, NULL, cm_id, &cq_attr);
	return ib_alloc_cq(cm_id->device, cm_id, 128 * 2, 0, IB_POLL_WORKQUEUE);
}

static int __init rdma_init(void) {
	int ret;
	struct sockaddr_in *addr;
	char *ip = "192.168.122.152";
	char _addr[16] = {0};
	int port = 1;

	addr = (struct sockaddr_in *)&rdma_d.sin;
	addr->sin_family = AF_INET;
	addr->sin_port = port;
	in4_pton(ip, -1, _addr, -1, NULL);
	memcpy((void *)&addr->sin_addr.s_addr, _addr, 4);
//	in4_pton(SERVER_IP, -1, rdma_d.addr, -1, NULL); 
//	memcpy((void *)&addr->sin_addr.s_addr, rdma_d.addr, 4);

	init_rdma_struct(&rdma_d);
	rdma_d.cm_id = rdma_create_id(&init_net, rdma_cm_handler, &rdma_d, RDMA_PS_TCP, IB_QPT_RC);
	if (IS_ERR(rdma_d.cm_id)) {
		printk(KERN_ERR "create cm_id failed.\n");
		return 0;
	}

	ret = rdma_bind_addr(rdma_d.cm_id, (struct sockaddr *)addr);
	if (ret < 0) {
		printk(KERN_ERR "bind failed.\n");
		goto destroy_cm_id;
	}

	ret = rdma_listen(rdma_d.cm_id, 10);
	if (ret < 0) {
		printk(KERN_ERR "listen failed.\n");
		goto destroy_cm_id;
	}

	return 0;

destroy_cm_id:
	if (rdma_d.cm_id) {
		rdma_destroy_id(rdma_d.cm_id);
		rdma_d.cm_id = NULL;
	}
	return ret;
}

static void __exit rdma_exit(void)
{
	struct rdma_connection *next, *pos;

	printk(KERN_ERR "enter rdma_exit.\n");
	mutex_lock(&rdma_d.connection_lock);
	list_for_each_entry_safe(pos, next, &rdma_d.connection_list, list) {
/*		list_del_init(&pos->list);
		BUG_ON(pos->cm_id == NULL);
		printk(KERN_ERR "do disconnect 0x%p.\n", pos);
		rdma_disconnect(pos->cm_id);
		ib_drain_qp(pos->cm_id->qp);
		ib_destroy_qp(pos->cm_id->qp);
		printk(KERN_ERR "do destroy cm_id.\n");
		rdma_destroy_id(pos->cm_id);
		ib_destroy_cq(pos->cq);
		ib_dealloc_pd(pos->pd);
		kfree(pos);
*/
		set_bit(REMOVING, &pos->state);
		INIT_WORK(&pos->disconnect_work, do_disconnect);
		schedule_work(&pos->disconnect_work);
	}
	mutex_unlock(&rdma_d.connection_lock);
	flush_scheduled_work();

	printk(KERN_ERR "destroy rdma_d.cm_id\n");
	if (rdma_d.cm_id)
		rdma_destroy_id(rdma_d.cm_id);
}

MODULE_LICENSE("GPL");
module_init(rdma_init);
module_exit(rdma_exit);
