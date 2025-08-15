#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/inet.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/debugfs.h>

#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <rdma/rw.h>

#define BY_SEND_CMD			(1)
#define BY_RDMA_WRITE_CMD	(2)
#define BY_RDMA_READ_CMD	(3)
int send_method = BY_RDMA_READ_CMD;

struct dentry *debugfs_root = NULL;
struct dentry *send_file = NULL;

enum rdma_struct_flags_bit {
	ADDR_RESOLVED = 0,
	ROUTE_RESOLVED,
};

struct rkey_msg {
	u64 remote_key;
	u64 remote_addr;
};

struct rdma_struct {
	unsigned long flags;
	unsigned int error;
	int send_mr_finished;
	int recv_mr_finished;
	struct sockaddr_storage sin;

	struct rdma_cm_id *cm_id;
	struct rdma_cm_event *event;
	struct rdma_listener *listener;

	struct ib_pd *pd;
	struct ib_cq *cq;
	struct ib_mr *mr;

#define BUF_SIZE	256		// 256 * 16 = 4096
	struct ib_sge recv_sgl;
	struct ib_recv_wr rq_wr;
	struct ib_cqe rq_cqe;
	char *recv_buf;
	dma_addr_t recv_dma_addr;	// dma addr of recv_buf

	struct ib_sge send_sgl;
	struct ib_send_wr sq_wr;
	struct ib_cqe sq_cqe;
	char *send_buf;
	dma_addr_t send_dma_addr;	// dma addr of send_buf

	// for rdma write
	struct ib_sge rdma_sgl;
	struct ib_rdma_wr rdma_sq_wr;
	struct ib_cqe rdma_sq_cqe;
	// for rdma read
	struct ib_sge rdma_read_sgl;
	struct ib_rdma_wr rdma_read_wr;
	struct ib_cqe rdma_read_cqe;
	char *rdma_buf;
	dma_addr_t rdma_dma_addr;	// dma addr of rdma_buf

	struct ib_reg_wr reg_mr_wr;
	struct ib_cqe reg_mr_cqe;

	u64 remote_key;
	u64 remote_addr;
	u64 local_key;

	wait_queue_head_t wait;
	struct work_struct send_data_work;
};

struct rdma_struct rdma_d;

static int do_alloc_qp(struct rdma_cm_id *cm_id, struct ib_pd *pd, struct ib_cq *cq);
static struct ib_cq *do_alloc_cq(struct rdma_cm_id *cm_id);
static int send_data(struct rdma_struct *rdma_d);
static int send_data_by_rdma_write_with_imm(struct rdma_struct *rdma_d);
static int read_data_by_rdma_read(struct rdma_struct *rdma_d);
static int send_rdma_addr(struct rdma_struct *rdma_d);
static int send_mr(struct rdma_struct *rdma_d);
static int recv_rkey(struct rdma_struct *rdma_d);

static int send_file_show(struct seq_file *m, void *ignored)
{
	return 0;
}

static int send_file_open(struct inode *inode, struct file *file) {
	return single_open(file, send_file_show, inode->i_private);
}

static ssize_t send_file_write(struct file *file, const char __user *ubuf, size_t cnt, loff_t *ppos)
{
	if (cnt <= 0 || cnt >= 4095) {
		printk(KERN_ERR "data error.\n");
		return -ENOSPC;
	}
	if (send_method == BY_SEND_CMD) {
		memset(rdma_d.send_buf, 0x0, PAGE_SIZE);
		if (copy_from_user(rdma_d.send_buf, ubuf, cnt))
			return -EFAULT;
		rdma_d.send_buf[cnt] = '\0';
		send_data(&rdma_d);
	} else if (send_method == BY_RDMA_WRITE_CMD) {
		memset(rdma_d.rdma_buf, 0x0, PAGE_SIZE);
		if (copy_from_user(rdma_d.rdma_buf, ubuf, cnt))
			return -EFAULT;
		rdma_d.send_buf[cnt] = '\0';
		send_data_by_rdma_write_with_imm(&rdma_d);
	} else if (send_method == BY_RDMA_READ_CMD) {
		// Read data from server

		memset(rdma_d.rdma_buf, 0x0, PAGE_SIZE);
		read_data_by_rdma_read(&rdma_d);
	}

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

static void rdma_recv_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct rkey_msg *msg;
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_struct *rdma_d = cm_id->context;

	if (likely(wc->status == IB_WC_SUCCESS)) {
		if (rdma_d->recv_mr_finished == 0) {
			rdma_d->recv_mr_finished = 1;
			msg = (struct rkey_msg *)rdma_d->recv_buf;
			rdma_d->remote_key = cpu_to_be64(msg->remote_key);
			rdma_d->remote_addr = cpu_to_be64(msg->remote_addr);
			printk(KERN_ERR "recv mr finished, rkey=%lld, addr=0x%llx.\n", rdma_d->remote_key, rdma_d->remote_addr);
		} else
			printk(KERN_ERR "recv data finished.\n");

	}
	return;
}

static void rdma_send_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_struct *rdma_d = cm_id->context;

	if (likely(wc->status == IB_WC_SUCCESS)) {
		if (rdma_d->send_mr_finished == 0) {
			rdma_d->send_mr_finished = 1;
			printk(KERN_ERR "send mr finished.\n");
		} else {
			printk(KERN_ERR "send data finished.\n");
		}
	}
	return;
}

static void rdma_rdma_write_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_struct *rdma_d = cm_id->context;

	if (likely(wc->status == IB_WC_SUCCESS)) {
		printk(KERN_ERR "send rdma data finished.\n");
		printk(KERN_ERR "rdma write from lkey %d, laddr 0x%llx, len %d\n", 
				rdma_d->rdma_sq_wr.wr.sg_list->lkey, (unsigned long long)rdma_d->rdma_sq_wr.wr.sg_list->addr,
				rdma_d->rdma_sq_wr.wr.sg_list->length);
	} else
		printk(KERN_ERR "%s(): status=0x%x.\n", __func__, wc->status);
	return;
}

static void rdma_rdma_read_done(struct ib_cq *cq, struct ib_wc *wc)
{
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_struct *rdma_d = cm_id->context;

	if (likely(wc->status == IB_WC_SUCCESS)) {
		printk(KERN_ERR "read rdma data finished.\n");
		printk(KERN_ERR "data=\"%s\"\n", rdma_d->rdma_buf);
	} else
		printk(KERN_ERR "%s(): status=0x%x.\n", __func__, wc->status);
	return;
}

static void rdma_reg_mr_done(struct ib_cq *cq, struct ib_wc *wc)
{

	if (likely(wc->status == IB_WC_SUCCESS)) {
		printk(KERN_ERR "reg_mr done.\n");
	}
	return;
}

static void init_requests(struct rdma_struct *rdma_d)
{
	// recv request
	rdma_d->recv_sgl.addr = rdma_d->recv_dma_addr;
	rdma_d->recv_sgl.length = PAGE_SIZE;
	rdma_d->recv_sgl.lkey = rdma_d->pd->local_dma_lkey;

	rdma_d->rq_wr.sg_list = &rdma_d->recv_sgl;
	rdma_d->rq_wr.num_sge = 1;
	rdma_d->rq_wr.wr_cqe = &rdma_d->rq_cqe;
	rdma_d->rq_cqe.done = rdma_recv_done;

	// send request
	rdma_d->send_sgl.addr = rdma_d->send_dma_addr;
	rdma_d->send_sgl.length = PAGE_SIZE;
	rdma_d->send_sgl.lkey = rdma_d->pd->local_dma_lkey;

	rdma_d->sq_wr.opcode = IB_WR_SEND;
	rdma_d->sq_wr.send_flags = IB_SEND_SIGNALED;
	rdma_d->sq_wr.sg_list = &rdma_d->send_sgl;
	rdma_d->sq_wr.num_sge = 1;
	rdma_d->sq_wr.wr_cqe = &rdma_d->sq_cqe;
	rdma_d->sq_cqe.done = rdma_send_done;

	// rdma write request
	rdma_d->rdma_sq_wr.wr.opcode = IB_WR_RDMA_WRITE_WITH_IMM;
	rdma_d->rdma_sgl.addr = rdma_d->rdma_dma_addr;
	rdma_d->rdma_sq_wr.wr.send_flags = IB_SEND_SIGNALED;
	rdma_d->rdma_sq_wr.wr.sg_list = &rdma_d->rdma_sgl;
	rdma_d->rdma_sq_wr.wr.num_sge = 1;
	rdma_d->rdma_sq_wr.wr.wr_cqe = &rdma_d->rdma_sq_cqe;
	rdma_d->rdma_sq_cqe.done = rdma_rdma_write_done;

	// rdma read request
//	rdma_d->rdma_read_wr.wr.opcode = IB_WR_RDMA_READ_WITH_INV;
	rdma_d->rdma_read_wr.wr.opcode = IB_WR_RDMA_READ;
	rdma_d->rdma_read_sgl.addr = rdma_d->rdma_dma_addr;
//	rdma_d->rdma_read_wr.wr.send_flags = IB_SEND_SIGNALED;
	rdma_d->rdma_read_wr.wr.sg_list = &rdma_d->rdma_read_sgl;
	rdma_d->rdma_read_wr.wr.num_sge = 1;
	rdma_d->rdma_read_wr.wr.wr_cqe = &rdma_d->rdma_read_cqe;
	rdma_d->rdma_read_cqe.done = rdma_rdma_read_done;

	// reg mr request
	rdma_d->reg_mr_wr.wr.opcode = IB_WR_REG_MR;
	rdma_d->reg_mr_wr.mr = rdma_d->mr;
	rdma_d->reg_mr_wr.wr.wr_cqe = &rdma_d->reg_mr_cqe;
	rdma_d->reg_mr_cqe.done = rdma_reg_mr_done;
}

static int prepare_buffer(struct rdma_struct *rdma_d)
{
	rdma_d->recv_buf = (char *)__get_free_page(GFP_KERNEL | GFP_DMA);
	if (IS_ERR(rdma_d->recv_buf)) {
		printk(KERN_ERR "alloc recv_buf failed.\n");
		return -ENOMEM;
	}
	rdma_d->send_buf = (char *)__get_free_page(GFP_KERNEL | GFP_DMA);
	if (IS_ERR(rdma_d->send_buf)) {
		printk(KERN_ERR "alloc send_buf failed.\n");
		goto free_recv_buf;
	}
	rdma_d->recv_dma_addr = ib_dma_map_single(rdma_d->pd->device, rdma_d->recv_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	rdma_d->send_dma_addr = ib_dma_map_single(rdma_d->pd->device, rdma_d->send_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	rdma_d->rdma_buf = ib_dma_alloc_coherent(rdma_d->pd->device, PAGE_SIZE, &rdma_d->rdma_dma_addr, GFP_KERNEL);
	if (!rdma_d->rdma_buf || !rdma_d->send_dma_addr || !rdma_d->recv_dma_addr) {
		printk(KERN_ERR "map dma addr failed\n");
		goto free_dma_addr;
	}

	rdma_d->mr = ib_alloc_mr(rdma_d->pd, IB_MR_TYPE_MEM_REG, PAGE_SIZE);
	if (IS_ERR(rdma_d->mr)) {
		printk(KERN_ERR "alloc mr failed.\n");
		goto free_dma_addr;
	}

	init_requests(rdma_d);

	return 0;
free_dma_addr:
	if (rdma_d->recv_dma_addr)
		ib_dma_unmap_single(rdma_d->pd->device, (unsigned long)rdma_d->recv_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_d->send_dma_addr)
		ib_dma_unmap_single(rdma_d->pd->device, (unsigned long)rdma_d->send_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_d->rdma_buf)
		ib_dma_free_coherent(rdma_d->pd->device, PAGE_SIZE, rdma_d->rdma_buf, rdma_d->rdma_dma_addr);
	free_page((unsigned long)rdma_d->send_buf);
free_recv_buf:
	free_page((unsigned long)rdma_d->recv_buf);
	return -ENOMEM;
}

static void destroy_buffer(struct rdma_struct *rdma_d)
{
	if (rdma_d->mr)
		ib_dereg_mr(rdma_d->mr);
	if (rdma_d->recv_dma_addr)
		ib_dma_unmap_single(rdma_d->pd->device, (unsigned long)rdma_d->recv_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_d->send_dma_addr)
		ib_dma_unmap_single(rdma_d->pd->device, (unsigned long)rdma_d->send_buf, PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (rdma_d->rdma_buf)
		ib_dma_free_coherent(rdma_d->pd->device, PAGE_SIZE, rdma_d->rdma_buf, rdma_d->rdma_dma_addr);
	if (rdma_d->send_buf)
		free_page((unsigned long)rdma_d->send_buf);
	if (rdma_d->recv_buf)
		free_page((unsigned long)rdma_d->recv_buf);
}

static int send_mr(struct rdma_struct *rdma_d)
{
	const struct ib_send_wr *bad_wr = NULL;
	int ret = 0;
	u8 key = 0;
	struct scatterlist sg = {0};
	
	printk(KERN_ERR "%s()\n", __func__);
	ib_update_fast_reg_key(rdma_d->mr, ++key);
	rdma_d->reg_mr_wr.key = rdma_d->mr->rkey;
	// IB_ACCESS_REMOTE_READ: 远程有读取这段内存的权限(当远端做RDMA_READ时需要)
	// IB_ACCESS_REMOTE_WRITE: 远程有写入这段内存的权限(当远端做RDMA_WRITE时需要)
	// IB_ACCESS_LOCAL_WRITE: RDMA模块有写入这段内存的权限(当做RDMA_READ时需要, 因为RDMA_READ需要将远程数据写入这段内存)
	rdma_d->reg_mr_wr.access = IB_ACCESS_REMOTE_READ | IB_ACCESS_REMOTE_WRITE | IB_ACCESS_LOCAL_WRITE;
	if (send_method == BY_SEND_CMD) {
		sg_dma_address(&sg) = rdma_d->send_dma_addr;
	} else
		sg_dma_address(&sg) = rdma_d->rdma_dma_addr;
	sg_dma_len(&sg) = PAGE_SIZE;

	ret = ib_map_mr_sg(rdma_d->mr, &sg, 1, NULL, PAGE_SIZE);
	if (ret < 0 || ret > PAGE_SIZE) {
		printk(KERN_ERR "map_mr_sg failed\n");
		return -1;
	}

	ret = ib_post_send(rdma_d->cm_id->qp, &rdma_d->reg_mr_wr.wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post reg_mr_wr failed\n");
		return -2;
	}

	rdma_d->local_key = rdma_d->mr->rkey;
	return 0;
}

static int send_data(struct rdma_struct *rdma_d)
{
	const struct ib_send_wr *bad_wr = NULL;
	int ret;

	printk(KERN_ERR "%s()\n", __func__);
	ret = ib_post_send(rdma_d->cm_id->qp, &rdma_d->sq_wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post sq_wr failed\n");
		return -2;
	}
	if (bad_wr != NULL) {
		printk(KERN_ERR "bad_wr is not NULL.");
	}

	return 0;
}

static int send_data_by_rdma_write_with_imm(struct rdma_struct *rdma_d)
{
	const struct ib_send_wr *bad_wr = NULL;
	int ret;

	printk(KERN_ERR "%s()\n", __func__);
	rdma_d->rdma_sq_wr.rkey = rdma_d->remote_key;
	rdma_d->rdma_sq_wr.remote_addr = rdma_d->remote_addr;
	rdma_d->rdma_sgl.lkey = rdma_d->local_key;
	rdma_d->rdma_sq_wr.wr.sg_list->length = PAGE_SIZE;
	rdma_d->rdma_sq_wr.wr.next = NULL;
	ret = ib_post_send(rdma_d->cm_id->qp, &rdma_d->rdma_sq_wr.wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post rdma_wr failed\n");
		return -2;
	}
	if (bad_wr != NULL) {
		printk(KERN_ERR "bad_wr is not NULL.");
	}

	return 0;
}

static int read_data_by_rdma_read(struct rdma_struct *rdma_d)
{
	const struct ib_send_wr *bad_wr = NULL;
	int ret;

	printk(KERN_ERR "%s()\n", __func__);
	rdma_d->rdma_read_wr.rkey = rdma_d->remote_key;
	rdma_d->rdma_read_wr.remote_addr = rdma_d->remote_addr;
	rdma_d->rdma_read_sgl.lkey = rdma_d->local_key;
	rdma_d->rdma_read_wr.wr.sg_list->length = PAGE_SIZE;
	rdma_d->rdma_read_wr.wr.next = NULL;
	printk("RDMA read data from rkey=%lld, raddr=0x%llx.\n", rdma_d->remote_key, rdma_d->remote_addr);
	ret = ib_post_send(rdma_d->cm_id->qp, &rdma_d->rdma_read_wr.wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post rdma_wr failed\n");
		return -2;
	}
	if (bad_wr != NULL) {
		printk(KERN_ERR "bad_wr is not NULL.");
	}

	return 0;
}

static int send_rdma_addr(struct rdma_struct *rdma_d)
{
	const struct ib_send_wr *bad_wr = NULL;
	int ret;
	struct rkey_msg *msg;

	printk(KERN_ERR "%s()\n", __func__);
	msg = (struct rkey_msg *)rdma_d->send_buf;
	msg->remote_key = be64_to_cpu(rdma_d->mr->rkey);
	msg->remote_addr = be64_to_cpu((unsigned long)rdma_d->recv_buf);
	ret = ib_post_send(rdma_d->cm_id->qp, &rdma_d->sq_wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post sq_wr failed\n");
		return -2;
	}

	printk(KERN_ERR "%s(): rkey=%d, raddr=0x%lx", __func__, rdma_d->mr->rkey, (unsigned long)rdma_d->recv_buf);
	return 0;
}

static int recv_rkey(struct rdma_struct *rdma_d)
{
	const struct ib_recv_wr *bad_wr;
	int ret;

	ret = ib_post_recv(rdma_d->cm_id->qp, &rdma_d->rq_wr, &bad_wr);
	if (ret) {
		printk(KERN_ERR "post rkey recv failed.\n");
		return -1;
	}

	return 0;
}

/*
static void rdma_cq_event_handler(struct ib_cq *cq, void *ctx)
{
	int ret;
	struct rkey_msg *msg;
	struct rdma_cm_id *cm_id = cq->cq_context;
	struct rdma_struct *rdma_d = cm_id->context;
	struct ib_wc wc = {0};
	printk(KERN_ERR "enter %s().\n", __func__);

	if (cq != rdma_d->cq) {
		printk(KERN_ERR "cq is diff.\n");
	}
	while ((ret = ib_poll_cq(cq, 1, &wc)) == 1) {
		printk("opcode=0x%x, state=%d.\n", wc.opcode, wc.status);

		switch (wc.opcode) {
			case IB_WC_SEND:
				if (rdma_d->send_mr_finished == 0) {
					rdma_d->send_mr_finished = 1;
					recv_rkey(rdma_d);
					printk(KERN_ERR "send mr finished.\n");
				} else
					printk(KERN_ERR "send data finished.\n");
				break;
			case IB_WC_RDMA_WRITE:
				break;
			case IB_WC_RDMA_READ:
				break;
			case IB_WC_REG_MR:
				printk(KERN_ERR "REG_MR event");
				send_rdma_addr(rdma_d);
				break;
			case IB_WC_RECV:
				if (rdma_d->recv_mr_finished == 0) {
					msg = (struct rkey_msg *)rdma_d->recv_buf;
					rdma_d->remote_key = cpu_to_be64(msg->remote_key);
					rdma_d->remote_addr = cpu_to_be64(msg->remote_addr);
					printk(KERN_ERR "recv mr finished, rkey=%lld, addr=0x%llx.\n", rdma_d->remote_key, rdma_d->remote_addr);
				} else
					printk(KERN_ERR "recv data finished.\n");
				break;
			default:
				printk("unknow opcode=0x%x", wc.opcode);
				break;
		}
	}

	printk(KERN_ERR "exit %s().\n", __func__);
}
*/

static int rdma_cm_handler(struct rdma_cm_id *cm_id, struct rdma_cm_event *event)
{
	int err = 0;
	struct rdma_struct *rdma_d = cm_id->context;
	struct rdma_conn_param conn_param = {0};
	struct ib_pd *pd;
	struct ib_cq *cq;

	if (cm_id != rdma_d->cm_id) {
		printk(KERN_ERR "cm_id is diff.\n");
	}
	switch (event->event) {
		case RDMA_CM_EVENT_ADDR_RESOLVED:
			printk(KERN_ERR "event is ADDR_RESOLVED.\n");
			set_bit(ADDR_RESOLVED, &rdma_d->flags);
			err = rdma_resolve_route(rdma_d->cm_id, 2000);
			if (err) {
				printk(KERN_ERR "resolve route failed.\n");
				rdma_d->error = 1;
				wake_up_interruptible(&rdma_d->wait);
			}
			break;
		case RDMA_CM_EVENT_CONNECT_REQUEST:
			printk(KERN_ERR "event is connect_request.\n");
			break;
		case RDMA_CM_EVENT_ESTABLISHED:
			printk(KERN_ERR "event is ESTABLISHED.\n");
			recv_rkey(rdma_d);
			err = send_mr(rdma_d);
			if (err) {
				printk(KERN_ERR "send mr failed.\n");
				return err;
			}
			err = send_rdma_addr(rdma_d);
			if (err) {
				printk(KERN_ERR "send rdma addr failed.\n");
				return err;
			}
			break;
		case RDMA_CM_EVENT_DISCONNECTED:
			printk(KERN_ERR "event is DISCONNECTED.\n");
			break;
		case RDMA_CM_EVENT_ROUTE_RESOLVED:
			printk(KERN_ERR "event is ROUTE_RESOLVED.\n");
			set_bit(ROUTE_RESOLVED, &rdma_d->flags);
			// alloc pd
			if (cm_id->device == NULL) {
				printk(KERN_ERR "device is NULL\n");
				err = -ENOMEM;
				break;
			}
			pd = ib_alloc_pd(cm_id->device, 0);
			if (IS_ERR(pd)) {
				printk(KERN_ERR "alloc pd failed.\n");
				err = PTR_ERR(pd);
				rdma_d->error = 1;
				break;
			}
			// create cq
			cq = do_alloc_cq(cm_id);
			if (IS_ERR(cq)) {
				printk(KERN_ERR "alloc cq failed.\n");
				rdma_d->error = 1;
				err = PTR_ERR(cq);
				ib_dealloc_pd(pd);
				break;
			}
			// create qp
			err = do_alloc_qp(cm_id, pd, cq);
			if (err < 0) {
				printk(KERN_ERR "alloc qp failed.\n");
				rdma_d->error = 1;
				ib_destroy_cq(cq);
				ib_dealloc_pd(pd);
				break;
			}

			rdma_d->pd = pd;
			rdma_d->cq = cq;
			prepare_buffer(rdma_d);

			conn_param.responder_resources = 1;
			conn_param.initiator_depth = 1;
			conn_param.retry_count = 10;
			printk(KERN_ERR "do connect.\n");
			err = rdma_connect(cm_id, &conn_param);
			if (err < 0) {
				printk(KERN_ERR "connect failed.\n");
			}

			break;
		default:
			printk(KERN_ERR "event is unrecognized(event=0x%x).\n", event->event);
			break;
	}

	return err;
}

static int do_rdma_resolve_addr(struct rdma_struct *rdma_d, struct sockaddr_in *addr)
{
	int ret = rdma_resolve_addr(rdma_d->cm_id, NULL, (struct sockaddr *)addr, 2000);
	if (ret < 0) {
		printk(KERN_ERR "resolve failed.\n");
		return ret;
	}

	return ret;
}

static void init_rdma_struct(struct rdma_struct *rdma_d)
{
	rdma_d->send_mr_finished = 0;
	rdma_d->recv_mr_finished = 0;
	init_waitqueue_head(&rdma_d->wait);
}

static int do_alloc_qp(struct rdma_cm_id *cm_id, struct ib_pd *pd, struct ib_cq *cq)
{
	struct ib_qp_init_attr qp_attr = {0};

	qp_attr.send_cq = cq;
	qp_attr.recv_cq = cq;
	qp_attr.qp_type = IB_QPT_RC;

	qp_attr.cap.max_send_wr = 128;
	qp_attr.cap.max_recv_wr = 128;
	qp_attr.cap.max_send_sge = 1;
	qp_attr.cap.max_recv_sge = 1;

	return rdma_create_qp(cm_id, pd, &qp_attr);
}

static struct ib_cq *do_alloc_cq(struct rdma_cm_id *cm_id)
{
	struct ib_cq_init_attr cq_attr = {0};

	cq_attr.cqe = 128 * 2;
	cq_attr.comp_vector = 0;
	return ib_alloc_cq(cm_id->device, cm_id, 128 * 2, 0, IB_POLL_WORKQUEUE);
//	return ib_create_cq(cm_id->device, rdma_cq_event_handler, NULL, cm_id, &cq_attr);
}

static void debugfs_cleanup(void)
{
	debugfs_remove(send_file);
	send_file = NULL;
	debugfs_remove(debugfs_root);
	debugfs_root = NULL;
}

static void __init debugfs_init(void)
{
	struct dentry *dentry;

	dentry = debugfs_create_dir("rdma_demo", NULL);
	debugfs_root = dentry;

	send_file = debugfs_create_file("send", 0600, debugfs_root, NULL, &send_file_fops);
}

static int __init rdma_init(void)
{
	int ret = 0;
	struct sockaddr_in *addr;
	char *s_ip = "192.168.122.152";
	char _addr[16] = {0};
	int port = 1;

	addr = (struct sockaddr_in *)&rdma_d.sin;
	addr->sin_family = AF_INET;
	addr->sin_port = port;
	in4_pton(s_ip, -1, _addr, -1, NULL);
	memcpy((void *)&addr->sin_addr.s_addr, _addr, 4);

	init_rdma_struct(&rdma_d);
	rdma_d.cm_id = rdma_create_id(&init_net, rdma_cm_handler, &rdma_d, RDMA_PS_TCP, IB_QPT_RC);
	if (IS_ERR(rdma_d.cm_id)) {
		printk(KERN_ERR "create cm_id failed.\n");
		return 0;
	}

	// waiting RDMA_CM_EVENT_ROUTE_RESOLVED;
	ret = do_rdma_resolve_addr(&rdma_d, addr);
	if (ret < 0)
		goto destroy_cm_id;
	debugfs_init();

	return 0;

destroy_cm_id:
	if (rdma_d.cm_id) {
		if (rdma_d.cm_id->qp && !IS_ERR(rdma_d.cm_id->qp))
			ib_destroy_qp(rdma_d.cm_id->qp);
		rdma_destroy_id(rdma_d.cm_id);
		rdma_d.cm_id = NULL;
	}
	if (rdma_d.cq && !IS_ERR(rdma_d.cq)) {
		ib_destroy_cq(rdma_d.cq);
		rdma_d.cq = NULL;
	}
	if (rdma_d.pd && !IS_ERR(rdma_d.pd)) {
		ib_dealloc_pd(rdma_d.pd);
		rdma_d.pd = NULL;
	}
	return ret;
}

static void __exit rdma_exit(void)
{
	flush_scheduled_work();
	debugfs_cleanup();
	if (rdma_d.cm_id) {
		if (rdma_d.cm_id->qp && !IS_ERR(rdma_d.cm_id->qp))
			ib_destroy_qp(rdma_d.cm_id->qp);
		rdma_destroy_id(rdma_d.cm_id);
	}
	destroy_buffer(&rdma_d);
	if (rdma_d.cq && !IS_ERR(rdma_d.cq)) {
		ib_destroy_cq(rdma_d.cq);
	}
	if (rdma_d.pd && !IS_ERR(rdma_d.pd)) {
		ib_dealloc_pd(rdma_d.pd);
	}
}

MODULE_LICENSE("GPL");
module_init(rdma_init);
module_exit(rdma_exit);
