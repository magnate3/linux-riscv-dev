/*******************************************************************************
 * This file contains the smbdirect driver for Samba
 *
 * (c) Richard Sharpe <rsharpe@samba.org>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 ****************************************************************************/

#include <linux/delay.h>
#include <linux/init.h>
#include <linux/string.h>
#include <linux/module.h>
#include <linux/socket.h>
#include <linux/ioctl.h>
#include <linux/inet.h>
#include <uapi/linux/in.h>
#include <linux/in6.h>
#include <rdma/ib_verbs.h>
#include <rdma/rdma_cm.h>
#include <linux/seq_file.h>

#include "smbdirect.h"

/*
 * TODO: Convert to using dev_dbg, but probably need a platform_device for
 * that.
 */

/*
 * The port number we listen on
 */
#define SMB_DIRECT_PORT 5445

/* Max CQ depth per queue ... */
#define MAX_CQ_DEPTH 128

/* Our device number, for reporting */
static dev_t smbdirect_dev_no;
static int connection_count = 0;

/*
 * A /proc file for some debugging ... replace this with configfs stuff ...
 */
static int read_proc_stuff(struct seq_file *m, void *data)
{

	seq_printf(m, " %s: Loaded with major = %u, minor = %u\n"
			"Connection Count = %d\n",
			"smbdirect",
			connection_count,
			MAJOR(smbdirect_dev_no),
			MINOR(smbdirect_dev_no));
	return 0;
}

/*
 * The device and file ops etc
 */
static struct smbd_device smbd_device;

static struct workqueue_struct *smbd_wq;

int smbd_open(struct inode *inode, struct file *filp)
{

	return 0;
}

int smbd_release(struct inode *inode, struct file *filp)
{
	return 0;
}

/*
 * Relocate these definitions to a .h file
 */
#define SMBD_IOC_VAL 's'
#define SMBD_SET_PARAMS     _IOR(SMBD_IOC_VAL, 1, void *)
#define SMBD_GET_MEM_PARAMS _IOW(SMBD_IOC_VAL, 2, void *)
#define SMBD_SET_SESSION_ID _IOR(SMBD_IOC_VAL, 3, void *)

/*
 * Get the params and mark us as initialized
 */
long handle_set_params(unsigned long arg)
{
	int res = 0;
	struct smbd_params *params = (void __user *)arg;

	res = copy_from_user(&smbd_device.params, 
			params, 
			sizeof(struct smbd_params));

	if (!res) {
		printk(KERN_ERR "Error: Memory for SMBD_SET_PARAMS "
			"not accessible\n");
		return -EFAULT;
	}

	/*
	 * Check the values and also copy the blob
	 */

	/*
	 * We are initialized now
	 */
	smbd_device.initialized = 1;

	return res;
}

long smbd_ioctl(struct file *filp, unsigned int cmd,
		unsigned long arg)
{
	int res = -EINVAL;

	printk(KERN_INFO "Handling ioctl: %0X\n", cmd);

	switch (cmd) {
	case SMBD_SET_PARAMS:
		return handle_set_params(arg);
		break;

	case SMBD_GET_MEM_PARAMS:

		break;

	case SMBD_SET_SESSION_ID:

		break;

	default:

		break;
	}

	return res;
}

struct file_operations smbd_fops = {
	.owner = THIS_MODULE,
	.open = smbd_open,
	.unlocked_ioctl = smbd_ioctl,
	.release = smbd_release,
};

/*
 * Clean up buffers etc ...
 */
static void
smbd_free_buffers(struct connection_struct *conn)
{
	if (conn->recv_mr)
		ib_dereg_mr(conn->recv_mr);
	ib_dma_unmap_single(conn->cm_id->device,
			    conn->recv_buf_dma,
			    sizeof(conn->recv_buf),
			    DMA_FROM_DEVICE);
}

/*
 * Clean up a connection ...
 */
static void
smbd_clean_connection(struct connection_struct *conn, bool disconnect)
{
	smbd_free_buffers(conn);
	printk(KERN_INFO "Freed the buffers for conn: %p\n", conn);
	msleep(1);

	if (disconnect)
		rdma_disconnect(conn->cm_id);
	printk(KERN_INFO "Disconnected ...\n");
	msleep(1);

	ib_destroy_qp(conn->qp);
	ib_destroy_cq(conn->cq);
	ib_dealloc_pd(conn->pd);
	printk(KERN_INFO "Cleaned up qp, cq and pd ... \n");
	msleep(100);

	rdma_destroy_id(conn->cm_id);
	printk(KERN_INFO "Destroyed the cm_id\n");
}

/*
 * Start a send ... if an error occurs, set the state to ERROR and let others
 * clean up.
 */
static void
start_send(struct connection_struct *conn)
{
	int res = 0;
	struct ib_send_wr *bad_wr;

	printk(KERN_INFO "Starting to send ...\n");

	conn->send_buf_dma = ib_dma_map_single(conn->cm_id->device,
					       conn->send_buf,
					       sizeof(conn->send_buf),
					       DMA_TO_DEVICE);
	res = ib_dma_mapping_error(conn->cm_id->device, conn->send_buf_dma);
	if (res) {
		printk(KERN_ERR "Failed to create send DMA mapping: %d\n", res);
		goto error;
	}

	printk(KERN_INFO "Mapped the buffer ...\n");

	conn->send_mr = ib_get_dma_mr(conn->pd, 0);
	if (IS_ERR(conn->send_mr)) {
		printk(KERN_ERR "Failed to get the send MR: %ld\n",
			PTR_ERR(conn->recv_mr));
		goto error_dma;
	}

	printk(KERN_INFO "Got the memory region ...\n");

	/*
	 * Set up the work request ...
	 */
	conn->send_sgl.addr = conn->send_buf_dma;
	conn->send_sgl.length = sizeof(conn->send_buf);
	conn->send_sgl.lkey = conn->send_mr->lkey;

	conn->send_wr.opcode = IB_WR_SEND;
	conn->send_wr.send_flags = IB_SEND_SIGNALED;
	conn->send_wr.sg_list = &conn->send_sgl;
	conn->send_wr.num_sge = 1;
	conn->send_wr.next = NULL;

	res = ib_post_send(conn->qp, &conn->send_wr, &bad_wr);
	if (res) {
		printk(KERN_ERR "Unable to post a send: %d\n", res);
		goto error_mr;
	}
	return;

error_mr:
	ib_dereg_mr(conn->send_mr);
error_dma:
	ib_dma_unmap_single(conn->cm_id->device,
			    conn->send_buf_dma,
			    sizeof(conn->send_buf),
			    DMA_TO_DEVICE);
error:	
	conn->state = SMBD_ERROR;
}

/*
 * Handle a recv completion.
 *
 * This is a state machine. If we are in SMBD_NEGOTIATE, then handle the
 * NEGOTIATE request. If in SMBD_TRANSFER state, then we have PDUs to deal
 * with.
 */
static int
handle_recv_completion(struct ib_wc *wc, struct connection_struct *conn)
{
	int res = 0;
	struct smbd_negotiate_req *neg_req;
	struct smbd_negotiate_resp *neg_resp;

	switch (conn->state) {
	case SMBD_NEGOTIATE:
		if (wc->byte_len != sizeof(struct smbd_negotiate_req)) {
			printk(KERN_ERR "Incorrect negotiate request size: "
				"%d, dropping connection\n", wc->byte_len);
			conn->state = SMBD_ERROR;
			break;
		}

		neg_req = (struct smbd_negotiate_req *)conn->recv_buf;
		printk(KERN_INFO "Negotiate request. Length: %lu.\n"
			"\tMin version: 0x%04x,"
			"Max version: 0x%04x, Credits requested: %d\n",
			sizeof(struct smbd_negotiate_req),
			neg_req->min_version, neg_req->max_version,
			neg_req->credits_requested);

		/* We should initialize the parameters correctly */

		/* We should save these values as well */

		conn->state = SMBD_TRANSFER;
		neg_resp = (struct smbd_negotiate_resp *)conn->send_buf;

		neg_resp->min_version = neg_req->min_version;
		neg_resp->max_version = neg_req->max_version;
		neg_resp->negotiated_version = neg_req->min_version;
		neg_resp->reserved = 0;
		neg_resp->credits_requested = 255;
		neg_resp->credits_granted = 16;
		neg_resp->status = 0;
		neg_resp->max_read_write_size = 65536;

		/* Now, send the response */

		start_send(conn);

		break;

	case SMBD_TRANSFER:

		break;

	case SMBD_ERROR:


		break;

	default:
		printk(KERN_ERR "Connection in unknown state: %u, "
			"disconnecting\n",
			conn->state);
		conn->state = SMBD_ERROR;
		break;
	}

	return res;
}

/*
 * Handle a completion event ... in a work qeue
 */
static void
conn_cq_work(struct work_struct *work)
{
	struct connection_struct *conn = container_of(work,
						struct connection_struct,
						cq_work);

	int res = 0;
	struct ib_wc wc;

	printk(KERN_INFO "Handling a completion event on conn: %p\n", conn);

	/*
	 * Get all the available completions ...
	 */
	while ((res = ib_poll_cq(conn->cq, 1, &wc)) == 1) {
		int local_res = 0;

		/*
		 * If it is a bad WC, I think we need to go to ERROR state
		 */
		if (wc.status) {
			printk(KERN_ERR "cq completion failed with ID: %Lx "
				"status %d opcode 0x%0x vendor_err 0x%0x\n",
				wc.wr_id, wc.status, wc.opcode, 
				wc.vendor_err);
			conn->state = SMBD_ERROR;
			continue;
		}

		switch (wc.opcode) {
		case IB_WC_SEND:
			printk(KERN_INFO "SEND completion ...\n");
			break;

		case IB_WC_RECV:
			printk(KERN_INFO "RECV completion ...\n");
			local_res = handle_recv_completion(&wc, conn);
			break;

		case IB_WC_RDMA_WRITE:
			printk(KERN_INFO "RDMA_WRITE completion ...\n");
			break;

		case IB_WC_RDMA_READ:
			printk(KERN_INFO "RDMA_READ completion ...\n");
			break;

		default:
			printk(KERN_ERR "%s:%d Unexpected completion: %d\n",
				__func__, __LINE__, wc.opcode);
			break;
		}
	}

	if (res) {
		printk(KERN_ERR "%s: ib poll error: %d\n", __func__, res);

		/*
		 * We have a problem, so we should drop the connection
		 */
		conn->state = SMBD_ERROR;
	}

	/*
	 * Clean up the connection
	 */
	if (conn->state == SMBD_ERROR) {

		goto done;
	}

	/*
	 * Make sure we get some more completion events.
	 */
	res = ib_req_notify_cq(conn->cq, IB_CQ_NEXT_COMP);
	if (res) {
		printk(KERN_ERR "Could not request notification: %d\n", res);
	}

done:
	return;
}

/*
 * Handle a completion event ... place on a work queue
 */
static void handle_completion_event(struct ib_cq *cq, void *ctx)
{
	struct connection_struct *conn = ctx;

	if (conn->cq != cq) {
		printk(KERN_ERR "The completion queue is wrong: %p, %p\n",
			conn->cq, cq);
		return;
	}

	printk(KERN_INFO "Handing completion off to a work queue\n");
	INIT_WORK(&conn->cq_work, conn_cq_work);
	queue_work(smbd_wq, &conn->cq_work);
}

/*
 * Handle connection requests ... build a new connection and set up the 
 * protection domain, completion queue and queue pair, hang a receive and
 * accept the connection.
 */
static int
handle_connect_request(struct rdma_cm_id *cm_id,
		       struct smbd_device *smbd_dev)
{
	int res = 0;
	struct connection_struct *conn = NULL;
	struct ib_qp_init_attr conn_attr;
	struct rdma_conn_param cparam;
	struct ib_recv_wr *bad_wr;

	printk(KERN_INFO "Handling connection request ...\n");
	
	conn = kzalloc(sizeof(conn), GFP_KERNEL);
	if (!conn) {
		printk(KERN_ERR "Unable to allocate connection\n");
		return -ENOMEM;
	}

	conn->cm_id = cm_id;
	conn->dev = smbd_dev;
	init_waitqueue_head(&conn->wait_queue);

	/*
	 * Now allocate a protection domain, completion queue etc.
	 */
	conn->pd = ib_alloc_pd(cm_id->device);
	if (IS_ERR(conn->pd)) {
		res = PTR_ERR(conn->pd);
		printk(KERN_ERR "Allocation of PD failed: %d\n", res);
		goto clean_conn;
	}

	conn->cq = ib_create_cq(cm_id->device, 
				handle_completion_event,
				NULL,
				conn,
				MAX_CQ_DEPTH * 2,
				0);
	if (IS_ERR(conn->cq)) {
		res = PTR_ERR(conn->cq);
		printk(KERN_ERR "Unable to create completion queue: %d\n",
			res);
		goto clean_pd;
	}

	/*
	 * Request notifies on that completion queue
	 */
	res = ib_req_notify_cq(conn->cq, IB_CQ_NEXT_COMP);
	if (res) {
		printk(KERN_ERR "Unable to request notifications: %d\n", res);
		goto clean_cq;
	}

	memset(&conn_attr, 0, sizeof(conn_attr));
	conn_attr.cap.max_send_wr = 10;
	conn_attr.cap.max_recv_wr = 10;
	conn_attr.cap.max_recv_sge = 2;
	conn_attr.cap.max_send_sge = 2;
	conn_attr.qp_type = IB_QPT_RC;
	conn_attr.send_cq = conn->cq;
	conn_attr.recv_cq = conn->cq;
	conn_attr.sq_sig_type = IB_SIGNAL_REQ_WR;

	res = rdma_create_qp(conn->cm_id, conn->pd, &conn_attr);
	if (res) {
		printk(KERN_ERR "Unable to create queue pair: %d\n", res);
		goto clean_cq;
	}

	conn->qp = conn->cm_id->qp;

	/*
	 * Setup the receive buff params for first receive
	 */
	conn->recv_buf_dma = ib_dma_map_single(conn->cm_id->device,
					       conn->recv_buf,
					       sizeof(conn->recv_buf),
					       DMA_FROM_DEVICE);
	res = ib_dma_mapping_error(conn->cm_id->device, conn->recv_buf_dma);
	if (res) {
		printk(KERN_ERR "Failed to create DMA mapping: %d\n", res);
		goto clean_qp;
	}

	conn->recv_mr = ib_get_dma_mr(conn->pd, IB_ACCESS_LOCAL_WRITE);
	if (IS_ERR(conn->recv_mr)) {
		res = PTR_ERR(conn->recv_mr);
		printk(KERN_ERR "Unable to get recv MR: %d\n", res);
		goto clean_dma;
	}

	/*
	 * Set up the work request ...
	 */
	conn->recv_sgl.addr = conn->recv_buf_dma;
	conn->recv_sgl.length = sizeof(conn->recv_buf);
	conn->recv_sgl.lkey = conn->recv_mr->lkey;

	conn->recv_wr.sg_list = &conn->recv_sgl;
	conn->recv_wr.num_sge = 1;
	conn->recv_wr.next = NULL;

	/*
	 * Post that receive before we accept
	 */
	res = ib_post_recv(conn->qp, &conn->recv_wr, &bad_wr);
	if (res) {
		printk(KERN_ERR "Unable to post a recv: %d\n", res);
		goto clean_mr;
	}

	/*
	 * Accept the request ...
	 */
	memset(&cparam, 0, sizeof(cparam));
	cparam.responder_resources = 1;
	cparam.initiator_depth = 1;

	res = rdma_accept(cm_id, &cparam);
	if (res) {
		printk(KERN_ERR "Unable to accept connection: %d\n", res);
		goto clean_recv;
	}

	connection_count++;

	cm_id->context = conn; /* How we find it again */
	conn->state = SMBD_NEGOTIATE;

	mutex_lock(&smbd_dev->connection_list_mutex);
	list_add_tail(&conn->connect_ent, &smbd_dev->connection_list);
	mutex_unlock(&smbd_dev->connection_list_mutex);

	printk(KERN_INFO "Accepted the connection ...\n");

	return res;

clean_recv:

clean_mr:
	if (conn->recv_mr)
		ib_dereg_mr(conn->recv_mr);
clean_dma:
	ib_dma_unmap_single(conn->cm_id->device,
			    conn->recv_buf_dma,
			    sizeof(conn->recv_buf),
			    DMA_FROM_DEVICE);
clean_qp: 
	ib_destroy_qp(conn->qp);
clean_cq:
	ib_destroy_cq(conn->cq);
clean_pd:
	ib_dealloc_pd(conn->pd);
clean_conn:
	rdma_destroy_id(conn->cm_id);
	kfree(conn);
	return res;
}

/*
 * Handle a connection disconnect. Simply clean the connection, delete it
 * from the list and free it.
 */
static void
smbd_disconnect_work(struct work_struct *work)
{
	struct connection_struct *conn = container_of(work,
						struct connection_struct,
						disconnect_work);
	struct smbd_device *smbd_dev = conn->dev;

	msleep(10);
	/*
	 * No need to disconnect. It is going away.
	 */
	smbd_clean_connection(conn, false);
	printk(KERN_INFO "Cleaned the connection ...\n");

	mutex_lock(&smbd_dev->connection_list_mutex);
	list_del_init(&conn->connect_ent);
	mutex_unlock(&smbd_dev->connection_list_mutex);

	connection_count--;

	printk(KERN_INFO "Freeing the connection and we are done ...\n");
	kfree(conn);
}

static int
handle_disconnect(struct rdma_cm_id *cma_id)
{
	int res = 0;
	struct connection_struct *conn = cma_id->context;

	/*
	 * We are called in a context here where we cannot call
	 * a bunch of CM function, so handle this in a workqueue
	 */
	if (conn) {
		printk(KERN_INFO "Handling disconnect for %p\n", conn);
		INIT_WORK(&conn->disconnect_work, smbd_disconnect_work);
		queue_work(smbd_wq, &conn->disconnect_work);
	} else {
		printk(KERN_INFO "Received disconnect on incorrect cm_id: %p\n",
		cma_id);
	}

	return res;
}

/*
 * Handle CMA events ...
 */
static int
smbd_cma_handler(struct rdma_cm_id *cma_id,
		 struct rdma_cm_event *event)
{
	int res = 0;
	struct smbd_device *smbd_dev = cma_id->context;

	printk(KERN_INFO "cma_event type %d cma_id %p\n",
		event->event, cma_id);

	switch (event->event) {
	case RDMA_CM_EVENT_CONNECT_REQUEST:
		res = handle_connect_request(cma_id, smbd_dev);
		break;
	case RDMA_CM_EVENT_ESTABLISHED:
		printk("Received an established event. Ignored.\n");
		break;
	case RDMA_CM_EVENT_DISCONNECTED:
		res = handle_disconnect(cma_id);
		break;
	case RDMA_CM_EVENT_DEVICE_REMOVAL:
		printk("Received a device removal event. Ignored.\n");
		break;
	default:
		printk(KERN_ERR "Unknown event type: %d\n", event->event);
		break;
	}

	return res;
}

/*
 * Set up an RDMA CM Listen on our port ... and pass the device struct as
 * the context ...
 *
 * TODO: Generalize to IPV6 as well as IPV4.
 */
static int
setup_listen(struct smbd_device *smbd_dev)
{
	int res = 0;
	struct rdma_cm_id *smbd_lid = NULL;
	struct sockaddr_in sa;

	smbd_lid = rdma_create_id(smbd_cma_handler,
				  smbd_dev,
				  RDMA_PS_TCP,
				  IB_QPT_RC);
	if (IS_ERR(smbd_lid)) {
		res = PTR_ERR(smbd_lid);
		printk(KERN_ERR "rdma_create_id error %d\n", res);
		goto err;
	}

	smbd_dev->cm_lid = smbd_lid;

	/*
	 * Now bind to INADDR_ANY and our port.
	 */
	memset(&sa, 0, sizeof(sa));
	sa.sin_family = AF_INET;
	sa.sin_addr.s_addr = INADDR_ANY;
	sa.sin_port = ntohs(SMB_DIRECT_PORT);

	res = rdma_bind_addr(smbd_lid, (struct sockaddr *)&sa);
	if (res) {
		printk(KERN_ERR "rdma_bind_addr error %d\n", res);
		goto unlisten;
	}

	printk(KERN_INFO "rdma_bind_addr done\n");

	/* TODO, allow the backlog to be tuned */
	res = rdma_listen(smbd_lid, 5);
	if (res) {
		printk(KERN_ERR "rdma_listen failed %d\n", res);
		goto unlisten;
	}

	printk(KERN_INFO "rdma_listen done, lid %p\n", smbd_lid);

	/* The callback function will handle things from here ... */

	return res;

unlisten:
	rdma_destroy_id(smbd_lid);
err:
	return res;
}

/*
 * Tear down the listens and any connections ...
 */
static int teardown_listen_and_connections(struct smbd_device *smbd_dev)
{
	int res = 0;
	struct connection_struct *conn, *temp;

	printk(KERN_INFO "%s: Emptying list ...\n", __func__);
	/* We need to go through the list of connections and drop them */
	mutex_lock(&smbd_dev->connection_list_mutex);
	list_for_each_entry_safe(conn, 
				temp,
				&smbd_dev->connection_list, 
				connect_ent) {

		smbd_free_buffers(conn);
		rdma_disconnect(conn->cm_id);

		ib_destroy_qp(conn->qp);
		ib_destroy_cq(conn->cq);
		ib_dealloc_pd(conn->pd);

		rdma_destroy_id(conn->cm_id);

		list_del(&conn->connect_ent);
		kfree(conn);
	}
	mutex_unlock(&smbd_dev->connection_list_mutex);

	if (smbd_dev->cm_lid)
		rdma_destroy_id(smbd_dev->cm_lid);

	return res;
}

static int smbd_proc_rd_open(struct inode *inode, struct file *file)
{
	return single_open(file, read_proc_stuff, NULL);
}

static const struct file_operations smbd_proc_rd_fops = {
	.open = smbd_proc_rd_open,
	.read = seq_read,
	.llseek = seq_lseek,
	.release = seq_release,
};

static int __init smbdirect_init(void)
{
	int res = 0;

	proc_create("driver/smbdirect", 0, NULL, &smbd_proc_rd_fops);

	/*
	 * Allocate a new device major number
	 */
	res = alloc_chrdev_region(&smbdirect_dev_no, 0, 1, "smbdirect");
	if (res < 0) {
		printk(KERN_ERR "Major number allocation failed\n");
		return res;
	}

	smbd_wq = alloc_workqueue("SMBDirect Work Queue", 0, 0);
	if (!smbd_wq) {
		printk(KERN_ERR "Unable to allocate work queue\n");
		return ENOMEM;
	}

	memset(&smbd_device, 0, sizeof(smbd_device));
	mutex_init(&smbd_device.connection_list_mutex);
	INIT_LIST_HEAD(&smbd_device.connection_list);
	cdev_init(&smbd_device.cdev, &smbd_fops);
	smbd_device.cdev.owner = THIS_MODULE;

	res = cdev_add(&smbd_device.cdev, smbdirect_dev_no, 1);
	if (res) {
		printk(KERN_ERR "Unable to add smbdirect device: %d\n", res);
		goto no_cdev;
	}

	/*
	 * This should be called in the open function when we have initialized
	 */
	res = setup_listen(&smbd_device);
	if (res) {

		goto no_listen;
	}

	return res;

no_listen:
	cdev_del(&smbd_device.cdev);
	destroy_workqueue(smbd_wq);
no_cdev:
	return res;
}

static void __exit smbdirect_exit(void)
{
	(void)teardown_listen_and_connections(&smbd_device);
	cdev_del(&smbd_device.cdev);
	destroy_workqueue(smbd_wq);
	remove_proc_entry("driver/smbdirect", NULL);
}

MODULE_DESCRIPTION("smbdirect driver for Samba");
MODULE_VERSION("0.1");
MODULE_AUTHOR("rsharpe@samba.org");
MODULE_LICENSE("GPL");

module_init(smbdirect_init);
module_exit(smbdirect_exit);
