#include "pp_common.h"

static inline char *get_link_layer_str(int layer)
{
	if (layer == IBV_LINK_LAYER_INFINIBAND)
		return "infiniband";
	else if (layer == IBV_LINK_LAYER_ETHERNET)
		return "ethernet";
	else
		return "unspecified";

}

static int pp_open_ibvdevice(const char *ibv_devname, struct pp_context *ppctx)
{
	int i = 0, ret = 0;
	const char *devname;
	struct ibv_device **dev_list;

	dev_list = ibv_get_device_list(NULL);
	if (!dev_list) {
		ERR("ibv_get_device_list()");
		return -errno;
	}

	for (i = 0; dev_list[i] != NULL; i++) {
		devname = ibv_get_device_name(dev_list[i]);
		//printf("=DEBUG:%s:%d: %s, %s\n", __func__, __LINE__, dev, s_ibv_devname);
		if (strncmp(devname, ibv_devname, strlen(ibv_devname)) == 0) {
			INFO("Device found: %d/%s\n", i, devname);
			break;
		}
	}
	if (dev_list[i] == NULL) {
		ERR("Device not found: %d/%s\n", i, ibv_devname);
		return -1;
	}

	ppctx->dev_list = dev_list;
	ppctx->ibctx = ibv_open_device(dev_list[i]);
	if (ppctx->ibctx == NULL) {
		ERR("ibv_open_device(%s), i=%d", ibv_devname, i);
		return errno;
	}

	ppctx->port_num = PORT_NUM;
	do {
		ret = ibv_query_port(ppctx->ibctx, PORT_NUM, &ppctx->port_attr);
		if (ret) {
			perror("ibv_query_port");
			return ret;
		}
		INFO("ibdev %s port %d port_state %d (expect %d) phy_state %d\n", ibv_devname, ppctx->port_num,
		     ppctx->port_attr.state, IBV_PORT_ACTIVE, ppctx->port_attr.phys_state);

		if (ppctx->port_attr.state == IBV_PORT_ACTIVE)
			break;
		sleep(1);
	} while (1);

	INFO("ibdev %s port %d lid %d state %d, mtu max %d active %d, link_layer %d(%s) phy_state %d speed %d\n",
	     ibv_devname, ppctx->port_num, ppctx->port_attr.lid, ppctx->port_attr.state,
	     ppctx->port_attr.max_mtu, ppctx->port_attr.active_mtu,
	     ppctx->port_attr.link_layer, get_link_layer_str(ppctx->port_attr.link_layer),
	     ppctx->port_attr.phys_state, ppctx->port_attr.active_speed);

	/*
        if (ppctx->port_attr.link_layer != IBV_LINK_LAYER_ETHERNET) {
		server_sgid_idx = 0;
		client_sgid_idx = 0;
	}
	*/
	return 0;
}

static void pp_close_ibvdevice(struct ibv_context *ibctx)
{
	int err = ibv_close_device(ibctx);
	if (err)
		perror("mz_close_ibvdevice");
}


int pp_ctx_init(struct pp_context *pp, const char *ibv_devname,
		int use_vfio, const char *vfio_pci_name)
{
	struct mlx5dv_vfio_context_attr vfio_ctx_attr = {
		.pci_name = vfio_pci_name,
		.flags = MLX5DV_VFIO_CTX_FLAGS_INIT_LINK_DOWN,
		.comp_mask = 0,
	};
	int ret, i;

	pp->cap.max_send_wr = PP_MAX_WR;
	pp->cap.max_recv_wr = PP_MAX_WR;
	pp->cap.max_send_sge = 1;
	pp->cap.max_recv_sge = 1;
	pp->cap.max_inline_data = 64;

	if (use_vfio) {
		struct ibv_device *ibdev;

		pp->dev_list = mlx5dv_get_vfio_device_list(&vfio_ctx_attr);
		if (!pp->dev_list) {
			ERR("mlx5dv_get_vfio_device_list returns NULL\n");
			return errno;
		}
		ibdev = pp->dev_list[0];
		pp->ibctx = ibv_open_device(ibdev);
		if (!pp->ibctx) {
			ERR("ibv_open_device(%s) failed: %d\n", vfio_pci_name, errno);
			return errno;
		}
		pp->port_num = PORT_NUM;
	} else {
		ret = pp_open_ibvdevice(ibv_devname, pp);
		if (ret)
			return ret;
	}

	pp->pd = ibv_alloc_pd(pp->ibctx);
	if (!pp->pd) {
		ERR("ibv_alloc_pd() failed\n");
		ret = errno;
		goto fail_alloc_pd;
	}

	pp->mrbuflen = PP_DATA_BUF_LEN;
	for (i = 0; i < PP_MAX_WR; i++) {
		pp->mrbuf[i] = memalign(sysconf(_SC_PAGESIZE), pp->mrbuflen);
		if (!pp->mrbuf[i]) {
			ERR("%d: memalign(0x%lx) failed\n", i, pp->mrbuflen);
			ret = errno;
			goto fail_memalign;
		}

		pp->mr[i] = ibv_reg_mr(pp->pd, pp->mrbuf[i], pp->mrbuflen,
				       IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
		if (!pp->mr[i]) {
			ERR("%d: ibv_reg_mr() failed\n", i);
			ret = errno;
			goto fail_reg_mr;
		}
	}

	if (use_vfio)
		INFO("VFIO open(%s) succeeds, flags 0x%x\n\n", vfio_pci_name, vfio_ctx_attr.flags);
	else
		INFO("Initialization succeeds(regular)\n\n");


	return 0;

fail_reg_mr:
	for (i = 0; i < PP_MAX_WR; i++)
		if (pp->mr[i])
			ibv_dereg_mr(pp->mr[i]);
fail_memalign:
	for (i = 0; i < PP_MAX_WR; i++)
		free(pp->mrbuf[i]);
	ibv_dealloc_pd(pp->pd);
fail_alloc_pd:
	pp_close_ibvdevice(pp->ibctx);

	return ret;
}

void pp_ctx_cleanup(struct pp_context *pp)
{
	int i;

	for (i = 0; i < PP_MAX_WR; i++)
		ibv_dereg_mr(pp->mr[i]);
	for (i = 0; i < PP_MAX_WR; i++)
		free(pp->mrbuf[i]);
	ibv_dealloc_pd(pp->pd);
	pp_close_ibvdevice(pp->ibctx);
}

static void print_gid(struct pp_context *ppc, unsigned char *p)
{
	if (ppc->port_attr.link_layer != IBV_LINK_LAYER_ETHERNET)
		return;

	printf("                                    gid %02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x:%02x\n",
	       p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7],
	       p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
}

extern int sock_client(const char *server_ip, char *sendbuf, int send_buflen,
		       char *recvbuf, int recv_buflen);
extern int sock_server(char *sendbuf, int send_buflen, char *recvbuf, int recv_buflen);

/* %sip NULL means local is server, otherwise local is client */
int pp_exchange_info(struct pp_context *ppc, int my_sgid_idx,
		     int my_qp_num, uint32_t my_psn,
		     struct pp_exchange_info *remote, const char *sip)
{
	char sendbuf[4096] = {}, recvbuf[4096] = {};
	struct pp_exchange_info *local = (struct pp_exchange_info *)sendbuf;
	struct pp_exchange_info *r = (struct pp_exchange_info *)recvbuf;
	unsigned char *p;
	int ret, i;

	if (!ppc->port_num) {
		ERR("pp_context isn't initialized: pp->port_num is 0\n");
		return -1;
	}

	if (ppc->port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
		ret = ibv_query_gid(ppc->ibctx, ppc->port_num,
				    my_sgid_idx, &local->gid);
		if (ret) {
			ERR("ibv_query_gid failed %d\n", ret);
			return ret;
		}
	} else {
		local->lid = htobe32(ppc->port_attr.lid);
	}

	local->qpn = htobe32(my_qp_num);
	local->psn = htobe32(my_psn);
	for (i = 0; i < PP_MAX_WR; i++) {
		local->addr[i] = (void *)htobe64((uint64_t)ppc->mrbuf[i]);
		local->mrkey[i] = htobe32(ppc->mr[i]->lkey);
	}
	p = local->gid.raw;
	INFO("Local(%s): port_num %d, lid %d, psn 0x%x, qpn 0x%x(%d), addr %p, mrkey 0x%x\n",
	     sip ? "Client" : "Server", ppc->port_num, ppc->port_attr.lid, my_psn,
	     my_qp_num, my_qp_num, ppc->mrbuf[PP_MAX_WR - 1], ppc->mr[PP_MAX_WR - 1]->lkey);
	print_gid(ppc, p);

	if (sip)
		ret = sock_client(sip, sendbuf, sizeof(*local),
				  recvbuf, sizeof(recvbuf));
	else
		ret = sock_server(sendbuf, sizeof(*local),
				  recvbuf, sizeof(recvbuf));
	if (ret) {
		ERR("socket failed %d, server_ip %s\n", ret, sip ? sip : "");
		return ret;
	}

	remote->lid = be32toh(r->lid);
	remote->qpn = be32toh(r->qpn);
	remote->psn = be32toh(r->psn);
	for (i = 0; i < PP_MAX_WR; i++) {
		remote->addr[i] = (void *)be64toh((uint64_t)r->addr[i]);
		remote->mrkey[i] = be32toh(r->mrkey[i]);
	}
	memcpy(&remote->gid, &r->gid, sizeof(remote->gid));
	p = remote->gid.raw;
	INFO("Remote(%s): lid %d, psn 0x%x, qpn 0x%x(%d), addr %p, mrkey 0x%x\n",
	     sip ? "Server" : "Client", remote->lid, remote->psn,
	     remote->qpn, remote->qpn, remote->addr[PP_MAX_WR - 1], remote->mrkey[PP_MAX_WR - 1]);
	print_gid(ppc, p);
	printf("\n");

	return 0;
}

/* To dump a long string like this: "0: 0BCDEFGHIJKLMNOP ... nopqrstuvwxyABC" */
void dump_msg_short(int index, struct pp_context *ppc)
{
	ppc->mrbuf[index][ppc->mrbuflen - 1] = '\0';
	if (ppc->mrbuflen <= 32) {
		printf("    %2d: %s\n", index, ppc->mrbuf[index]);
	} else {
		ppc->mrbuf[index][16] = '\0';
		printf("    %2d (len = 0x%lx): %s...%s\n", index, ppc->mrbuflen,
		       ppc->mrbuf[index], ppc->mrbuf[index] + ppc->mrbuflen - 16);
	}
}
