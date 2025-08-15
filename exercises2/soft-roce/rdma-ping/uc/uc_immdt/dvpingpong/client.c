#include "pp_common.h"
#include "pp_verb.h"

#define SERVER_IP "10.22.116.221"
//#define SERVER_IP "192.168.60.205"

static char ibv_devname[100] = "mlx5_1";
static int client_sgid_idx = 3;

//#define PP_VERB_OPCODE_CLIENT IBV_WR_SEND_WITH_IMM
#define PP_VERB_OPCODE_CLIENT IBV_WR_RDMA_WRITE_WITH_IMM

#define PP_SEND_WRID_CLIENT  0x1000
#define PP_RECV_WRID_CLIENT  0x4000

static struct pp_verb_ctx ppv_ctx;
static struct pp_exchange_info server = {};

static int client_traffic_verb(struct pp_verb_ctx *ppv)
{
	struct ibv_send_wr wrs[PP_MAX_WR] = {}, *bad_wr_send;
	struct ibv_recv_wr wrr[PP_MAX_WR] = {}, *bad_wr_recv;
	struct ibv_sge sglists[PP_MAX_WR] = {};
	int max_wr_num = PP_MAX_WR, ret;

	DBG("Pause 1sec ");
	sleep(1);		/* Wait until server side is ready to recv */
	DBG("Do post_send %d messages with length 0x%lx..\n", max_wr_num, ppv->ppc.mrbuflen);

	prepare_send_wr_verb(ppv, wrs, sglists, &server, max_wr_num,
			     PP_SEND_WRID_CLIENT, PP_VERB_OPCODE_CLIENT, true);

	/* 1. Send "ping" */
	ret = ibv_post_send(ppv->cqqp.qp, wrs, &bad_wr_send);
	if (ret) {
		ERR("%d: ibv_post_send failed %d\n", max_wr_num, ret);
		return ret;
	}

	ret = poll_cq_verb(ppv, max_wr_num, false);
	if (ret) {
		ERR("poll_cq_verb failed\n");
		return ret;
	}

	INFO("Send done, now recving reply...\n");
	prepare_recv_wr_verb(ppv, wrr, sglists, max_wr_num, PP_RECV_WRID_CLIENT);
	/* 2. Get "pong" with same data */
	ret = ibv_post_recv(ppv->cqqp.qp, wrr, &bad_wr_recv);
	if (ret) {
		ERR("%d: ibv_post_send failed %d\n", max_wr_num, ret);
		return ret;
	}

	ret = poll_cq_verb(ppv, max_wr_num, true);
	if (ret) {
		ERR("poll_cq_verb failed\n");
		return ret;
	}

	INFO("Client(verb) traffic test done\n");
	return 0;
}

int main(int argc, char *argv[])
{
	int ret;

	if (argv[1]) {
		memset(ibv_devname, 0, sizeof(ibv_devname));
		strcpy(ibv_devname, argv[1]);
	}
	INFO("IB device %s, server ip %s\n", ibv_devname, SERVER_IP);

	ret = pp_ctx_init(&ppv_ctx.ppc, ibv_devname, 0, NULL);
	if (ret)
		return ret;

	ret = pp_create_cq_qp_verb(&ppv_ctx.ppc, &ppv_ctx.cqqp);
	if (ret)
		goto out_create_cq_qp;

	ret = pp_exchange_info(&ppv_ctx.ppc, client_sgid_idx, ppv_ctx.cqqp.qp->qp_num,
			       CLIENT_PSN, &server, SERVER_IP);
	if (ret)
		goto out_exchange;

	ret = pp_move2rts_verb(&ppv_ctx.ppc, ppv_ctx.cqqp.qp, client_sgid_idx,
			       CLIENT_PSN, &server);
	if (ret)
		goto out_exchange;

	ret = client_traffic_verb(&ppv_ctx);

out_exchange:
	pp_destroy_cq_qp_verb(&ppv_ctx.cqqp);
out_create_cq_qp:
	pp_ctx_cleanup(&ppv_ctx.ppc);
	return ret;
}
