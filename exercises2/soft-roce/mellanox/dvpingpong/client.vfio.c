#include <signal.h>
#include <sys/epoll.h>

#include "pp_common.h"
#include "pp_dv.h"
#include "pp_vfio.h"

#define SERVER_IP "10.237.1.205"

char *vfio_pci_name = "0000:3b:00.1"; /* env(VFIO_PCI_NAME) */

static struct pp_dv_ctx ppvfio;
static struct pp_exchange_info server = {};

static int client_traffic_dv(struct pp_dv_ctx *ppdv)
{
	int num_post = PP_MAX_WR, num_comp, i, ret;
	//int opcode = MLX5_OPCODE_RDMA_WRITE_IMM;
	int opcode = MLX5_OPCODE_SEND_IMM;

	DBG("Pause 1sec before post send, opcode %d\n", opcode);
	sleep(1);

	for (i = 0; i < num_post; i++) {
		mem_string(ppdv->ppc.mrbuf[i], ppdv->ppc.mrbuflen);
		*ppdv->ppc.mrbuf[i] = i % ('z' - '0') + '0';
	}

	ret = pp_dv_post_send(&ppdv->ppc, &ppdv->qp, &server, num_post,
			      opcode, IBV_SEND_SIGNALED);
	if (ret) {
		ERR("pp_dv_post_send failed\n");
		return ret;
	}

	num_comp = 0;
	while (num_comp < num_post) {
		ret = pp_dv_poll_cq(&ppdv->cq, 1);
		if (ret == CQ_POLL_ERR) {
			ERR("poll_cq(send) failed %d, %d/%d\n", ret, num_comp, num_post);
			return ret;
		}
		if (ret > 0)
			num_comp++;
	}

	/* Reset the buffer so that we can check it the received data is expected */
	for (i = 0; i < num_post; i++)
		memset(ppdv->ppc.mrbuf[i], 0, ppdv->ppc.mrbuflen);

	INFO("Send done (num_post %d), now recving reply...\n", num_post);
	ret = pp_dv_post_recv(&ppdv->ppc, &ppdv->qp, num_post);
	if (ret) {
		ERR("pp_dv_post_recv failed\n");
		return ret;
	}

	num_comp = 0;
	while (num_comp < num_post) {
		ret = pp_dv_poll_cq(&ppdv->cq, 1);
		if (ret == CQ_POLL_ERR) {
			ERR("poll_cq(recv) failed %d, %d/%d\n", ret, num_comp, num_post);
			return ret;
		}
		if (ret > 0) {
			dump_msg_short(num_comp, &ppdv->ppc);
			num_comp++;
		}
	}

	INFO("Client(dv) traffic test done\n");
	return 0;
}

#define MAX_EVENTS 1
void *vfio_poll_eq_event_routine(void *arg)
{
	struct pp_context *pp = (struct pp_context *)arg;
	struct epoll_event ev, events[MAX_EVENTS];
	int vfio_efd, epoll_fd, nfds, i, ret;

	INFO("running poll thread...\n");

	vfio_efd = mlx5dv_vfio_get_events_fd(pp->ibctx);
	if (vfio_efd < 0) {
		ERR("mlx5dv_vfio_get_events_fd failed %d\n", vfio_efd);
		return NULL;
	}

	epoll_fd = epoll_create1(0);
	if (epoll_fd < 0) {
		ERR("epoll_create1 failed\n");
		return NULL;
	}
	ev.events = EPOLLIN;
	ev.data.fd = vfio_efd;
	ret = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, vfio_efd, &ev);
	if (ret < 0) {
		ERR("epoll_ctl failed\n");
		return NULL;
	}

	while (1) {
		nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
		if (nfds < 0) {
			ERR("epoll_wait failed\n");
			return NULL;
		}

		for (i = 0; i < nfds; i++) {
			DBG("%d/%d: event 0x%x\n", i, nfds, events[i].events);
			if ((events[i].events & EPOLLERR) ||
			    (events[i].events & EPOLLHUP) ||
			    (!(events[i].events & EPOLLIN))) {
				ERR("events 0x%x\n", events[i].events);
				return NULL;
			}
			mlx5dv_vfio_process_events(pp->ibctx);
		}
	}

	ERR("returned unexpectedly");
	return NULL;
}

static pthread_t event_tid;
int setup_event_routine(struct pp_context *pp)
{
	int ret;

	ret = pthread_create(&event_tid, NULL, vfio_poll_eq_event_routine, pp);
	if (ret) {
		perror("pthread_create");
		return ret;
	}

	usleep(100);
	INFO("pthread created\n");
	return 0;
}

static void sig_handler(int signum)
{
	ERR("Signal %d is captured!\n", signum);
}

static int setup_sighandler(void)
{
	struct sigaction new_action;
	int ret;

	new_action.sa_handler = sig_handler;
	sigemptyset(&new_action.sa_mask);
	new_action.sa_flags = 0;
	ret = sigaction(SIGABRT, &new_action, NULL);
	if (ret)
		ERR("sigaction(SIGABRT) failed %d\n", ret);

	return ret;
}

static int vfio_init(struct pp_context *ppc)
{
	int ret;

	/* Dump some hca_cap to check if vfio works */
	ret = pp_query_hca_cap(ppc);
	if (ret)
		return ret;

	ret = setup_sighandler();
	if (ret)
		return ret;

	ret = setup_event_routine(ppc);
	if (ret)
		return ret;

	ret = pp_config_port(ppc->ibctx, MLX5_PORT_UP);
	if (ret)
		return ret;

	do {
		ret = pp_query_mad_ifc_port(ppc->ibctx, 1, &ppc->port_attr);
		if (ret)
			return ret;

		if ((ppc->port_attr.state >= IBV_PORT_ACTIVE) &&
		    (ppc->port_attr.lid != 65535))
			break;

		sleep(1);
	} while (1);
	INFO("Pause 3 seconds to make sure server start to listen...\n\n");
	sleep(3);
	return 0;
}

static void vfio_cleanup(struct pp_context *ppc)
{
	void *res;

	pthread_cancel(event_tid);
	pthread_join(event_tid, &res);
}

static void parse_arg(int argc, char *argv[])
{
	char *v;

	v = getenv("VFIO_PCI_NAME");
	if (v)
		vfio_pci_name = v;
}

int main(int argc, char *argv[])
{
	int ret;

	parse_arg(argc, argv);
	INFO("VFIO pci device: %s\n", vfio_pci_name);

	ret = pp_ctx_init(&ppvfio.ppc, NULL, true, vfio_pci_name);
	if (ret)
		return ret;

	ret = vfio_init(&ppvfio.ppc);
	if (ret)
		goto out_vfio_init;

	ret = pp_create_cq_dv(&ppvfio.ppc, &ppvfio.cq);
	if (ret)
		goto out_create_cq;

	ret = pp_create_qp_dv(&ppvfio.ppc, &ppvfio.cq, &ppvfio.qp);
	if (ret)
		goto out_create_qp;

	ret = pp_exchange_info(&ppvfio.ppc, 0, ppvfio.qp.qpn,
			       CLIENT_PSN, &server, SERVER_IP);
	if (ret)
		goto out_exchange;

	ret = pp_move2rts_dv(&ppvfio.ppc, &ppvfio.qp, 0,
			     CLIENT_PSN, &server);
	if (ret)
		goto out_exchange;

	ret = client_traffic_dv(&ppvfio);


out_exchange:
	pp_destroy_qp_dv(&ppvfio.qp);
out_create_qp:
	pp_destroy_cq_dv(&ppvfio.cq);
out_create_cq:
	vfio_cleanup(&ppvfio.ppc);
out_vfio_init:
	pp_ctx_cleanup(&ppvfio.ppc);
	return ret;
}
