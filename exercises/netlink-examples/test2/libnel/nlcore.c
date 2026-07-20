/* Netlink core functions */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <stdarg.h>
#include <stdlib.h>
#include <syslog.h>
#include <sys/socket.h>
#include <net/if.h>
#include <linux/netlink.h>

#include "nlcore.h"

static int nlog_dbg;

void nlog(int priority, const char *frmt, ...)
{
	va_list args;

	if (priority == LOG_DEBUG && !nlog_dbg)
		return;

	va_start(args, frmt);

	vsyslog(priority, frmt, args);

	va_end(args);
}

int nl_open(struct nl_sock *nlsock, int service)
{
	struct sockaddr_nl sa;
	int n;

	if (getenv("LIBNEL_DEBUG"))
		nlog_dbg = 1;

	if (nlsock->pid > 0)
		nl_close(nlsock);

	nlsock->sock = socket(AF_NETLINK, SOCK_RAW, service);
	if (nlsock->sock < 0) {
		ERRNO("failed to open netlink sock");
		return -1;
	}

	memset(&sa, 0, sizeof(sa));
	sa.nl_family = AF_NETLINK;
	/* sa.nl_pid = getpid(); */
	/* sa.nl_groups = 0; */
	if (bind(nlsock->sock, (struct sockaddr *)&sa, sizeof(sa))) {
		ERRNO("failed to bind netlink socket");
		close(nlsock->sock);
		return -1;
	}

	n = sizeof(sa);
	getsockname(nlsock->sock, (struct sockaddr *)&sa, &n);
	nlsock->pid = sa.nl_pid;
	nlsock->seq = 0;
	nlsock->service = service;

	return 0;
}

void nl_close(struct nl_sock *nlsock)
{
	if (nlsock->pid <= 0)
		return;

	close(nlsock->sock);
	nlsock->sock = -1;
	nlsock->pid = -1;
	nlsock->seq = 0;
	nlsock->service = -1;
}

char *nlmsg_put_hdr(char *buf, int type, int flags)
{
	struct nlmsghdr *nlhdr = (struct nlmsghdr *)buf;

	memset(buf, 0, NLMSG_HDRLEN);

	nlhdr->nlmsg_len = NLMSG_HDRLEN;
	nlhdr->nlmsg_type = type;
	nlhdr->nlmsg_flags = flags | NLM_F_REQUEST;

	return buf + NLMSG_HDRLEN;
}

int nl_wait_ack(struct nl_sock *nlsock)
{
	struct sockaddr_nl sa;
	int n;
	char buf[128];
	struct nlmsghdr *nlhdr, *nlhdr2;
	struct nlmsgerr *errmsg;

	while (1) {
		n = sizeof(sa);
		n = recvfrom(nlsock->sock, buf, sizeof(buf), 0, (struct sockaddr *)&sa, &n);
		if (n < 0) {
			if (errno == EINTR || errno == EAGAIN)
				continue;
			ERRNO("failed to recv");
			goto err;
		}

		break;
	}

	nlhdr = (struct nlmsghdr *)buf;

	if (!NLMSG_OK(nlhdr, n)) {
		ERROR("invalid netlink header?");
		goto err;
	}

	/*
	 * NOTE: I don't know why pid is copied from the request (and not 0, like we
	 * expect to get from kernel).
	 */
	if (nlhdr->nlmsg_seq != nlsock->seq || nlhdr->nlmsg_type != NLMSG_ERROR) {
		ERROR("unexpected msg");
		goto err;
	}

	nlhdr2 = NLMSG_NEXT(nlhdr, n);
	if (NLMSG_OK(nlhdr2, n)) {
		ERROR("we don't expect anything after ack");
		goto err;
	}

	errmsg = NLMSG_DATA(nlhdr);
	DEBUG("error msg with code (errno)=%d (%s)", -errmsg->error,
		strerror(-errmsg->error));
	return errmsg->error;

err:
	nl_open(nlsock, nlsock->service);
	return -1;
}

int nl_send_msg(struct nl_sock *nlsock, char *buf, int len)
{
	struct nlmsghdr *nlhdr = (struct nlmsghdr *)buf;
	struct sockaddr_nl sa;
	int n;

	memset(&sa, 0, sizeof(sa));
	sa.nl_family = AF_NETLINK;
	//sa.nl_pid = 0; /* To kernel */
	//sa.nl_groups = 0; /* Unicast */

	nlhdr->nlmsg_len = len;
	nlhdr->nlmsg_pid = nlsock->pid;
	nlhdr->nlmsg_seq = ++nlsock->seq;

	n = NLMSG_ALIGN(nlhdr->nlmsg_len);
	if (sendto(nlsock->sock, nlhdr, n, 0, (struct sockaddr *)&sa, sizeof(sa)) != n) {
		ERRNO("failed to send");
		return -1;
	}

	DEBUG("send %d bytes", n);

	return 0;
}

int nl_recv_msg(struct nl_sock *nlsock, int type, int (*cb)(struct nlmsghdr *, void *),
		void *cb_priv)
{
	struct sockaddr_nl sa;
	int n;
	char buf[4096];
	struct nlmsghdr *nlhdr;
	struct nlmsgerr *errmsg;

	while (1) {
		n = sizeof(sa);
		n = recvfrom(nlsock->sock, buf, sizeof(buf), 0, (struct sockaddr *)&sa, &n);
		if (n < 0) {
			if (errno == EINTR || errno == EAGAIN)
				continue;
			ERRNO("failed to recv");
			goto err;
		}

		DEBUG("recv %d bytes", n);

		for (nlhdr = (struct nlmsghdr *)buf; NLMSG_OK(nlhdr, n); nlhdr = NLMSG_NEXT(nlhdr, n)) {
			DEBUG("get new msg: len=%d, type=0x%02x", nlhdr->nlmsg_len, nlhdr->nlmsg_type);

			if (nlhdr->nlmsg_type == NLMSG_ERROR) {
				errmsg = NLMSG_DATA(nlhdr);
				DEBUG("err msg: error=%d", errmsg->error);
				return -1;
			}

			if (nlhdr->nlmsg_type == NLMSG_DONE) {
				DEBUG("done msg");
				return cb(NULL, cb_priv);
			}

			if (nlhdr->nlmsg_seq != nlsock->seq || nlhdr->nlmsg_type != type) {
				ERROR("unexpected msg");
				goto err;
			}

			if (cb(nlhdr, cb_priv))
				goto err;

			if (!(nlhdr->nlmsg_flags & NLM_F_MULTI)) {
				DEBUG("msg with unset 'multi' flag");
				return cb(NULL, cb_priv);
			}
		}
	}

err:
	nl_open(nlsock, nlsock->service);
	return -1;
}
