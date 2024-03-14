#ifndef _NLCORE_H
#define _NLCORE_H

#include <syslog.h>
#include <string.h>
#include <errno.h>
#include <linux/netlink.h>

struct nl_sock {
	int sock; /* Socket file descriptor */
	int seq; /* Sequence of sent message */
	int pid; /* port (kernel sock has port=0) */
	int service; /* NETLINK_ROUTE, NETLINK_GENERIC, ... */
};

int nl_open(struct nl_sock *nlsock, int service);
void nl_close(struct nl_sock *nlsock);

char *nlmsg_put_hdr(char *buf, int type, int flags);

int nl_wait_ack(struct nl_sock *nlsock);

int nl_send_msg(struct nl_sock *nlsock, char *buf, int len);
int nl_recv_msg(struct nl_sock *nlsock, int type,
		int (*cb)(struct nlmsghdr *, void *), void *cb_priv);

#define NLMSG_DATA_LEN(nlhdr) ((nlhdr)->nlmsg_len - NLMSG_HDRLEN)

#define ERROR(frmt, ...) nlog(LOG_ERR, "libnel: %s: "frmt, __func__, ##__VA_ARGS__)
#define DEBUG(frmt, ...) nlog(LOG_DEBUG, "libnel: %s: "frmt, __func__, ##__VA_ARGS__)
#define ERRNO(frmt, ...) nlog(LOG_ERR, "libnel: %s: "frmt": %s", __func__, ##__VA_ARGS__, strerror(errno))

void nlog(int lvl, const char *frmt, ...);

#endif
