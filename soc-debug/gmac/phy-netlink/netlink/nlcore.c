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
#include "netlink_test.h"

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
	sa.nl_pid = getpid(); 
	//sa.nl_pid = getpid(); 
	sa.nl_groups = 0; 
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
void nl_pad_msg_hdr(struct nl_sock* nlsock, struct nlmsghdr *nlhdr, int type, int flags)
{
	nlhdr->nlmsg_type = type;
	nlhdr->nlmsg_flags = flags | NLM_F_REQUEST;
	nlhdr->nlmsg_pid = nlsock->pid;
	nlhdr->nlmsg_seq = ++nlsock->seq;
}
char * nl_pad_str_msg(struct nl_sock* nlsock, const  void * data,const int data_len, char *buf, int type, int flags)
{
	struct nlmsghdr *nlhdr = (struct nlmsghdr *)buf;
        void *ptr;
        //NLMSG_DATA use NLMSG_LENGTH(0)
        const size_t real_len =  NLMSG_LENGTH(0) + data_len +1;
        /* We need NLMSG_SPACE to calculate how much space we must allocate strlen + 1 */
        size_t nlmsg_len = NLMSG_SPACE(sizeof(data_len +1));
        if (nlmsg_len  <  real_len)
        {
              nlmsg_len = real_len ;
        }
	memset(buf, 0, NLMSG_HDRLEN);
	nlhdr->nlmsg_len = nlmsg_len;
        ptr = NLMSG_DATA(nlhdr);
        memcpy(ptr, data, data_len);
        memset(ptr + data_len, 0, 1);
        nl_pad_msg_hdr(nlsock, nlhdr, type, flags);
#if 0
        printf("real_len : %d, nlmsg_len %d \n ", real_len, nlmsg_len);
        printf("ptr is %s and ptr end : %x \n ", ptr, ptr + data_len);
        printf("begin %x and end %x \n", buf, (void*)buf + nlmsg_len); 
#endif
        // NLMSG_NEXT
	return  (void*)buf +  NLMSG_ALIGN(nlmsg_len);
}
char * nl_pad_msg(struct nl_sock* nlsock, const  void * data,const int data_len, char *buf, int type, int flags)
{
	struct nlmsghdr *nlhdr = (struct nlmsghdr *)buf;
        void *ptr;
        const size_t real_len =  NLMSG_LENGTH(0) + data_len;
        /* We need NLMSG_SPACE to calculate how much space we must allocate */
        size_t nlmsg_len = NLMSG_SPACE(sizeof(data_len));
        if (nlmsg_len  <  real_len)
        {
              nlmsg_len = real_len ;
        }
        ptr = NLMSG_DATA(nlhdr);
	memset(buf, 0, NLMSG_HDRLEN);
	nlhdr->nlmsg_len = nlmsg_len;
        memcpy(ptr, data, data_len);
        nl_pad_msg_hdr(nlsock, nlhdr, type, flags);
#if 0
        printf("real_len : %d, nlmsg_len %d \n ", real_len, nlmsg_len);
        printf("ptr is %x and ptr end : %x \n ", ptr, ptr + data_len -1);
        printf("begin %x and end %x \n", buf, (void*)buf + nlmsg_len); 
#endif
        // NLMSG_NEXT
	return  (void*)buf +  NLMSG_ALIGN(nlmsg_len);
}
int nl_send_msg_simple(struct nl_sock *nlsock, char *buf, int len)
{
	struct sockaddr_nl sa;
	int n;

	memset(&sa, 0, sizeof(sa));
	sa.nl_family = AF_NETLINK;
	sa.nl_pid = 0; /* To kernel */
	sa.nl_groups = 0; /* Unicast */
	//n = NLMSG_ALIGN(len);
        n = len;
	if (sendto(nlsock->sock, buf, n, 0, (struct sockaddr *)&sa, sizeof(sa)) != n) {
		ERRNO("failed to send");
		return -1;
	}

	DEBUG("send %d bytes", n);

	return 0;
}
int nl_recv_msg_simple(struct nl_sock *nlsock, int count, int (*cb)(struct nlmsghdr *, void *), void *cb_priv)
{
	struct sockaddr_nl sa;
        struct   phy_op_t * phy;
	int n;
	char buf[4096];
	struct nlmsghdr *nlhdr;
	sa.nl_family = AF_NETLINK;
	sa.nl_pid = USER_PORT; /* To kernel */
	sa.nl_groups = 0; /* Unicast */
	n = sizeof(sa);
        while(--count >= 0)
        {
	    n = recvfrom(nlsock->sock, buf, sizeof(buf), 0, (struct sockaddr *)&sa, &n);
       	    if (n < 0) {
	    	if (errno == EINTR || errno == EAGAIN)
            	printf("failed to recv");
	    	goto err;
	    }

             for (nlhdr = (struct nlmsghdr *)buf; NLMSG_OK(nlhdr, n); nlhdr = NLMSG_NEXT(nlhdr, n)) {
             //printf("get new msg from kernel : len=%d, type=0x%02x \n", nlhdr->nlmsg_len, nlhdr->nlmsg_type);

             switch (nlhdr->nlmsg_type) {
             case MSG_TYPE_STRING:
              	printf(" msg: %s \n", (char*)NLMSG_DATA(nlhdr));
                     break;
             case MSG_TYPE_PHY: 
              	phy = (struct phy_op_t *)NLMSG_DATA(nlhdr);
	        printf("get new msg from kernel : dev name %s ,op %d,  page 0x%x, reg: 0x%x, val: 0x%x \n", phy->name,phy->op,phy->page, phy->reg, phy->val);
                     break;
             default:
                     printf("Unhandled message type: %d\n", nlhdr->nlmsg_len);
                     break;
             }
    
      }
     }

err:
	return -1;
}
