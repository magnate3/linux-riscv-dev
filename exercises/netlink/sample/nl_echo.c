#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/socket.h>
#include <linux/netlink.h>
#include <errno.h>

#define NETLINK_TEST 30

#define MAX_MSGSIZE 1024

int main(int argc, char *argv[])
{
	struct sockaddr_nl saddr, daddr;
	struct nlmsghdr *nlh = NULL;
	struct msghdr msg;
	struct iovec iov;
	struct timeval to;
	int sd;
	fd_set sds;
	int ret;
	size_t str_len;

	if (argc != 2) {
		printf("Usage: %s string\n", argv[0]);
		return -1;
	}

	str_len = strlen(argv[1]);

	sd = socket(AF_NETLINK, SOCK_RAW, NETLINK_TEST);
	if (sd == -1) {
		printf("socket failed\n");
		return -1;
	}

	memset(&saddr, 0 ,sizeof(struct sockaddr_nl));
	saddr.nl_family = AF_NETLINK;
	saddr.nl_pid = getpid();
	saddr.nl_groups = 0;
	ret = bind(sd, (struct sockaddr *)&saddr, sizeof(struct sockaddr_nl));
	if (ret < 0) {
		printf("bind failed, ret=%d %s\n", ret, strerror(errno));
		close(sd);
		return -1;
	}

	nlh = (struct nlmsghdr *)malloc(NLMSG_SPACE(MAX_MSGSIZE));
	if (nlh == NULL) {
		printf("malloc failed\n");
		close(sd);
		return -1;
	}

	memset(&daddr, 0, sizeof(struct sockaddr_nl));
	daddr.nl_family = AF_NETLINK;
	daddr.nl_pid = 0;
	daddr.nl_groups = 0;
	nlh->nlmsg_len = NLMSG_SPACE(str_len); 
	nlh->nlmsg_pid = getpid();
	nlh->nlmsg_flags = 0;
	strncpy(NLMSG_DATA(nlh), argv[1], str_len);
	iov.iov_base = nlh;
	iov.iov_len = NLMSG_SPACE(str_len);

	memset(&msg, 0, sizeof(struct msghdr));
	msg.msg_name = &daddr;
	msg.msg_namelen = sizeof(struct sockaddr_nl);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	printf("start send msng\n");

	if (sendmsg(sd, &msg, 0) < 0) {
		printf("send failed, errno=%d, %s\n", errno, strerror(errno));
		free(nlh);
		close(sd);
		return -1;
	}

	iov.iov_len = NLMSG_SPACE(MAX_MSGSIZE);

	for (;;) {
		FD_ZERO(&sds);
		FD_SET(sd, &sds);

		to.tv_sec = 10;
		to.tv_usec = 0;
		ret = select(sd + 1, &sds, NULL, NULL, &to);
		if (ret == 0) {
			printf("time out\n");
			break;
		} else if (ret > 0) {
			if (FD_ISSET(sd, &sds)) {
				memset(nlh, 0, NLMSG_SPACE(MAX_MSGSIZE));

				recvmsg(sd, &msg, 0);

				printf("Received message: %s\n",(char *) NLMSG_DATA(nlh));
			}
		} else {
			printf("select failed, ret=%d %s\n", ret, strerror(errno));
			break;
		}
	}


	free(nlh);
	close(sd);

	return 0;
}
