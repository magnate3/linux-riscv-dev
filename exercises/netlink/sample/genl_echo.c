#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <errno.h>
#include <linux/genetlink.h>
#include "my_genl_attr.h"

int send_genl_msg(int fd, int id, unsigned int pid, unsigned char cmd,
		  unsigned char ver, int type, void *data, size_t len)
{
	struct nlattr *na;
	struct sockaddr_nl daddr;
	unsigned char *buf;
	struct nlmsghdr *nlh;
	struct genlmsghdr *gnlh;
	unsigned char *payload;
	ssize_t sent_bytes = 0;
	size_t total_len = 0;

	if (id == 0) {
		return -1;
	}

	buf = (unsigned char *)malloc(NLMSG_SPACE(256));
	if (buf == NULL) {
		printf("malloc failed\n");
		return -1;
	}
	memset(buf, 0 , 256);

	nlh = (struct nlmsghdr *)buf;
	nlh->nlmsg_len = NLMSG_SPACE(GENL_HDRLEN);
	nlh->nlmsg_type = id;
	nlh->nlmsg_flags =  NLM_F_REQUEST;
	nlh->nlmsg_pid = getpid(); /*getpid();*/
	nlh->nlmsg_seq = 0;

	gnlh = (struct genlmsghdr *)NLMSG_DATA(nlh);
	gnlh->cmd = cmd;
	gnlh->version = ver;
	
	na = (struct nlattr *)((unsigned char *)gnlh + GENL_HDRLEN);
	na->nla_type = type;
	na->nla_len = len + NLA_HDRLEN;

	payload = (unsigned char *)((unsigned char *)na + NLA_HDRLEN);
	memcpy(payload, data, len);

	nlh->nlmsg_len += NLMSG_ALIGN(na->nla_len);

	memset(&daddr, 0 , sizeof(struct sockaddr_nl));
	daddr.nl_family = AF_NETLINK;
	daddr.nl_pid = 0;
	daddr.nl_groups = 0;

	total_len = nlh->nlmsg_len;

	printf("%s: len: payload:%lu, +attr:%d, +nlmsg:%d\n",
		__func__, len, na->nla_len, nlh->nlmsg_len);

	
	while ((sent_bytes = sendto(fd, &buf[sent_bytes], total_len, 0,
				    (struct sockaddr *)&daddr,
				    sizeof(struct sockaddr_nl)))
		< total_len) {
		if (sent_bytes > 0) {
			printf("%s: send bytes:%ld\n", __func__, sent_bytes);
			buf += sent_bytes;
			total_len -= sent_bytes;
		} else if (errno != EAGAIN) {

			return -1;
		}
	}

	printf("%s, send bytes: %ld\n", __func__, sent_bytes);

	free(buf);

	return 0;
}

int recv_genl_msg(int family_id, int fd, void *data)
{
	ssize_t recv_bytes = 0;
	unsigned char *buf;
	struct nlmsghdr *nlh;
	struct nlattr *nla;
	int ret;

	buf = (unsigned char *)malloc(256);

	recv_bytes = recv(fd, buf, 256, 0);
	if (recv_bytes < 0) {
		printf("%s: recv failed\n", __func__);
		free(buf);
		return -1;
	}
	printf("%s: recv len is %ld\n", __func__, recv_bytes);

	nlh = (struct nlmsghdr *)buf;
	if (nlh->nlmsg_type == NLMSG_ERROR || !NLMSG_OK(nlh, recv_bytes)) {
		printf("%s: recv len %ld is error\n", __func__, recv_bytes);
		free(buf);
		return -1;
	}
	if (nlh->nlmsg_type == family_id && family_id != 0) {
		nla = (struct nlattr *)(NLMSG_DATA(nlh) + GENL_HDRLEN);
		printf("%s: tlv type:%u, len:%u\n", __func__, nla->nla_type, nla->nla_len);
		strcpy(data, (char *)nla + NLA_HDRLEN);
		ret = 0;
	} else {
		ret = -1;
	}

	free(buf);
	return ret;
}

int get_family_id(int fd, char *name)
{
	int ret = 0;
	struct sockaddr_nl daddr;
	struct msghdr msg;
	struct iovec iov;
	ssize_t recv_len;
	unsigned char *buf;
	struct nlmsghdr *nlh;
	struct nlattr *nla;
	unsigned short id;

	ret = send_genl_msg(fd, GENL_ID_CTRL, 0, CTRL_CMD_GETFAMILY, 1,
			    CTRL_ATTR_FAMILY_NAME, name, strlen(name) + 1);
	if (ret != 0) {
		printf("%s: send genl msg failed, ret=%d\n", __func__, ret);
		return -1;
	}

	printf("%s: send ok\n", __func__);

	buf = (unsigned char *)malloc(256);

	daddr.nl_family = AF_NETLINK;
	daddr.nl_pid = getpid();
	daddr.nl_groups = 0xffffffff;

	iov.iov_base = buf;
	iov.iov_len = 256;

	memset(&msg, 0, sizeof(struct msghdr));
	msg.msg_name = &daddr;
	msg.msg_namelen = sizeof(struct sockaddr_nl);
	msg.msg_iov = &iov;
	msg.msg_iovlen = 1;

	printf("start recv\n");
/*	recv_len = recv(fd, buf, 256, 0);*/
	recv_len = recvmsg(fd, &msg, 0);
	if (recv_len < 0) {
		printf("%s: receive reply failed with %d (%s)\n",
			__func__, errno, strerror(errno));
		free(buf);
		return -1;
	}

	nlh = (struct nlmsghdr *)buf;
	if (nlh->nlmsg_type == NLMSG_ERROR || !NLMSG_OK(nlh, recv_len)) {
		printf("%s: recv len %ld is error\n", __func__, recv_len);
		free(buf);
		return -1;
	}

	nla = (struct nlattr *)(NLMSG_DATA(nlh) + GENL_HDRLEN);
	nla = (struct nlattr *)((unsigned char *)nla + NLA_ALIGN(nla->nla_len));
	if (nla->nla_type == CTRL_ATTR_FAMILY_ID) {
		id = *(unsigned short *)((unsigned char *)nla + NLA_HDRLEN);
		ret = id;
	} else {
		printf("%s: type %u is error\n", __func__, nla->nla_type);
		ret = -1;
	}

	free(buf);

	return ret;
}

int main(int argc, char *argv[])
{
	struct sockaddr_nl saddr;
	int sd;
	int ret;
	int family_id;
	char str[256] = {0};

	sd = socket(AF_NETLINK, SOCK_RAW, NETLINK_GENERIC);
	if (sd == -1) {
		printf("create generic netlink failed with %d (%s)\n", errno, strerror(errno));
		return -1;
	}

	memset(&saddr, 0, sizeof(struct sockaddr_nl));
	saddr.nl_family = AF_NETLINK;
	saddr.nl_pid = getpid();
	saddr.nl_groups = 0;

	printf("start bind\n");
	ret = bind(sd, (struct sockaddr *)&saddr, sizeof(struct sockaddr_nl));
	if (ret < 0) {
		printf("bind failed with %d (%s)\n", errno, strerror(errno));
		close(sd);
		return -1;
	}

	printf("start get family id\n");
	family_id = get_family_id(sd, "my_genl");
	printf("family id=%d\n", family_id);


	sprintf(str, "this is user space");
	if (send_genl_msg(sd, family_id, getpid(), MY_CMD_ECHO, 1, MY_ATTR_MSG, str, strlen(str) + 1) != 0) {
		close(sd);
		return -1;
	}
	memset(str, 0, 256);
	if (recv_genl_msg(family_id, sd, str) == 0) {
		printf("get echo: %s\n", str);
	} else {
		printf("nothing to get\n");
	}

	close(sd);
	return 0;
}
