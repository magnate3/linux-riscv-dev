#include<stdio.h>
#include<stdlib.h>
#include<sys/socket.h>
#include<string.h>
#include<linux/netlink.h>
#include<stdint.h>
#include<unistd.h>
#include<errno.h>

#include "netlink_test.h"
#define NETLINK_TEST	30
#define MSG_LEN 	125
#define MAX_PLOAD	125

typedef struct _user_msg_info {
	struct nlmsghdr hdr;
	char msg[MSG_LEN];
}user_msg_info;

int main(int argc,char **argv) {
	int skfd;
	user_msg_info u_info;
	socklen_t len;
	struct nlmsghdr *nlh=NULL;
	struct sockaddr_nl saddr,daddr;
	char *umsg = "hello netlink!!";
	
	//create NETLINK socket
	skfd = socket(AF_NETLINK,SOCK_RAW,NETLINK_TEST);

	memset(&saddr,0,sizeof(saddr));
	saddr.nl_family = AF_NETLINK;
	saddr.nl_pid = getpid();//user port ID
	//saddr.nl_pid = 100;//user port ID
	saddr.nl_groups= 0;

	//bind
	bind(skfd,(struct sockaddr *)&saddr,sizeof(saddr));

	memset(&daddr,0,sizeof(daddr));
	daddr.nl_family = AF_NETLINK;
	daddr.nl_pid = 0;//kernel port ID
	daddr.nl_groups = 0;

	//nlmsghdr
	nlh = (struct nlmsghdr*)malloc(NLMSG_SPACE(MAX_PLOAD));
	memset(nlh,0,sizeof(struct nlmsghdr));
	nlh->nlmsg_len = NLMSG_SPACE(MAX_PLOAD);
	nlh->nlmsg_flags = 0;
	nlh->nlmsg_type = MSG_TYPE_STRING;
	nlh->nlmsg_seq = 0;
	nlh->nlmsg_pid = saddr.nl_pid;//send user's pid

	//semd message to kernel
	memcpy(NLMSG_DATA(nlh),umsg,strlen(umsg));
	sendto(skfd,nlh,nlh->nlmsg_len,0,(struct sockaddr*)&daddr,sizeof(struct sockaddr_nl));
	printf("send kernel:%s\n",umsg);

	//receive message
	memset(&u_info,0,sizeof(u_info));
	len = sizeof(struct sockaddr_nl);
	recvfrom(skfd,&u_info,sizeof(user_msg_info),0,(struct sockaddr*)&daddr,&len);

	printf("from kernel:%s\n",u_info.msg);

	close(skfd);

	free((void*)nlh);
	return 0;
}
