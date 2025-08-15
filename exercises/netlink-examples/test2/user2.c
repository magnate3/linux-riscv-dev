#include<stdio.h>
#include<stdlib.h>
#include<sys/socket.h>
#include<string.h>
#include<linux/netlink.h>
#include<stdint.h>
#include<unistd.h>
#include<errno.h>
#include "nlcore.h"
#include "netlink_test.h"
#define MSG_LEN 	125
#define MAX_PLOAD	125

typedef struct _user_msg_info {
	struct nlmsghdr hdr;
	char msg[MSG_LEN];
}user_msg_info;
#if 0
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
	saddr.nl_pid = 100;//user port ID
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
	nlh->nlmsg_type = 0;
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
#else
#define TEST_OP 0
struct Test_data {
    int val;
    char name[64];
};
int nlr_init(struct nl_sock * nlsock)
{
        /* Only the first call actually inits. */
        if (nl_open(nlsock, NETLINK_TEST))
        {
                        return -1;
        }
        return 0;
}

void nlr_fin(struct nl_sock * nlsock)
{
     nl_close(nlsock);
}

int main(int argc,char **argv) {
     struct nl_sock nlsock;
     //char buf[64] = "hello to kernel\n"; 
     char buf[128],  *p;;
     struct Test_data data;
     struct shared_msg_err err;
     int total, total2;
     
     err.number = 999;
     err.critical = 1;
     nlr_init(&nlsock);
     memset(buf, 0, sizeof(buf));
 
#if 0
     memset(&data, 0, sizeof(struct Test_data));
     data.val = 99;
     memcpy(data.name, "dirk", strlen("dirk"));
     p = nlmsg_put_hdr(buf, MSG_TYPE_TEST, 0);
     //p = nlmsg_put_hdr(buf, MSG_TYPE_TEST, NLM_F_ACK);
     p = pad_data(p, &data, sizeof(struct Test_data));
#else
     struct nlmsghdr * nlhdr;
     const char * name1 ="baide";
     const char * name2 ="apple";
     p=nl_pad_str_msg(&nlsock, name1, strlen(name1), buf,  MSG_TYPE_STRING, 0);
     p=nl_pad_str_msg(&nlsock, name2, strlen(name2), p,  MSG_TYPE_STRING, 0);
     p=nl_pad_msg(&nlsock, &err, sizeof(err), p,  MSG_TYPE_ERROR, 0);
     total2 = total = p - buf;
     printf("*********** totalmsg len %d \n", total);
     for (nlhdr = (struct nlmsghdr *)buf; NLMSG_OK(nlhdr, total); nlhdr = NLMSG_NEXT(nlhdr, total)) {
     printf("get new msg: len=%d, type=0x%02x, pid = %d  \n", nlhdr->nlmsg_len, nlhdr->nlmsg_type, nlhdr->nlmsg_pid);

        switch (nlhdr->nlmsg_type) {
        case MSG_TYPE_STRING:
         	printf(" msg: %s \n", NLMSG_DATA(nlhdr));
                break;
        case MSG_TYPE_ERROR: {
         	struct shared_msg_err * errmsg = (struct shared_msg_err *)NLMSG_DATA(nlhdr);
         	printf("err msg: error=%d \n", errmsg->number);
                break;
        }
        default:
                printf("Unhandled message type: %d\n", nlhdr->nlmsg_len);
                break;
        }
     }
     nl_send_msg_simple(&nlsock, buf, total2);
     nl_recv_msg_simple(&nlsock, 3, NULL, NULL);
#endif
     nlr_fin(&nlsock);
}
#endif
