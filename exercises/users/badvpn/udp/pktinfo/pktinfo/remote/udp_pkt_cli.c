#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <string.h>
#include <netinet/ip6.h>
#include <errno.h>
#include <net/if.h>
#include <linux/sockios.h>
#include <linux/in.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#define MY_UDP_PORT 5005
#define MY_UDP_IP "10.10.16.251"

#define SRV_UDP_PORT 5000
#define SRV_UDP_IP "10.10.16.82"

int main(void)
{
	const int on=1, off=0;
	int result;
	int soc;
        struct sockaddr_in     addr; 
	soc = socket(AF_INET, SOCK_DGRAM, 0);
	setsockopt(soc, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
	setsockopt(soc, IPPROTO_IP, IP_PKTINFO, &on, sizeof(on));
	memset(&addr, 0, sizeof(addr)); 
        
        addr.sin_family = AF_INET; 
        addr.sin_port = htons(MY_UDP_PORT); 
        addr.sin_addr.s_addr = inet_addr(MY_UDP_IP);;
	result = bind(soc, (struct sockaddr*)&addr, sizeof(addr));

	printf("bind result %d port %d\n", result, MY_UDP_PORT);
	if(result < 0){
		printf("error %s\n", strerror(errno));
	}

    union {
        char in[CMSG_SPACE(sizeof(struct in_pktinfo))];
    } cdata;
		int ret;
		struct iovec iovec[1];
		struct msghdr msg;
		char * udp_packet="hello from udp sendmsg";
                struct sockaddr_in     servaddr;
                struct cmsghdr *cmsg ;
                size_t controllen = 0;
                char buf[64];
                servaddr.sin_family = AF_INET;
                servaddr.sin_port = htons(SRV_UDP_PORT);
                servaddr.sin_addr.s_addr = inet_addr(SRV_UDP_IP);;
#if 0
           if(connect(soc, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0)
           {
                       printf("\n Error : Connect Failed \n");
                       close(soc);
                       exit(0);
           }
#endif
		iovec[0].iov_base = udp_packet;
		iovec[0].iov_len = sizeof(udp_packet);
		msg.msg_name = &servaddr;
		msg.msg_namelen = sizeof(servaddr);
		msg.msg_iov = iovec;
		msg.msg_iovlen = sizeof(iovec) / sizeof(*iovec);
		msg.msg_flags = 0;
#if 1
	msg.msg_control = &cdata;
        cmsg = CMSG_FIRSTHDR(&msg);
	cmsg->cmsg_level = IPPROTO_IP;
	cmsg->cmsg_type = IP_PKTINFO;
	cmsg->cmsg_len = CMSG_LEN(sizeof(struct in_pktinfo));
        struct in_pktinfo *pktinfo = (struct in_pktinfo *)CMSG_DATA(cmsg);
        pktinfo->ipi_spec_dst.s_addr = servaddr.sin_addr.s_addr ; 
        controllen += CMSG_SPACE(sizeof(struct in_pktinfo));
	msg.msg_controllen = controllen;
#endif
#if 0
	*(struct in_pktinfo*)CMSG_DATA(cmsg) = in_pktinfo;
	cmsg_space += CMSG_SPACE(sizeof(in_pktinfo));
	//in_pktinfo.ipi_spec_dst.sin_port = htons(5002);
	inet_ntop(AF_INET,&(in_pktinfo.ipi_spec_dst),buf,sizeof (buf));
        printf("ipi_spec_dst ip:port: %s \n",buf);
	memset(buf, 0, sizeof(buf));
	inet_ntop(AF_INET,&(in_pktinfo.ipi_addr),buf,sizeof (buf));
        printf("ipi_addr ip:port: %s \n",buf);
        //printf("ip:port: %s:%d \n",buf, ntohs(in_pktinfo.ipi_spec_dst.sin_port))
#endif
		ret = sendmsg(soc, &msg, 0);
		printf("sent %d bytes\n", ret);
		if(ret < 0){
			printf("error %s\n", strerror(errno));
		}

	return 0;

}
 
