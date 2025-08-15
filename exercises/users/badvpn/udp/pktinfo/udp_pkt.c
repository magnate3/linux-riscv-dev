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
#include <sys/ioctl.h>
#include <sys/time.h>
#define MY_UDP_PORT 5000
#define MY_UDP_IP "10.10.16.82"

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
        
    // Filling server information 
    addr.sin_family = AF_INET; 
    addr.sin_port = htons(MY_UDP_PORT); 
    addr.sin_addr.s_addr = inet_addr(MY_UDP_IP);;
	result = bind(soc, (struct sockaddr*)&addr, sizeof(addr));

	printf("bind result %d port %d\n", result, MY_UDP_PORT);
	if(result < 0){
		printf("error %s\n", strerror(errno));
	}

	do{
		int bytes_received;
		struct sockaddr_in6 from;
		struct iovec iovec[1];
		struct msghdr msg;
		char msg_control[1024];
		char udp_packet[1500];

		iovec[0].iov_base = udp_packet;
		iovec[0].iov_len = sizeof(udp_packet);
		msg.msg_name = &from;
		msg.msg_namelen = sizeof(from);
		msg.msg_iov = iovec;
		msg.msg_iovlen = sizeof(iovec) / sizeof(*iovec);
		msg.msg_control = msg_control;
		msg.msg_controllen = sizeof(msg_control);
		msg.msg_flags = 0;
		printf("ready to recvmsg\n");
		bytes_received = recvmsg(soc, &msg, 0);
		printf("recieved %d bytes\n", bytes_received);

		struct in_pktinfo in_pktinfo;
		int have_in_pktinfo = 0;
		struct cmsghdr* cmsg;

		for (cmsg = CMSG_FIRSTHDR(&msg); cmsg != 0; cmsg = CMSG_NXTHDR(&msg, cmsg))
		{
			printf("asdf\n");
			if (cmsg->cmsg_level == IPPROTO_IP && cmsg->cmsg_type == IP_PKTINFO)
			{
				in_pktinfo = *(struct in_pktinfo*)CMSG_DATA(cmsg);
				have_in_pktinfo = 1;
				printf("recieved ipv4\n");
			}
		}
		//printf("iovlen = %zu \n", msg.msg_iov->iov_len );
		printf("recv msg :%s \n", udp_packet);

		int cmsg_space;
		int ret;
                char buf[64];
		iovec[0].iov_base = udp_packet;
		iovec[0].iov_len = bytes_received;
		msg.msg_name = &from;
		msg.msg_namelen = sizeof(from);
		msg.msg_iov = iovec;
		msg.msg_iovlen = sizeof(iovec) / sizeof(*iovec);
		msg.msg_control = msg_control;
		msg.msg_controllen = sizeof(msg_control);
		msg.msg_flags = 0;
		cmsg_space = 0;
		cmsg = CMSG_FIRSTHDR(&msg);
		if (have_in_pktinfo)
		{
			printf("sent via ipv4\n");
			cmsg->cmsg_level = IPPROTO_IP;
			cmsg->cmsg_type = IP_PKTINFO;
			cmsg->cmsg_len = CMSG_LEN(sizeof(in_pktinfo));
			*(struct in_pktinfo*)CMSG_DATA(cmsg) = in_pktinfo;
			cmsg_space += CMSG_SPACE(sizeof(in_pktinfo));
			//in_pktinfo.ipi_spec_dst.sin_port = htons(5002);
			inet_ntop(AF_INET,&(in_pktinfo.ipi_spec_dst),buf,sizeof (buf));
                        printf("ipi_spec_dst ip:port: %s \n",buf);
			memset(buf, 0, sizeof(buf));
			inet_ntop(AF_INET,&(in_pktinfo.ipi_addr),buf,sizeof (buf));
                        printf("ipi_addr ip:port: %s \n",buf);
                        //printf("ip:port: %s:%d \n",buf, ntohs(in_pktinfo.ipi_spec_dst.sin_port))

		}
		msg.msg_controllen = cmsg_space;
		ret = sendmsg(soc, &msg, 0);
		printf("sent %d bytes\n", ret);
		if(ret < 0){
			printf("error %s\n", strerror(errno));
		}
	}while(1);

	return 0;

}
 
