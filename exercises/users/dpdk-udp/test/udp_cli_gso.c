
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <netinet/if_ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <sys/uio.h>
#define TCP_SEND_LEN (2048*2 + 2048)
//#define TCP_SEND_LEN (2048*3)
//#define ETH_DATA_LEN 1514
#ifndef UDP_SEGMENT
#define UDP_SEGMENT		103
#endif
static int set_pmtu_discover(int fd, bool is_ipv4)
{
	int level, name, val;
	level	= SOL_IP;
	name	= IP_MTU_DISCOVER;
        val	= IP_PMTUDISC_DO;
	if (setsockopt(fd, level, name, &val, sizeof(val)))
	{
	    perror("set pmtu");
	    return -1;
	}
	return 0;
}
int main(int argc,char* argv[]) {
	int i = 0,sent;
    int mode = 0;
	int ret = 0;
	int gso_size = ETH_DATA_LEN - sizeof(struct iphdr) - sizeof(struct udphdr);
	const int server_port = 10001;
    char buf[TCP_SEND_LEN] ={0}; 
	struct sockaddr_in server_address;
    if(argc < 2){
        printf("nee more parms \n");
        exit(0);
    }
    mode = atoi(argv[1]);
	memset(&server_address, 0, sizeof(server_address));
	server_address.sin_family = AF_INET;

	// creates binary representation of server name
	// and stores it as sin_addr
	// http://beej.us/guide/bgnet/output/html/multipage/inet_ntopman.html
        inet_pton(AF_INET, "10.10.103.251", &server_address.sin_addr);
	// htons: port in network order format
	server_address.sin_port = htons(server_port);

	// open socket
	int sock;
	if ((sock = socket(PF_INET, SOCK_DGRAM, 0)) < 0) {
		printf("could not create socket\n");
		return 1;
	}
    uint8_t msg_ctrl[32];
    struct cmsghdr *cm;
    struct iovec iov;
    switch (mode) {
       case 0:
	       gso_size -= 200;
	       printf("gso size %d \n",gso_size);
               for(i=0; i < TCP_SEND_LEN; ++i)
               {
                   buf[i] = 'a';
               }
	       iov.iov_base = buf;
	       iov.iov_len = TCP_SEND_LEN;
	       struct msghdr msg = {0};
	       msg.msg_iov = &iov;
	       msg.msg_iovlen = 1;
	       msg.msg_name = &server_address;
	       msg.msg_namelen = sizeof(server_address);
	       if(TCP_SEND_LEN> gso_size)
	       {
	                msg.msg_control = msg_ctrl;
                   assert(sizeof(msg_ctrl) >= CMSG_SPACE(sizeof(uint16_t)));
                   msg.msg_controllen = CMSG_SPACE(sizeof(uint16_t));
                   cm = CMSG_FIRSTHDR(&msg);
                   cm->cmsg_level = SOL_UDP;
                   cm->cmsg_type = UDP_SEGMENT;
                   cm->cmsg_len = CMSG_LEN(sizeof(uint16_t));
                   *(uint16_t *)(void *)CMSG_DATA(cm) = gso_size& 0xffff;
                   sent = sendmsg(sock, &msg, 0);
            }   
            break;
        case 1:
            for(i=0; i < TCP_SEND_LEN; ++i)
            {
                buf[i] = 'a';
            }
            sendto(sock, buf, sizeof(buf), 0,(struct sockaddr*)&server_address, sizeof(server_address));
            break;
        default:
            break;
    }
error:
	// close the socket
	close(sock);
	return 0;
}
