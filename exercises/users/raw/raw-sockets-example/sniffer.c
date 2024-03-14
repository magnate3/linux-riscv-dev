#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<netinet/ip_icmp.h>
#include<netinet/tcp.h>
#include<netinet/udp.h>
#include<arpa/inet.h>
#include<sys/socket.h>
#include<sys/types.h>

#define BUFFSIZE 1024

int main(){

	int rawsock;
	char buff[BUFFSIZE];
	int n;
	int count = 0;

	rawsock = socket(AF_INET,SOCK_RAW,IPPROTO_TCP);
//	rawsock = socket(AF_INET,SOCK_RAW,IPPROTO_UDP);
//	rawsock = socket(AF_INET,SOCK_RAW,IPPROTO_ICMP);
//	rawsock = socket(AF_INET,SOCK_RAW,IPPROTO_RAW);
	if(rawsock < 0){
		printf("raw socket error!\n");
		exit(1);
	}
	while(1){	
		n = recvfrom(rawsock,buff,BUFFSIZE,0,NULL,NULL);
		if(n<0){
			printf("receive error!\n");
			exit(1);
		}

		count++;
		struct ip *ip = (struct ip*)buff;
		unsigned short dst_port;
		memcpy(&dst_port, buff + 22, sizeof(dst_port));
        dst_port = ntohs(dst_port);
		if (5000 == dst_port || 6000 == dst_port)
		{
		printf("%5d	%20s",count,inet_ntoa(ip->ip_src));
		printf("%20s	%5d	%5d and port %d \n",inet_ntoa(ip->ip_dst),ip->ip_p,ntohs(ip->ip_len), dst_port);	
		printf("\n");
		}
	}
}	
