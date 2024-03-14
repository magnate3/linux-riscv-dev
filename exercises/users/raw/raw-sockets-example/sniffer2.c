#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<netinet/ip_icmp.h>
#include<netinet/tcp.h>
#include<netinet/udp.h>
#include<arpa/inet.h>
#include<sys/socket.h>
#include<sys/types.h>
#include <errno.h>

#define BUFFSIZE 1024
#define PORT 50000
#define PORT_CLIENT 50001
#define SERVER_ADDR "10.10.16.81"
#define CLIENT_ADDR "10.10.16.82"
int main(){

        int rawsock;
        char buff[BUFFSIZE];
        int n;
        int count = 0;
        struct sockaddr_in *servaddr = NULL, *client_addr = NULL;
        rawsock = socket(AF_INET,SOCK_RAW,IPPROTO_TCP);
		if(rawsock < 0){
                printf("raw socket error!\n");
                exit(1);
        }
        servaddr = (struct sockaddr_in *)malloc(sizeof(struct sockaddr_in));
        if (servaddr == NULL) {
                printf("could not allocate memory\n");
                goto end;
        }

        servaddr->sin_family = AF_INET;
        servaddr->sin_port = PORT;
        servaddr->sin_addr.s_addr = inet_addr(SERVER_ADDR);

        /* Part 2 – fill data structure and bind to socket */
        if (0 != (bind(rawsock, (struct sockaddr *)servaddr, sizeof(struct sockaddr_in)))) {
                printf("could not bind server socket to address\n");
                goto end1;
        }

        /* part 3: read and write data */
        client_addr = (struct sockaddr_in *)malloc(sizeof(struct sockaddr_in));
        if (client_addr == NULL) {
                printf("Unable to allocate memory to client address socket\n");
                goto end2;
        }

        client_addr->sin_family = AF_INET;
        client_addr->sin_port = PORT_CLIENT;
        client_addr->sin_addr.s_addr = inet_addr(CLIENT_ADDR);

		int error =0;
        error = connect(rawsock, (struct sockaddr *)client_addr, sizeof(struct sockaddr_in));
        if (error != 0) {
                printf("error %d", errno);
                printf("connect returned error\n");
                goto end2;
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
                printf("%5d     %20s",count,inet_ntoa(ip->ip_src));
                printf("%20s    %5d     %5d and port %d \n",inet_ntoa(ip->ip_dst),ip->ip_p,ntohs(ip->ip_len), dst_port);
                printf("\n");
                }
        }
end2:
		        free(client_addr);
end1:
				free(servaddr);
end:
				close(rawsock);
}
