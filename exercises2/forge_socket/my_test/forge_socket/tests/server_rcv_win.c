#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netinet/in.h>
#include "forge_socket.h"
#define CLIENT_QUEUE_LEN 10
#define SERVER_PORT 7002
#define TCP_SEND_LEN (2048*4)
const char * ifname ="enahisic2i0";
char html[] =
"HTTP/1.1 200 OK\r\n"
"Server: F-Stack\r\n"
"Date: Sat, 25 Feb 2017 09:26:33 GMT\r\n"
"Content-Type: text/html\r\n"
"Content-Length: 438\r\n"
"Last-Modified: Tue, 21 Feb 2017 09:44:03 GMT\r\n"
"Connection: keep-alive\r\n"
"Accept-Ranges: bytes\r\n"
"\r\n"
"<!DOCTYPE html>\r\n"
"<html>\r\n"
"<head>\r\n"
"<title>Welcome to F-Stack!</title>\r\n"
"<style>\r\n"
"    body {  \r\n"
"        width: 35em;\r\n"
"        margin: 0 auto; \r\n"
"        font-family: Tahoma, Verdana, Arial, sans-serif;\r\n"
"    }\r\n"
"</style>\r\n"
"</head>\r\n"
"<body>\r\n"
"<h1>Welcome to F-Stack!</h1>\r\n"
"\r\n"
"<p>For online documentation and support please refer to\r\n"
"<a href=\"http://F-Stack.org/\">F-Stack.org</a>.<br/>\r\n"
"\r\n"
"<p><em>Thank you for using F-Stack.</em></p>\r\n"
"</body>\r\n"
"</html>";
void print_state(struct tcp_state *st)
{
    printf("\tsrcip: %x:%d\n\tdstip:%x:%d\n", \
           ntohl(st->src_ip), ntohs(st->sport), \
           ntohl(st->dst_ip), ntohs(st->dport));
    printf("\tseq: %u\n\tack:%u\n", st->seq, st->ack);
    //printf("\tseq: 0x%x\n\tack:0x%x\n", st->seq, st->ack);
    //printf("\tsnd_una: %x\n", st->snd_una);
    printf("\tsnd_una: %u, snd_nxt: %u, snd_wnd: %u\n", st->snd_una, st->snd_nxt, st->snd_wnd);
    printf("\trcv_wup: %u, rcv_nxt %u, rcv_wnd: %u\n", st->rcv_wup, st->rcv_nxt, st->rcv_wnd);
    printf("\ttstamp_ok: %d\n", st->tstamp_ok);
    printf("\tsack_ok: %d\n", st->sack_ok);
    printf("\twscale_ok: %d\n", st->wscale_ok);
    printf("\tecn_ok: %d\n", st->ecn_ok);
    printf("\tsnd_wscale: %d\n", st->snd_wscale);
    printf("\trcv_wscale: %d\n", st->rcv_wscale);
    printf("\tinet_ttl: %d\n", st->inet_ttl);
}
int main(void)
{
    int listen_sock_fd = -1, client_sock_fd = -1;
    int recvbuf = TCP_SEND_LEN;
    struct tcp_state state;
    int len = sizeof(state);
    socklen_t optlen=sizeof(recvbuf);
	struct sockaddr_in server_addr, client_addr;
	socklen_t client_addr_len;
	char str_addr[INET_ADDRSTRLEN];
	int ret, flag;
	char buf[2056]={0};
 
	/* Create socket for listening (client requests) */
#if 0
	listen_sock_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
#else
        listen_sock_fd = socket(AF_INET, SOCK_FORGE, 0);
#endif
	if(listen_sock_fd == -1) {
		perror("socket()");
		return EXIT_FAILURE;
	}
 
	/* Set socket to reuse address */
	flag = 1;
	ret = setsockopt(listen_sock_fd, SOL_SOCKET, SO_REUSEADDR, &flag, sizeof(flag));
	if(ret == -1) {
		perror("setsockopt()");
		return EXIT_FAILURE;
	}
        ret = setsockopt(listen_sock_fd, SOL_SOCKET,  SO_RCVBUF, (void *)&recvbuf, (socklen_t )optlen); 
	if(ret == -1) {
		perror("setsockopt() recvbuf");
		return EXIT_FAILURE;
	}
	server_addr.sin_family = AF_INET;
    inet_pton(AF_INET, "10.10.16.251", &server_addr.sin_addr);
	server_addr.sin_port = htons(SERVER_PORT);
	/* Bind address and socket together */
	ret = bind(listen_sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
	if(ret == -1) {
		perror("bind()");
		close(listen_sock_fd);
		return EXIT_FAILURE;
	}
 
	/* Create listening queue (client requests) */
	ret = listen(listen_sock_fd, CLIENT_QUEUE_LEN);
	if (ret == -1) {
		perror("listen()");
		close(listen_sock_fd);
		return EXIT_FAILURE;
	}
 
	client_addr_len = sizeof(client_addr);
		/* Do TCP handshake with client */
		client_sock_fd = accept(listen_sock_fd,
				(struct sockaddr*)&client_addr,
				&client_addr_len);
		if (client_sock_fd == -1) {
			perror("accept()");
			close(listen_sock_fd);
			return EXIT_FAILURE;
		}
 
		inet_ntop(AF_INET, &(client_addr.sin_addr),
				str_addr, sizeof(str_addr));
		printf("New connection from: %s:%d ...\n",
				str_addr,
				ntohs(client_addr.sin_port));
                while(true){
		    ret = read(client_sock_fd, buf, sizeof(buf));
		    if (ret == -1) {
		    	perror("read()");
		    	close(client_sock_fd);
                        break;
		    }
                    ret = getsockopt(client_sock_fd, IPPROTO_TCP, TCP_STATE, &state, &len);
                    if (ret != 0) {
                        perror("getsockopt tcp state");
		        close(client_sock_fd);
                        break;
                    }
                    print_state(&state);
                }
		/* Do TCP teardown */
		ret = close(client_sock_fd);
		if (ret == -1) {
			perror("close()");
			client_sock_fd = -1;
		}
 
	printf("Connection closed\n");
	close(listen_sock_fd);
	return EXIT_SUCCESS;
}
