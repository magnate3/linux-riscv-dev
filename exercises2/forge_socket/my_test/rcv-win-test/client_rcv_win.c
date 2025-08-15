#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>
#define SERVER_PORT 7002
#define CLIENT_PORT 8788
#define SCOPE_LINK 0x20
#define SCOPE_SITE 0x40
#define TCP_SEND_LEN (2048*4)
//const char * ifname ="enp125s0f0"; 
const char * ifname ="enahisic2i0"; 
const char * http ="hello dpdk";
//const char * http ="GET /\r\n";
#define HTTP_REQ_FORMAT         \
    "GET %s HTTP/1.1\r\n"       \
    "User-Agent: dperf\r\n"     \
    "Host: %s\r\n"              \
    "Accept: */*\r\n"           \
    "P: aa\r\n"                 \
    "\r\n"


int main(int argc, char *argv[])
{
	int sock_fd = -1;
	struct sockaddr_in server_addr;
	struct sockaddr_in client_addr;
	struct ifreq ifr;
        int i = 0;
	int opt = 1;
	int ret;
    char buf[TCP_SEND_LEN] ={0}; 
    char ch;
	/* Arguments could be used in getaddrinfo() to get e.g. IP of server */
	(void)argc;
	(void)argv;
 
	/* Create socket for communication with server */
	sock_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (sock_fd == -1) {
		perror("socket()");
		return EXIT_FAILURE;
	}
 
#if 1
        if(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int))<0) {
	    perror("setsockopt() reuse");
	    goto err1;
	}
#else
        if(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int))< 0) {
	    perror("setsockopt() reuse");
	    goto err1;
	}
#endif
	/* Connect to server running on localhost */
	server_addr.sin_family = AF_INET;
    inet_pton(AF_INET, "10.10.16.251", &server_addr.sin_addr);
	server_addr.sin_port = htons(SERVER_PORT);
#if 1
	memset(&ifr, 0, sizeof(ifr));
        strncpy(ifr.ifr_name, ifname, sizeof(ifr.ifr_name));
	if (setsockopt(sock_fd, SOL_SOCKET, SO_BINDTODEVICE, (void *)&ifr, sizeof(ifr)) < 0) {
             perror("Cannot bind socket to device"); 
	     goto err1;
	}
#endif
#if 1
	client_addr.sin_family = AF_INET;
	inet_pton(AF_INET, "10.10.16.81", &client_addr.sin_addr);
	client_addr.sin_port = htons(CLIENT_PORT);
	/* Bind address and socket together */
        ret = bind(sock_fd, (struct sockaddr*)&client_addr, sizeof(client_addr));
	if(ret == -1) {
		perror("bind()");
		goto err1;
	}
#endif
	/* Try to do TCP handshake with server */
	ret = connect(sock_fd, (struct sockaddr*)&server_addr, sizeof(server_addr));
	if (ret == -1) {
		perror("connect()");
		goto err1;
	}
 
	printf("connect to server sussessfully\n");
	/* Send data to server */
	//snprintf(buf, sizeof(buf) -1, HTTP_REQ_FORMAT,http,"ubuntu81");

        for(i=0; i < TCP_SEND_LEN -1; ++i)
        {
            buf[i] = 'a';
        }
        printf("input char   to cause write");
        scanf("%c",&ch);
        while(1)
        {
	    ret = write(sock_fd, buf, sizeof(buf));
	    if (ret == -1) {
	    	perror("write");
	    	close(sock_fd);
	    	return EXIT_FAILURE;
	    }
        sleep(1);
        }
 
	/* DO TCP teardown */
	ret = close(sock_fd);
	if (ret == -1) {
		perror("close()");
		return EXIT_FAILURE;
	}
 
	return EXIT_SUCCESS;
err1:
	close(sock_fd);
        return EXIT_FAILURE;
}
