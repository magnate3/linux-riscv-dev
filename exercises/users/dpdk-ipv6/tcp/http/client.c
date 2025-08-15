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
#define TEST_DPDK251 1 
#if !TEST_DPDK251
#define SERVER_PORT 7002
#else
#define SERVER_PORT 80
#endif
#define CLIENT_PORT 8788
#define SCOPE_LINK 0x20
#define SCOPE_SITE 0x40
const char * ifname ="enahisic2i3"; 
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
	struct sockaddr_in6 server_addr;
	struct sockaddr_in6 client_addr;
	struct ifreq ifr;
	int opt = 1;
	int ret;
        char buf[1024] ={0}; 
	/* Arguments could be used in getaddrinfo() to get e.g. IP of server */
	(void)argc;
	(void)argv;
 
	/* Create socket for communication with server */
	sock_fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
	if (sock_fd == -1) {
		perror("socket()");
		return EXIT_FAILURE;
	}
 
	if (setsockopt(sock_fd, IPPROTO_IPV6, IPV6_V6ONLY, &opt, sizeof(int)) < 0) {
             perror("Cannot ipv6 only"); 
	     goto err1;
	}
#if 1
        if(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(int))<0) {
	    perror("setsockopt() reuse");
	    goto err1;
	}
//#else
        if(setsockopt(sock_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(int))< 0) {
	    perror("setsockopt() reuse");
	    goto err1;
	}
#endif
	/* Connect to server running on localhost */
	server_addr.sin6_family = AF_INET6;
	server_addr.sin6_scope_id = if_nametoindex(ifname);
	//server_addr.sin6_scope_id = SCOPE_LINK;
#if !TEST_DPDK251
	// 81 host
	inet_pton(AF_INET6, "fe80::4a57:2ff:fe64:ea1e", &server_addr.sin6_addr);
#else
        inet_pton(AF_INET6, "fe80::4a57:2ff:fe64:e7a7", &server_addr.sin6_addr);
#endif
	//inet_pton(AF_INET6, "::1", &server_addr.sin6_addr);
	server_addr.sin6_port = htons(SERVER_PORT);
#if 1
	memset(&ifr, 0, sizeof(ifr));
        strncpy(ifr.ifr_name, ifname, sizeof(ifr.ifr_name));
	if (setsockopt(sock_fd, SOL_SOCKET, SO_BINDTODEVICE, (void *)&ifr, sizeof(ifr)) < 0) {
             perror("Cannot bind socket to device"); 
	     goto err1;
	}
#endif
#if 1
	client_addr.sin6_family = AF_INET6;
	inet_pton(AF_INET6, "fe80::4a57:2ff:fe64:e7ae", &client_addr.sin6_addr);
	client_addr.sin6_port = htons(CLIENT_PORT);
	client_addr.sin6_scope_id = if_nametoindex(ifname);
	//client_addr.sin6_scope_id = SCOPE_LINK;
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
	 snprintf(buf, sizeof(buf) -1, HTTP_REQ_FORMAT,http,"ubuntu81");
	ret = write(sock_fd, http, strlen(http));
	if (ret == -1) {
		perror("write");
		close(sock_fd);
		return EXIT_FAILURE;
	}
 
	printf("send  %s to server\n", http);
	/* Wait for data from server */
	ret = read(sock_fd, buf, sizeof(buf));
	if (ret == -1) {
		perror("read()");
		close(sock_fd);
		return EXIT_FAILURE;
	}
 
	printf("Received %s from server\n", buf);
 
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
