#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netinet/in.h>
 
#define CLIENT_QUEUE_LEN 10
#define TCP_BUF_LEN (2048)
#define SERVER_PORT 22222
const char * ifname ="enahisic2i0";
#if 0
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
#else
//char html[] ="****hello dpdk****";
char* html ="hello dpdk";
#endif
int main(void)
{
	int listen_sock_fd = -1, client_sock_fd = -1;
	struct sockaddr_in server_addr, client_addr;
	socklen_t client_addr_len;
	char str_addr[INET_ADDRSTRLEN];
	int ret, flag;
	char buf[TCP_BUF_LEN]={0};
    int i = 0,count=0; 
	/* Create socket for listening (client requests) */
	listen_sock_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
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

	server_addr.sin_family = AF_INET;
    inet_pton(AF_INET, "10.10.103.81", &server_addr.sin_addr);
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
 
	while(1) {
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
		/* Send response to client */
		//ret = write(client_sock_fd, html, sizeof(html));
		ret = write(client_sock_fd, html, strlen(html) + 1);
		if (ret == -1) {
			perror("write()");
			close(client_sock_fd);
			continue;
		}
 
#if 1 
		/* Wait for data from client */
		ret = read(client_sock_fd, buf, sizeof(buf));
		if (ret == -1) {
			perror("read()");
			close(client_sock_fd);
			continue;
		}
        printf("recv %s \n", buf);
        for(i=0; i < ret; ++i){
            if('A' == buf[i]){
                ++ count;
            }
        }
        printf("recv number of  A  : %d \n",count);
#endif
		/* Do very useful thing with received data :-) */
		/* Do TCP teardown */
		ret = close(client_sock_fd);
		if (ret == -1) {
			perror("close()");
			client_sock_fd = -1;
		}
 
		printf("Connection closed\n");
	}
	return EXIT_SUCCESS;
}
