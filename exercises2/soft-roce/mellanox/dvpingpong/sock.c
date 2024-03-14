// Server side C/C++ program to demonstrate Socket programming
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <errno.h>

#define PORT 8668

int sock_server(char *sendbuf, int send_buflen, char *recvbuf, int recv_buflen)
{
	int server_fd, new_socket, sent, readlen;
	struct sockaddr_in address;
	int opt = 1;
	int addrlen = sizeof(address);
	//char buffer[1024] = { 0 };
	//char *hello = "Hello from server";

	// Creating socket file descriptor
	if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
		perror("socket failed");
		exit(EXIT_FAILURE);
	}
	// Forcefully attaching socket to the port 8080
	if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT,
		       &opt, sizeof(opt))) {
		perror("setsockopt");
		exit(EXIT_FAILURE);
	}
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = INADDR_ANY;
	address.sin_port = htons(PORT);

	// Forcefully attaching socket to the port 8080
	if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
		perror("bind failed");
		exit(EXIT_FAILURE);
	}
	if (listen(server_fd, 3) < 0) {
		perror("listen");
		exit(EXIT_FAILURE);
 	}
	printf("    =DEBUG:%s:%d: waiting for client:\n", __func__, __LINE__);
	if ((new_socket = accept(server_fd, (struct sockaddr *)&address,
				 (socklen_t *) & addrlen)) < 0) {
		perror("accept");
		exit(EXIT_FAILURE);
	}

	readlen = read(new_socket, recvbuf, recv_buflen);
	if (readlen <= 0) {
		printf("=DEBUG:%s:%d: read %d expected %d errno %d\n", __func__, __LINE__,
		       readlen, recv_buflen, errno);
		return -1;
	}

	sent = send(new_socket, sendbuf, send_buflen, 0);
	if (sent < 0) {
		printf("=DEBUG:%s:%d: sent %d expected %d errno %d\n", __func__, __LINE__,
		       sent, send_buflen, 0);
		return -1;
	}

	//printf("    =DEBUG:%s:%d: Server info exchange done(recv_buflen %d readlen %d send_buflen %d sent %d)\n", __func__, __LINE__, recv_buflen, readlen, send_buflen, sent);
	return 0;
}

int sock_client(const char *server_ip, char *sendbuf, int send_buflen,
		char *recvbuf, int recv_buflen)
{
	//struct sockaddr_in address;
	int sock = 0, sent, readlen;
	struct sockaddr_in serv_addr;

	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		printf("\n Socket creation error \n");
		return -1;
	}

	memset(&serv_addr, '0', sizeof(serv_addr));

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(PORT);

	// Convert IPv4 and IPv6 addresses from text to binary form
	if (inet_pton(AF_INET, server_ip, &serv_addr.sin_addr) <= 0) {
		printf("\nInvalid address/ Address not supported \n");
		return -1;
	}

	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
		printf("\nConnection Failed \n");
		return -1;
	}

	sent = send(sock, sendbuf, send_buflen, 0);
	if (sent <= 0) {
		printf("=DEBUG:%s:%d: sent %d expected %d errno %d\n", __func__, __LINE__,
		       sent, send_buflen, 0);
		return -1;
	}

	readlen = read(sock, recvbuf, recv_buflen);
	if (readlen <= 0) {
		printf("=DEBUG:%s:%d: read %d expected %d errno %d\n", __func__, __LINE__,
		       readlen, recv_buflen, errno);
		return -1;
	}

	//printf("    =DEBUG:%s:%d: Client info exchange done(recv_buflen %d readlen %d send_buflen %d sent %d)\n", __func__, __LINE__, recv_buflen, readlen, send_buflen, sent);
	return 0;
}
