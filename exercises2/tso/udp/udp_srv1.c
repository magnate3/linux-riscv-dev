#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#define BUFFER_LEN 4096
int main(int argc, char *argv[]) {
	int i = 0;
	// port to start the server on
	int SERVER_PORT = 8877;

	// socket address used for the server
	struct sockaddr_in server_address;
	memset(&server_address, 0, sizeof(server_address));
	server_address.sin_family = AF_INET;

	// htons: host to network short: transforms a value in host byte
	// ordering format to a short value in network byte ordering format
	server_address.sin_port = htons(SERVER_PORT);
        inet_pton(AF_INET, "172.17.242.27", &server_address.sin_addr);
	// htons: host to network long: same as htons but to long
	//server_address.sin_addr.s_addr = htonl(INADDR_ANY);
       
	// create a UDP socket, creation returns -1 on failure
	int sock;
	if ((sock = socket(PF_INET, SOCK_DGRAM, 0)) < 0) {
		printf("could not create socket\n");
		return 1;
	}

	// bind it to listen to the incoming connections on the created server
	// address, will return -1 on error
	if ((bind(sock, (struct sockaddr *)&server_address,
	          sizeof(server_address))) < 0) {
		printf("could not bind socket\n");
                close(sock);
		return 1;
	}

	// socket address used to store client address
	struct sockaddr_in client_address;
	int client_address_len = sizeof(client_address);

	// run indefinitely
	while (true) {
		int count = 0;
		char buffer[BUFFER_LEN];

		memset(buffer, BUFFER_LEN, 0);
		// read content into buffer from an incoming client
		int len = recvfrom(sock, buffer, sizeof(buffer), 0,
		                   (struct sockaddr *)&client_address,
		                   &client_address_len);

		for(i = 0; i < len; ++ i)
		{
		    if('a' == buffer[i]) 
			++ count; 
		}
		// inet_ntoa prints user friendly representation of the
		// ip address
#if 0
		buffer[len] = '\0';
		printf("received: '%s' from client %s\n", buffer,
		       inet_ntoa(client_address.sin_addr));
#else
		printf("recv number of a:  %d \n",count);
#endif
		// send same content back to the client ("echo")

#if 0
		sendto(sock, buffer, len, 0, (struct sockaddr *)&client_address,
		       sizeof(client_address));
#endif
	}
        close(sock);
	return 0;
}
