 
#include <stdio.h>
#include <winsock2.h>

#pragma comment(lib,"ws2_32.lib") 
#pragma warning(disable:4996) 

#define SERVER "210.22.22.151"  // or "localhost" - ip address of UDP server
#define BUFLEN 512  // max length of answer
#define PORT  2666 // the port on which to listen for incoming data
#define true 1
int main(int argc,char *argv[])
{
    char * server_ip ;
    int port;
    int loop = 10;
    // initialise winsock
    WSADATA ws;
    if(argc < 3)
    {
          printf("input server ip and server port \n");
    }
    server_ip = argv[1];
    port  = atoi(argv[2]);
    printf("input server ip %s  and server port %d  \n", server_ip, port);
    printf("Initialising Winsock...");
    if (WSAStartup(MAKEWORD(2, 2), &ws) != 0)
    {
        printf("Failed. Error Code: %d", WSAGetLastError());
        return 1;
    }
    printf("Initialised.\n");

    // create socket
    struct sockaddr_in server;
    int client_socket;
    if ((client_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) == SOCKET_ERROR) // <<< UDP socket
    {
        printf("socket() failed with error code: %d", WSAGetLastError());
        return 2;
    }

    // setup address structure
    memset((char*)&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(port);
    server.sin_addr.S_un.S_addr = inet_addr(server_ip);
    
    // start communication
    while (--loop > 0)
    {
        char * message="hello world from windows";
        printf("send message: %s \n ", message);
        

        // send the message
        if (sendto(client_socket, message, strlen(message), 0, (struct sockaddr*)&server, sizeof(struct sockaddr_in)) == SOCKET_ERROR)
        {
            printf("sendto() failed with error code: %d", WSAGetLastError());
            return 3;
        }

        // receive a reply and print it
        // clear the answer by filling null, it might have previously received data
        char answer[BUFLEN] = {};

        // try to receive some data, this is a blocking call
        int slen = sizeof(struct sockaddr_in);
        int answer_length;
        if (answer_length = recvfrom(client_socket, answer, BUFLEN, 0, (struct sockaddr*)&server, &slen) == SOCKET_ERROR)
        {
            printf("recvfrom() failed with error code: %d", WSAGetLastError());
            exit(0);
        }

        printf("answer: %s \n", answer);
    }

    closesocket(client_socket);
    WSACleanup();
    return 0;
}
