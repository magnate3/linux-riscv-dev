 
#include <stdio.h>
#include <winsock2.h>

#pragma comment(lib,"ws2_32.lib") 
#pragma warning(disable:4996) 

#define SERVER "210.22.22.151"  // or "localhost" - ip address of UDP server
#define BUFLEN 512  // max length of answer
#define PORT  22 // the port on which to listen for incoming data
//#define PORT  2666 // the port on which to listen for incoming data
#define true 1
int main()
{
    int loop = 10;
    int ret = 0;
    // initialise winsock
    WSADATA ws;
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
    if ((client_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) == SOCKET_ERROR) // <<< UDP socket
    {
        printf("socket() failed with error code: %d", WSAGetLastError());
        return 2;
    }

    // setup address structure
    memset((char*)&server, 0, sizeof(server));
    server.sin_family = AF_INET;
    server.sin_port = htons(PORT);
    server.sin_addr.S_un.S_addr = inet_addr(SERVER);
    ret = connect(client_socket, (SOCKADDR *) &server, sizeof(server)); 
    if (ret)
    {
    closesocket(client_socket);
    WSACleanup();
        printf("connect error code: %d", WSAGetLastError());
        return 2;
    }
    // start communication
    while (--loop > 0)
    {
        char * message="hello world from windows";
        printf("send message: %s \n ", message);
        

        // send the message
        if (send(client_socket, message, strlen(message), 0) == SOCKET_ERROR)
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
        if (answer_length = recv(client_socket, answer, BUFLEN, 0) == SOCKET_ERROR)
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
