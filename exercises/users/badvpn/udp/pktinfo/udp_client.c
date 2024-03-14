// Client side implementation of UDP client-server model 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <string.h> 
#include <sys/types.h> 
#include <sys/socket.h> 
#include <arpa/inet.h> 
#include <netinet/in.h> 
#include <net/if.h>
#include <errno.h>
#define PORT     5000 
#define MAXLINE 1024 
    
// Driver code 
int main() { 
    int sockfd; 
    char buffer[MAXLINE]; 
    char *hello = "Hello from udp client"; 
    struct sockaddr_in     servaddr; 
    char *ip = "10.10.16.82";
    /* 指定接口 */
     struct ifreq nif;
    char *inface = "enp125s0f0";
    strcpy(nif.ifr_name, inface);
    // Creating socket file descriptor 
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0 ) { 
        perror("socket creation failed"); 
        exit(EXIT_FAILURE); 
    } 
	    /* 绑定接口 */
    if (setsockopt(sockfd, SOL_SOCKET, SO_BINDTODEVICE, (char *)&nif, sizeof(nif)) < 0)
    {
        close(sockfd);
        printf("bind interface fail, errno: %d \r\n", errno);
                return 0;
    }
    else
    {
        printf("bind interface success \r\n");
    }
    
    memset(&servaddr, 0, sizeof(servaddr)); 
        
    // Filling server information 
    servaddr.sin_family = AF_INET; 
    servaddr.sin_port = htons(PORT); 
    servaddr.sin_addr.s_addr = inet_addr(ip);; 
        
    int n, len; 
        
    sendto(sockfd, (const char *)hello, strlen(hello), 
        MSG_CONFIRM, (const struct sockaddr *) &servaddr,  
            sizeof(servaddr)); 
    printf("Hello message sent.\n"); 
            
    n = recvfrom(sockfd, (char *)buffer, MAXLINE,  
                MSG_WAITALL, (struct sockaddr *) &servaddr, 
                &len); 
    buffer[n] = '\0'; 
    printf("recv: %s\n", buffer); 
    
    close(sockfd); 
    return 0; 
}
