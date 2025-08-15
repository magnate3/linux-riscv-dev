#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <net/if.h>
 
int main(){
 
  char *ip = "20.205.243.166";
  int port = 443;
 
  int sock;
  struct sockaddr_in addr;
  socklen_t addr_size;
  char buffer[1024];
  int n;
      /* 指定接口 */
  struct ifreq nif;
  char *inface = "tun0";
  strcpy(nif.ifr_name, inface);
  sock = socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0){
    perror("[-]Socket error");
    exit(1);
  }
  printf("[+]TCP server socket created.\n");
    /* 绑定接口 */
    if (setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, (char *)&nif, sizeof(nif)) < 0)
    {
        close(sock);
        printf("bind interface fail, errno: %d \r\n", errno);
		return 0;		
    }
    else
    {
        printf("bind interface success \r\n");
    }
  memset(&addr, '\0', sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = port;
  addr.sin_addr.s_addr = inet_addr(ip);
  
  connect(sock, (struct sockaddr*)&addr, sizeof(addr));
  printf("Connected to the server.\n");
 
  bzero(buffer, 1024);
  strcpy(buffer, "HELLO, THIS IS CLIENT.");
  printf("Client: %s\n", buffer);
  send(sock, buffer, strlen(buffer), 0);
 
  bzero(buffer, 1024);
  recv(sock, buffer, sizeof(buffer), 0);
  printf("Server: %s\n", buffer);
 
  close(sock);
  printf("Disconnected from the server.\n");
 
  return 0;
}
 
