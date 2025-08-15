/*
  Example IPv6 UDP client.
  Copyright (C) 2010 Russell Bradford

  This program is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 2
  of the License, or (at your option) any later version.

  This program is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License (http://www.gnu.org/copyleft/gpl.html)
  for more details. 
*/

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>
#include <memory.h>

#define PORT 8080
#define MESSAGE "hi, I am ipv6 client"
#define SERVADDR "2001:db8:0:0::10.10.103.81"
//#define SERVADDR "2001:db8::a0a:6752"

#define UDP_BIG_PKT_LEN 1470
int main(void)
{
  int sock;
  socklen_t clilen;
  struct sockaddr_in6 server_addr, client_addr;
  char buffer[UDP_BIG_PKT_LEN*2];
  char addrbuf[INET6_ADDRSTRLEN];

  /* create a DGRAM (UDP) socket in the INET6 (IPv6) protocol */
  sock = socket(PF_INET6, SOCK_DGRAM, 0);

  if (sock < 0) {
    perror("creating socket");
    exit(1);
  }

  /* create server address: where we want to send to */

  /* clear it out */
  memset(&server_addr, 0, sizeof(server_addr));

  /* it is an INET address */
  server_addr.sin6_family = AF_INET6;

  /* the server IP address, in network byte order */
  inet_pton(AF_INET6, SERVADDR, &server_addr.sin6_addr);

  /* the port we are going to send to, in network byte order */
  server_addr.sin6_port = htons(PORT);
#if 0
  /* now send a datagram */
  if (sendto(sock, MESSAGE, sizeof(MESSAGE), 0,
             (struct sockaddr *)&server_addr,
	     sizeof(server_addr)) < 0) {
      perror("sendto failed");
      exit(4);
  }
#else
  int i = 0;
  while(i< UDP_BIG_PKT_LEN){

     buffer[i] = 'a';
     ++ i;
  }
  buffer[UDP_BIG_PKT_LEN] = 0;
  if (sendto(sock, buffer, UDP_BIG_PKT_LEN, 0,
             (struct sockaddr *)&server_addr,
	     sizeof(server_addr)) < 0) {
      perror("sendto failed");
      exit(4);
  }
  memset(buffer,0, sizeof(buffer));
#endif
#if 1
  printf("waiting for a reply...\n");
  clilen = sizeof(client_addr);
  if (recvfrom(sock, buffer, UDP_BIG_PKT_LEN, 0,
	       (struct sockaddr *)&client_addr,
               &clilen) < 0) {
      perror("recvfrom failed");
      exit(4);
  }

  printf("got '%s' from %s\n", buffer,
	 inet_ntop(AF_INET6, &client_addr.sin6_addr, addrbuf,
		   INET6_ADDRSTRLEN));
#else
  printf("client send susscessfully \n");
#endif
  /* close socket */
  close(sock);

  return 0;
}
