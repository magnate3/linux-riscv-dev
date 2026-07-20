#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <linux/udp.h>
#include <signal.h>
#include <errno.h>
//#include <linux/ip.h>
#include <netinet/ip.h>

struct sockaddr_in *servaddr = NULL, *client_addr = NULL;
int sock_fd;

#define PORT 50000
#define PORT_CLIENT 50001
#define SERVER_ADDR "10.10.16.81"
#define CLIENT_ADDR "10.10.16.82"
/* structure to calculate UDP checksum
* The below members are part of the IP header
* which do not change from the UDP layer and hence
* are used as a part of the UDP checksum */

struct pseudo_iphdr {
        unsigned int source_ip_addr;
        unsigned int dest_ip_addr;
        unsigned char fixed;
        unsigned char protocol;
        unsigned short udp_len;
};

/* checksum code to calculate TCP/UDP checksum
* Code taken from Unix network programming – Richard stevens*/

unsigned short in_cksum (uint16_t * addr, int len)
{
        int nleft = len;
        unsigned int sum = 0;
        unsigned short *w = addr;
        unsigned short answer = 0;

        /* Our algorithm is simple, using a 32 bit accumulator (sum), we add
        * sequential 16 bit words to it, and at the end, fold back all the
        * carry bits from the top 16 bits into the lower 16 bits.
        */
        while (nleft > 1) {
                sum += *w++;
                nleft -= 2;
         }

        /* mop up an odd byte, if necessary */
        if (nleft == 1) {
                *(unsigned char *) (&answer) = * (unsigned char *) w;
                sum += answer;
        }

        /* add back carry outs from top 16 bits to low 16 bits */
        sum = (sum >> 16) + (sum & 0xffff); /* add hi 16 to low 16 */
        sum += (sum >> 16); /* add carry */
        answer = (unsigned short) ~sum; /* truncate to 16 bits */
        return (answer);
}

/* Interrupt_handler – so that CTRL + C can be used to
* exit the program */
void interrupt_handler (int signum) {
        close(sock_fd);
        free(client_addr);
        exit(0);
}

#if DEBUG
/* print the IP and UDP headers */
void dumpmsg(unsigned char *recvbuffer, int length) {
        int count_per_length = 28, i = 0;
        for (i = 0; i < count_per_length; i++) {
                printf("%02x ", recvbuffer[i]);
        }
        printf("\n");
}
#endif

int main () {
        char buffer[1024] = {0};
        unsigned char recvbuffer[1024] = {0};
        int length;
        char *string = "Hello client";
        struct udphdr *udp_hdr = NULL;
        char *string_data = NULL;
        char *recv_string_data = NULL;
        char *csum_buffer = NULL;
        struct pseudo_iphdr csum_hdr;
        int error;

        signal (SIGINT, interrupt_handler);
        signal (SIGTERM, interrupt_handler);

        /* Part 1: create the socket */
        sock_fd = socket(AF_INET, SOCK_RAW, IPPROTO_UDP);
        if(0 > sock_fd) {
                printf("unable to create socket\n");
                exit(0);
        }

        servaddr = (struct sockaddr_in *)malloc(sizeof(struct sockaddr_in));
        if (servaddr == NULL) {
                printf("could not allocate memory\n");
                goto end;
        }

        servaddr->sin_family = AF_INET;
        servaddr->sin_port = PORT;
        servaddr->sin_addr.s_addr = inet_addr(SERVER_ADDR);

        /* Part 2 – fill data structure and bind to socket */
        if (0 != (bind(sock_fd, (struct sockaddr *)servaddr, sizeof(struct sockaddr_in)))) {
                printf("could not bind server socket to address\n");
                goto end1;
        }

        /* part 3: read and write data */
        client_addr = (struct sockaddr_in *)malloc(sizeof(struct sockaddr_in));
        if (client_addr == NULL) {
                printf("Unable to allocate memory to client address socket\n");
                goto end2;
        }

        client_addr->sin_family = AF_INET;
        client_addr->sin_port = PORT_CLIENT;
        client_addr->sin_addr.s_addr = inet_addr(CLIENT_ADDR);

        error = connect(sock_fd, (struct sockaddr *)client_addr, sizeof(struct sockaddr_in));
        if (error != 0) {
                printf("error %d", errno);
                printf("connect returned error\n");
                goto end2;
        }

        /* copy the data after the UDP header */
        string_data = (char *) (buffer + sizeof(struct udphdr));
        strncpy(string_data, string, strlen(string));

        /* Modify some parameters to send to client in UDP hdr */

        udp_hdr = (struct udphdr *)buffer;
        udp_hdr->source = htons(PORT);
        udp_hdr->dest = htons(PORT_CLIENT);
        udp_hdr->len = htons(sizeof(struct udphdr));

         /* calculate the UDP checksum – based on wikipedia
        * pseudo IP header + UDP HDR +
        * UDP data- check sum is calculated.
        * create a buffer to calculate CSUM and calculate CSUM*/

        csum_buffer = (char *)calloc((sizeof(struct pseudo_iphdr) + sizeof(struct udphdr) + strlen(string_data)), sizeof(char));
        if (csum_buffer == NULL) {
                printf("Unable to allocate csum buffer\n");
                goto end1;
        }

        csum_hdr.source_ip_addr = inet_addr(SERVER_ADDR);
        csum_hdr.dest_ip_addr = inet_addr(CLIENT_ADDR);
        csum_hdr.fixed = 0;
        csum_hdr.protocol = IPPROTO_UDP; /* UDP protocol */
        csum_hdr.udp_len = htons(sizeof(struct udphdr) + strlen(string_data) + 1);

        memcpy(csum_buffer, (char *)&csum_hdr, sizeof(struct pseudo_iphdr));
        memcpy(csum_buffer + sizeof(struct pseudo_iphdr), buffer, (sizeof(struct udphdr) + strlen(string_data) + 1));

        udp_hdr->check = (in_cksum((unsigned short *) csum_buffer,
(sizeof(struct pseudo_iphdr)+ sizeof(struct udphdr) + strlen(string_data) + 1)));

        printf("checksum is %x\n", udp_hdr->check);
        /* since we are resending the same packet over and over again
       * free the csum buffer here */
       free (csum_buffer);

       while (1) {
               memset(recvbuffer, 0, sizeof(recvbuffer));

               read(sock_fd, recvbuffer, sizeof(recvbuffer));

               udp_hdr = (struct udphdr *)(recvbuffer + sizeof (struct iphdr));
               recv_string_data = (char *) (recvbuffer + sizeof (struct iphdr) + sizeof (struct udphdr));
#if DEBUG
               dumpmsg((unsigned char *)&recvbuffer, (sizeof(struct iphdr) + sizeof(struct udphdr)+ strlen(string_data) + 1));
#endif
               //if (PORT == ntohs(udp_hdr->dest)) {
               //        printf("udp data received from port : %d\n",ntohs(udp_hdr->source));
               //        printf("data received at server from client is: %s\n", recv_string_data);
               //}
			   struct ip *ip = (struct ip*)recvbuffer;
			   printf("  %20s",inet_ntoa(ip->ip_src));
			   printf("%20s    %5d     %5d \n",inet_ntoa(ip->ip_dst),ip->ip_p,ntohs(ip->ip_len));
			   printf("\n");
               printf("udp data received from port : %d\n",ntohs(udp_hdr->source));
               printf("data received at server from client is: %s\n", recv_string_data);

               write(sock_fd, buffer, sizeof(buffer));
        }

end2:
        free(client_addr);
end1:
        free(servaddr);
end:
        close(sock_fd);

        return 0;
}
