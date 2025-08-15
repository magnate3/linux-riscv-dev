
#include <stdio.h>

#include <stdlib.h>

#include <unistd.h>

#include <string.h>

#include <time.h>

#include <arpa/inet.h>

#include <sys/socket.h>

#include <netinet/ip.h>

#include <netinet/tcp.h>


struct pseudo_header {

	u_int32_t source_address;

	u_int32_t dest_address;

	u_int8_t placeholder;

	u_int8_t protocol;
	
	u_int16_t tcp_length;

};

#define DATAGRAM_LEN 4096

#define OPT_SIZE 20

unsigned short checksum(const char *buf, unsigned size) {

	unsigned sum = 0, i;

	for(i = 0; i < size - 1; i += 2) {

		unsigned short word16 = *(unsigned short *) &buf[i];
		
		sum += word16;

	}

	if (size & 1) {

		unsigned short word16 = (unsigned char) buf[i];

		sum += word16;

	}

	while (sum >> 16) sum = (sum & 0xFFFF)+(sum >> 16);

	return ~sum;

}

void create_syn_packet(struct sockaddr_in* src, struct sockaddr_in* dst, char** out_packet, int* out_packet_len) {

	char *datagram = calloc(DATAGRAM_LEN, sizeof(char));

	struct iphdr *iph = (struct iphdr*)datagram;

	struct tcphdr *tcph = (struct tcphdr*)(datagram + sizeof(struct iphdr));

	struct pseudo_header psh;

	iph->ihl = 5;

	iph->version = 4;

	iph->tos = 0;

	iph->tot_len = sizeof(struct iphdr) + sizeof(struct tcphdr) + OPT_SIZE;

	iph->id = htonl(rand() % 65535); 

	iph->frag_off = 0;

	iph->ttl = 64;

	iph->protocol = IPPROTO_TCP;

	iph->check = 0; 

	iph->saddr = src->sin_addr.s_addr;

	iph->daddr = dst->sin_addr.s_addr;

	tcph->source = src->sin_port;

	tcph->dest = dst->sin_port;

	tcph->seq = htonl(rand() % 4294967295);

	tcph->ack_seq = htonl(0);

	tcph->doff = 10;

	tcph->fin = 0;

	tcph->syn = 1;

	tcph->rst = 0;

	tcph->psh = 0;

	tcph->ack = 0;

	tcph->urg = 0;

	tcph->check = 0;

	tcph->window = htons(5840);

	tcph->urg_ptr = 0;

	psh.source_address = src->sin_addr.s_addr;

	psh.dest_address = dst->sin_addr.s_addr;

	psh.placeholder = 0;

	psh.protocol = IPPROTO_TCP;

	psh.tcp_length = htons(sizeof(struct tcphdr) + OPT_SIZE);

	int psize = sizeof(struct pseudo_header) + sizeof(struct tcphdr) + OPT_SIZE;

	char* pseudogram = malloc(psize);

	memcpy(pseudogram, (char*)&psh, sizeof(struct pseudo_header));

	memcpy(pseudogram + sizeof(struct pseudo_header), tcph, sizeof(struct tcphdr) + OPT_SIZE);

	datagram[40] = 0x02;

	datagram[41] = 0x04;

	int16_t mss = htons(48);

	memcpy(datagram + 42, &mss, sizeof(int16_t));

	datagram[44] = 0x04;

	datagram[45] = 0x02;

	pseudogram[32] = 0x02;

	pseudogram[33] = 0x04;

	memcpy(pseudogram + 34, &mss, sizeof(int16_t));

	pseudogram[36] = 0x04;

	pseudogram[37] = 0x02;

	tcph->check = checksum((const char*)pseudogram, psize);

	iph->check = checksum((const char*)datagram, iph->tot_len);

	*out_packet = datagram;

	*out_packet_len = iph->tot_len;

	free(pseudogram);

}

void create_ack_packet(struct sockaddr_in* src, struct sockaddr_in* dst, int32_t seq, int32_t ack_seq, char** out_packet, int* out_packet_len) {

	char *datagram = calloc(DATAGRAM_LEN, sizeof(char));

	struct iphdr *iph = (struct iphdr*)datagram;

	struct tcphdr *tcph = (struct tcphdr*)(datagram + sizeof(struct iphdr));

	struct pseudo_header psh;

	iph->ihl = 5;

	iph->version = 4;

	iph->tos = 0;

	iph->tot_len = sizeof(struct iphdr) + sizeof(struct tcphdr) + OPT_SIZE;

	iph->id = htonl(rand() % 65535);

	iph->frag_off = 0;

	iph->ttl = 64;

	iph->protocol = IPPROTO_TCP;

	iph->check = 0;

	iph->saddr = src->sin_addr.s_addr;

	iph->daddr = dst->sin_addr.s_addr;

	tcph->source = src->sin_port;

	tcph->dest = dst->sin_port;

	tcph->seq = htonl(seq);

	tcph->ack_seq = htonl(ack_seq);

	tcph->doff = 10;

	tcph->fin = 0;

	tcph->syn = 0;

	tcph->rst = 0;

	tcph->psh = 0;

	tcph->ack = 1;

	tcph->urg = 0;

	tcph->check = 0; 

	tcph->window = htons(5840);

	tcph->urg_ptr = 0;

	psh.source_address = src->sin_addr.s_addr;

	psh.dest_address = dst->sin_addr.s_addr;

	psh.placeholder = 0;

	psh.protocol = IPPROTO_TCP;

	psh.tcp_length = htons(sizeof(struct tcphdr) + OPT_SIZE);

	int psize = sizeof(struct pseudo_header) + sizeof(struct tcphdr) + OPT_SIZE;


	char* pseudogram = malloc(psize);

	memcpy(pseudogram, (char*)&psh, sizeof(struct pseudo_header));

	memcpy(pseudogram + sizeof(struct pseudo_header), tcph, sizeof(struct tcphdr) + OPT_SIZE);

	tcph->check = checksum((const char*)pseudogram, psize);

	iph->check = checksum((const char*)datagram, iph->tot_len);

	*out_packet = datagram;

	*out_packet_len = iph->tot_len;

	free(pseudogram);

}

void create_data_packet(struct sockaddr_in* src, struct sockaddr_in* dst, int32_t seq, int32_t ack_seq, char* data, int data_len, char** out_packet, int* out_packet_len) {

	char *datagram = calloc(DATAGRAM_LEN, sizeof(char));

	struct iphdr *iph = (struct iphdr*)datagram;
	
	struct tcphdr *tcph = (struct tcphdr*)(datagram + sizeof(struct iphdr));
	
	struct pseudo_header psh;

	char* payload = datagram + sizeof(struct iphdr) + sizeof(struct tcphdr) + OPT_SIZE;

	memcpy(payload, data, data_len);

	iph->ihl = 5;

	iph->version = 4;

	iph->tos = 0;

	iph->tot_len = sizeof(struct iphdr) + sizeof(struct tcphdr) + OPT_SIZE + data_len;

	iph->id = htonl(rand() % 65535);

	iph->frag_off = 0;

	iph->ttl = 64;

	iph->protocol = IPPROTO_TCP;

	iph->check = 0;

	iph->saddr = src->sin_addr.s_addr;

	iph->daddr = dst->sin_addr.s_addr;

	tcph->source = src->sin_port;

	tcph->dest = dst->sin_port;

	tcph->seq = htonl(seq);

	tcph->ack_seq = htonl(ack_seq);

	tcph->doff = 10;

	tcph->fin = 0;

	tcph->syn = 0;

	tcph->rst = 0;

	tcph->psh = 1;

	tcph->ack = 1;

	tcph->urg = 0;

	tcph->check = 0;

	tcph->window = htons(5840);

	tcph->urg_ptr = 0;

	psh.source_address = src->sin_addr.s_addr;

	psh.dest_address = dst->sin_addr.s_addr;

	psh.placeholder = 0;

	psh.protocol = IPPROTO_TCP;

	psh.tcp_length = htons(sizeof(struct tcphdr) + OPT_SIZE + data_len);

	int psize = sizeof(struct pseudo_header) + sizeof(struct tcphdr) + OPT_SIZE + data_len;

	char* pseudogram = malloc(psize);

	memcpy(pseudogram, (char*)&psh, sizeof(struct pseudo_header));

	memcpy(pseudogram + sizeof(struct pseudo_header), tcph, sizeof(struct tcphdr) + OPT_SIZE + data_len);

	tcph->check = checksum((const char*)pseudogram, psize);

	iph->check = checksum((const char*)datagram, iph->tot_len);

	*out_packet = datagram;

	*out_packet_len = iph->tot_len;

	free(pseudogram);

}

void read_seq_and_ack(const char* packet, uint32_t* seq, uint32_t* ack) {

	uint32_t seq_num;

	memcpy(&seq_num, packet + 24, 4);

	uint32_t ack_num;

	memcpy(&ack_num, packet + 28, 4);

	*seq = ntohl(seq_num);

	*ack = ntohl(ack_num);

	printf("sequence number: %lu\n", (unsigned long)*seq);

	printf("acknowledgement number: %lu\n", (unsigned long)*seq);

}

int receive_from(int sock, char* buffer, size_t buffer_length, struct sockaddr_in *dst) {

	unsigned short dst_port;

	int received;

	do {

		received = recvfrom(sock, buffer, buffer_length, 0, NULL, NULL);

		if(received < 0)

			break;
		
		memcpy(&dst_port, buffer + 22, sizeof(dst_port));

	}

	while (dst_port != dst->sin_port);

	printf("received bytes: %d\n", received);

	printf("destination port: %d\n", ntohs(dst->sin_port));

	return received;

}

int main(int argc, char** argv) {

	if(argc != 4) {

		printf("invalid parameters.\n");

		printf("USAGE %s <source-ip> <target-ip> <port>\n", argv[0]);

		return 1;
	}

	srand(time(NULL));

	int sock = socket(AF_INET, SOCK_RAW, IPPROTO_TCP);

	if(sock == -1) {

		printf("socket creation failed\n");

		return 1;

	}

	struct sockaddr_in daddr;
	
	daddr.sin_family = AF_INET;
	
	daddr.sin_port = htons(atoi(argv[3]));
	
	if(inet_pton(AF_INET, argv[2], &daddr.sin_addr) != 1){

		printf("destination IP configuration failed\n");

		return 1;
	}

	struct sockaddr_in saddr;

	saddr.sin_family = AF_INET;

	saddr.sin_port = htons(rand() % 65535);

	if (inet_pton(AF_INET, argv[1], &saddr.sin_addr) != 1) {

		printf("source IP configuration failed\n");
		
		return 1;

	}

	printf("selected source port number: %d\n", ntohs(saddr.sin_port));

	int one = 1;

	const int *val = &one;

	if(setsockopt(sock, IPPROTO_IP, IP_HDRINCL, val, sizeof(one)) == -1) {

		printf("setsockopt(IP_HDRINCL, 1) failed\n");

		return 1;

	}

	char* packet;

	int packet_len;

	create_syn_packet(&saddr, &daddr, &packet, &packet_len);

	int sent;

	if((sent = sendto(sock, packet, packet_len, 0, (struct sockaddr*)&daddr, sizeof(struct sockaddr))) == -1) {

		printf("sendto() failed\n");

	}
	
	else {

		printf("successfully sent %d bytes SYN!\n", sent);

	}

	char recvbuf[DATAGRAM_LEN];

	int received = receive_from(sock, recvbuf, sizeof(recvbuf), &saddr);

	if(received <= 0) {

		printf("receive_from() failed\n");

	}
	
	else {
		
		printf("successfully received %d bytes SYN-ACK!\n", received);

	}

	uint32_t seq_num, ack_num;

	read_seq_and_ack(recvbuf, &seq_num, &ack_num);

	int new_seq_num = seq_num + 1;

	create_ack_packet(&saddr, &daddr, ack_num, new_seq_num, &packet, &packet_len);

	if((sent = sendto(sock, packet, packet_len, 0, (struct sockaddr*)&daddr, sizeof(struct sockaddr))) == -1) {

		printf("sendto() failed\n");
	
	}
	
	else {

		printf("successfully sent %d bytes ACK!\n", sent);
	
	}

	char request[] = "GET / HTTP/1.1\r\nHost: localhost\r\n\r\n";

	create_data_packet(&saddr, &daddr, ack_num, new_seq_num, request, sizeof(request) - 1/sizeof(char), &packet, &packet_len);

	if((sent = sendto(sock, packet, packet_len, 0, (struct sockaddr*)&daddr, sizeof(struct sockaddr))) == -1) {

		printf("send failed\n");

	}
	
	else {

		printf("successfully sent %d bytes PSH!\n", sent);

	}

	while((received = receive_from(sock, recvbuf, sizeof(recvbuf), &saddr)) > 0) {

		printf("successfully received %d bytes!\n", received);
		
		read_seq_and_ack(recvbuf, &seq_num, &ack_num);
		
		new_seq_num = seq_num + 1;
		
		create_ack_packet(&saddr, &daddr, ack_num, new_seq_num, &packet, &packet_len);
		
		if((sent = sendto(sock, packet, packet_len, 0, (struct sockaddr*)&daddr, sizeof(struct sockaddr))) == -1) {

			printf("send failed\n");

		}
		
		else {

			printf("successfully sent %d bytes ACK!\n", sent);
		
		}
	
	}

	close(sock);

	return 0;
	
}
