#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in_systm.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/ip_icmp.h>
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/time.h>

void sig_alrm(int);
void send_msg(void);
void handlePing(void);
unsigned short cksum_in(unsigned short *, int);
void tv_sub(struct timeval *, struct timeval *);

struct timeval * tvsend, tvrecv;
int sd;
pid_t pid;
int nsent = 0;

struct sockaddr_in sasend;
struct sockaddr_in sarecv;
int salen;

int main(int argc, char *argv[]){
	if(argc != 2){
		printf("usage : ping domain_name\n");
		exit(-1);
	}

	bzero((char *)&sasend, sizeof(sasend));
	sasend.sin_family = AF_INET;
	sasend.sin_addr.s_addr = inet_addr(argv[1]);
	salen = sizeof(sasend);

	pid = getpid() & 0xffff;

	handlePing();

	return 0;
}

void handlePing(void) {
	int len, hlen, icmplen;
	struct timeval tval;
	char buf[1500];

	fd_set readfd;
	struct iphdr *iph;
	struct icmp *icmp;

	double rtt;

	signal(SIGALRM, sig_alrm);

	if((sd=socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0) {
		printf("socket open error\n");
		exit(-1);
	}

	sig_alrm(SIGALRM);

	for(;;) {
		if((len = recvfrom(sd, buf, sizeof(buf), 0, NULL, NULL)) < 0){
			printf("read error\n");
			exit(-1);
		}

		iph = (struct iphdr *)buf;
		hlen = iph->ihl * 4;

		if(iph->protocol != IPPROTO_ICMP)
			return;

		if(iph->saddr == sasend.sin_addr.s_addr){
			icmp = (struct icmp *)(buf + hlen);
			icmplen = len - hlen;

			if(icmp->icmp_type == ICMP_ECHOREPLY){
				if(icmp->icmp_id != pid)
					return;

				gettimeofday(&tvrecv, NULL);
				tvsend = (struct timeval *)icmp->icmp_data;
				tv_sub(&tvrecv, tvsend);

				rtt = tvrecv.tv_sec * 1000.0 + tvrecv.tv_usec / 1000.0;
				
				printf("%d byte from ** : seq = %u, ttl = %d, rtt = %.3f ms \n", icmplen, icmp->icmp_seq, iph->ttl, rtt);
			}
		}
	}
}

void sig_alrm(int signo){
	send_msg();

	alarm(1);
	return;
}

void send_msg(void){
	int len;
	struct icmp *icmp;
	char sendbuf[1500];
	int datalen = 56;

	icmp = (struct icmp *)sendbuf;

	icmp->icmp_type = ICMP_ECHO;
	icmp->icmp_code = 0;
	icmp->icmp_id = pid;
	icmp->icmp_seq = nsent++;
	memset(icmp->icmp_data, 0xa5, datalen);

	gettimeofday((struct timeval *)icmp->icmp_data, NULL);

	len = 8 + datalen;
	icmp->icmp_cksum = 0;
	icmp->icmp_cksum = cksum_in((unsigned short *)icmp, len);

	sendto(sd, sendbuf, len, 0, (struct sockaddr *)&sasend, len);
}

void tv_sub(struct timeval *out, struct timeval *in){
	if((out->tv_usec -= in->tv_usec) < 0) {
		--out->tv_sec;
		out->tv_usec += 100000;
	}
	out->tv_sec -= in->tv_sec;
}

unsigned short cksum_in(unsigned short *addr, int len)
{
	unsigned long sum = 0;
	unsigned short answer = 0;
	unsigned short *w = addr;
	int nleft = len;

	while (nleft > 1) {
		sum += *w++;
		if (sum & 0x80000000)
			sum = (sum & 0xffff) + (sum >> 16);
		nleft -= 2;
	}

	if (nleft == 1) {
		*(unsigned char *)(&answer) = *(unsigned char *)w;
		sum += answer;
	}

	while (sum >> 16)
		sum = (sum & 0xffff) + (sum >> 16);

	return(sum == 0xffff) ? sum : ~sum;
}
