#include <stdio.h>
#include <stdlib.h> // for exit(1)
#include <string.h> //strerror
#include <sys/socket.h> //sockets duh
#include <arpa/inet.h> // contains lots of stuff- has def for IPPROTO_TCP and such
#include <errno.h> //For errno - the error number
#include <unistd.h> // for close

#include <netinet/ip.h> // needed for ip_icmp?
#include <netinet/ip_icmp.h> //https://www.cymru.com/Documents/ip_icmp.h

// checksum function BSD Tahoe
unsigned short in_cksum(unsigned short *addr, int len)
{
	int nleft = len;
	int sum = 0;
	unsigned short *w = addr;
	unsigned short answer = 0;

	while (nleft > 1) {
		sum += *w++;
		nleft -= 2;
	}

	if (nleft == 1) {
		*(unsigned char *) (&answer) = *(unsigned char *) w;
		sum += answer;
	}
	
	sum = (sum >> 16) + (sum & 0xFFFF);
	sum += (sum >> 16);
	answer = ~sum;
	return (answer);
}
	


int main(int argc, char const *argv[]){
	//--------------------------------------------------------------------
	//	struct ip {
    //	    u_int   ip_hl:4,                /* header length */
    //	            ip_v:4;                 /* ip version */
    //	    u_char  ip_tos;                 /* type of service */
    //	    u_short ip_len;                 /* total length */
    //	    u_short ip_id;                  /* identification */
    //	    u_short ip_off;                 /* fragment offset */
    //	    u_char  ip_ttl;                 /* time to live */
    //	    u_char  ip_p;                   /* protocol */
    //	    u_short ip_sum;                 /* checksum */
    //	    struct  in_addr ip_src,ip_dst;  /* source and dest address */
	//	};
	//--------------------------------------------------------------------
	struct ip ip;
	//--------------------------------------------------------------------
	//	struct icmp
	//	{
	//	  u_int8_t  icmp_type;	/* type of message, see below */
	//	  u_int8_t  icmp_code;	/* type sub code */
	//	  u_int16_t icmp_cksum;	/* ones complement checksum of struct */
	//	  union
	//	  {
	//	    u_char ih_pptr;		/* ICMP_PARAMPROB */
	//	    struct in_addr ih_gwaddr;	/* gateway address */
	//	    struct ih_idseq		/* echo datagram */
	//	    {
	//	      u_int16_t icd_id;
	//	      u_int16_t icd_seq;
	//	    } ih_idseq;
	//	    u_int32_t ih_void;
	//--------------------------------------------------------------------
	struct icmp icmp;
	int sd;
	const int on = 1;
	//--------------------------------------------------------------------
	//	struct sockaddr_in {
	//	    short            sin_family;   // e.g. AF_INET, AF_INET6
	//	    unsigned short   sin_port;     // e.g. htons(3490)
	//	    struct in_addr   sin_addr;     // see struct in_addr, below
	//	    char             sin_zero[8];  // zero this if you want to
	//	};
	//
	//	struct in_addr {
	//	    unsigned long s_addr;          // load with inet_pton()
	//	};
	//--------------------------------------------------------------------
	struct sockaddr_in sin;
	u_char *packet;

	packet = (u_char *)malloc(sizeof(struct ip)+sizeof(struct icmp));

	ip.ip_hl = 0x5;
	ip.ip_v = 0x4;
	ip.ip_tos = 0x0;
	ip.ip_len = sizeof(struct ip)+sizeof(struct icmp);
	ip.ip_id = htons(12830);
	ip.ip_off = 0x0;
	ip.ip_ttl = 64;
	ip.ip_p = IPPROTO_ICMP;
	ip.ip_sum = 0x0;
	ip.ip_src.s_addr = inet_addr("10.10.16.251");
	ip.ip_dst.s_addr = inet_addr("10.10.16.81");
	ip.ip_sum = in_cksum((unsigned short *)&ip, 20);
	memcpy(packet, &ip, sizeof(ip));

	icmp.icmp_type = ICMP_ECHO;
	icmp.icmp_code = 2;
	icmp.icmp_id = 1000;
	icmp.icmp_seq = 0;
	icmp.icmp_cksum = 0;
	icmp.icmp_cksum = in_cksum((unsigned short *)&icmp, sizeof(struct icmp));
	memcpy(packet + sizeof(struct ip), &icmp, sizeof(struct icmp));

	
	/*	http://linux.die.net/man/7/socket
	#
	#	sockfd = socket(int socket_family, int socket_type, int protocol); 
	#
	*/
	if ((sd = socket(AF_INET, SOCK_RAW, IPPROTO_ICMP)) < 0) {
		perror("raw socket");
		exit(1);
	}


	/*	http://pubs.opengroup.org/onlinepubs/009695399/functions/setsockopt.html
	#
	#	int setsockopt(int socket, int level, int option_name,
    #   		const void *option_value, socklen_t option_len);
    #
    */
	if (setsockopt(sd, IPPROTO_IP, IP_HDRINCL, &on, sizeof(on)) < 0) {
		perror("setsockopt");
		exit(1);
	}

	
	memset(&sin, 0, sizeof(sin));
	sin.sin_family = AF_INET;
	sin.sin_addr.s_addr = ip.ip_dst.s_addr;

	if (sendto(sd, packet, sizeof(struct icmp)+sizeof(struct ip), 0, (struct sockaddr *)&sin, sizeof(struct sockaddr)) < 0)  {
		perror("sendto");
		exit(1);
	}	

	unsigned char buf[sizeof(struct ip)+sizeof(struct icmphdr)];
	int bytes, len=sizeof(sin);

	bzero(buf, sizeof(buf));
	bytes = recvfrom(sd, buf, sizeof(buf), 0, (struct sockaddr*)&sin, &len);
	if ( bytes > 0 ) {
		printf("Bytes received: %d\n", bytes); //display(buf, bytes);
	} else {
		perror("recvfrom");
	}

    struct ip *iprecv = (struct ip *)(buf);
	puts("\nIP HEADER");
	printf("\tIP version: %d\n", iprecv->ip_v);	
	printf("\tProtocol: %d\n", iprecv->ip_p);
	printf("\tIdentification: 0x%X\n", ntohs(iprecv->ip_id));
	printf("\tHeader len: %i\n", iprecv->ip_hl*4);
	printf("\tChecksum: 0x%X\n",ntohs(iprecv->ip_sum));
	printf("\tTTL: %d\n", iprecv->ip_ttl);
	printf("\tSource IP: %s\n", inet_ntoa(iprecv->ip_src));
	printf("\tDestination IP: %s\n", inet_ntoa(iprecv->ip_dst));
	
	struct icmphdr *icmprecv = (struct icmphdr *)(buf+iprecv->ip_hl*4);
	puts("\nICMP HEADER");
	printf("\tType: %i\n", icmprecv->type);
	printf("\tCode: %d\n", icmprecv->code);
	printf("\tChecksum: 0x%X\n", ntohs(icmprecv->checksum)); 
	printf("\tIdentifier: %i\n", icmprecv->un.echo.id); 
	
	close(sd);
	return 0;
}
