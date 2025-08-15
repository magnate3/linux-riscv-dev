#ifndef _CHECKSUM_H_
#define _CHECKSUM_H_
#include <arpa/inet.h>
#define ETHER_HDRLEN			14
#define IP4_HDRLEN			20
#define IP6_HDRLEN			40
#define ICMP_HDRLEN			8
#define	ICMP_DATALEN			4
uint16_t checksum(uint16_t *addr, int len);
uint16_t icmp6_checksum(struct ip6_hdr iphdr, struct icmp6_hdr icmp6hdr, uint8_t *payload, int payloadlen);
#endif /* _CHECKSUM_H_ */
