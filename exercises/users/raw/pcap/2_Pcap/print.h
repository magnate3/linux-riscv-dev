/**
 * @file print.h
 * @brief キャプチャした情報を出力する関数群のヘッダファイル
 */

#ifndef PRINT_H_
#define PRINT_H_

char *my_ether_ntoa_r(u_char *hwaddr, char *buf, socklen_t size);
char *arp_ip2str(u_int8_t *ip, char *buf, socklen_t size);
char *ip_ip2str(u_int32_t ip, char *buf, socklen_t size);
int PrintEtherHeader(struct ether_header *eh, FILE *fp);
int PrintArp(struct ether_arp *eth_arp, FILE *fp);
int PrintIpHeader(struct iphdr *ip_hdr, u_char *option, int optionLen, FILE *fp);
int PrintIp6Header(struct ip6_hdr *ip6_hdr, FILE *fp);
int PrintIcmp(struct icmp *icmp, FILE *fp);
int PrintIcmp6(struct icmp6_hdr *icmp6, FILE *fp);
int PrintTcp(struct tcphdr *tcp, FILE *fp);
int PrintUdp(struct udphdr *udp, FILE *fp);

#endif