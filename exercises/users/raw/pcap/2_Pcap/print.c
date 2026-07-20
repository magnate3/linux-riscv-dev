/**
 * @file print.c
 * @brief キャプチャした情報を出力する関数群の実装ファイル
 */

#include <arpa/inet.h>
#include <linux/if.h>
#include <net/ethernet.h>
#include <netinet/icmp6.h>
#include <netinet/if_ether.h>
#include <netinet/ip.h>
#include <netinet/ip6.h>
#include <netinet/ip_icmp.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netpacket/packet.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef ETHERTYPE_IPV6
#define ETHERTYPE_IPV6 0x86dd
#endif

/**
 * @brief MACアドレスを文字列にする
 * @param hwaddr : MACアドレス
 * @param buf : MACアドレスの文字列格納用のバッファ
 * @param size : ソケットのサイズ
 * @return MACアドレスの文字列
 */
char *my_ether_ntoa_r(u_char *hwaddr, char *buf, socklen_t size)
{
    snprintf(buf, size, "%02x:%02x:%02x:%02x:%02x:%02x",
             hwaddr[0], hwaddr[1], hwaddr[2], hwaddr[3], hwaddr[4], hwaddr[5]);

    return buf;
}

/**
 * @brief u_int8_t用IPアドレスを文字列にする
 * @param ip : MACアドレス
 * @param buf : MACアドレスの文字列格納用のバッファ
 * @param size : ソケットのサイズ
 * @return MACアドレスの文字列
 */
char *arp_ip2str(u_int8_t *ip, char *buf, socklen_t size)
{
    snprintf(buf, size, "%u.%u.%u.%u", ip[0], ip[1], ip[2], ip[3]);

    return buf;
}

/**
 * @brief u_int32_t用IPアドレスを文字列にする
 * @param ip : MACアドレス
 * @param buf : MACアドレスの文字列格納用のバッファ
 * @param size : ソケットのサイズ
 * @return MACアドレスの文字列
 */
char *ip_ip2str(u_int32_t ip, char *buf, socklen_t size)
{
    struct in_addr *addr = NULL;

    addr = (struct in_addr *)&ip;
    inet_ntop(AF_INET, addr, buf, size);

    return buf;
}

/**
 * @brief Etherヘッダを表示する
 * @param ether_hdr : Ethernetヘッダ情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintEtherHeader(struct ether_header *ether_hdr, FILE *fp)
{
    char buf[80] = {'\0'};

    fprintf(fp, "\nether_header----------------------------\n");
    fprintf(fp, "ether_dhost=%s\n", my_ether_ntoa_r(ether_hdr->ether_dhost, buf, sizeof(buf)));  // destination eth addr
    fprintf(fp, "ether_shost=%s\n", my_ether_ntoa_r(ether_hdr->ether_shost, buf, sizeof(buf)));  // source ether addr
    fprintf(fp, "ether_type=%02X", ntohs(ether_hdr->ether_type));                                // packet type ID field
    switch (ntohs(ether_hdr->ether_type)) {
    case ETH_P_IP:
        fprintf(fp, "(IP)\n");
        break;
    case ETH_P_IPV6:
        fprintf(fp, "(IPv6)\n");
        break;
    case ETH_P_ARP:
        fprintf(fp, "(ARP)\n");
        break;
    default:
        fprintf(fp, "(unknown)\n");
        break;
    }

    return 0;
}

/**
 * @brief ARP情報を表示する
 * @param eth_arp : ARP情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintArp(struct ether_arp *eth_arp, FILE *fp)
{
    static char *hrd[] = {
        "From KA9Q: NET/ROM pseudo.",
        "Ethernet 10/100Mbps.",
        "Experimental Ethernet.",
        "AX.25 Level 2.",
        "PROnet token ring.",
        "Chaosnet.",
        "IEEE 802.2 Ethernet/TR/TB.",
        "ARCnet.",
        "APPLEtalk.",
        "undefine",
        "undefine",
        "undefine",
        "undefine",
        "undefine",
        "undefine",
        "Frame Relay DLCI.",
        "undefine",
        "undefine",
        "undefine",
        "ATM.",
        "undefine",
        "undefine",
        "undefine",
        "Metricom STRIP (new IANA id)."};
    static char *op[] = {
        "undefined",
        "ARP request.",
        "ARP reply.",
        "RARP request.",
        "RARP reply.",
        "undefined",
        "undefined",
        "undefined",
        "InARP request.",
        "InARP reply.",
        "(ATM)ARP NAK."};
    char buf[80] = {'\0'};

    fprintf(fp, "\narp-------------------------------------\n");
    fprintf(fp, "arp_hrd=%u", ntohs(eth_arp->arp_hrd));
    if (ntohs(eth_arp->arp_hrd) <= 23) {
        fprintf(fp, "(%s)\n", hrd[ntohs(eth_arp->arp_hrd)]);
    }
    else {
        fprintf(fp, "(undefined)\n");
    }
    fprintf(fp, "arp_pro=%u", ntohs(eth_arp->arp_pro));
    switch (ntohs(eth_arp->arp_pro)) {
    case ETHERTYPE_IP:
        fprintf(fp, "(IP)\n");
        break;
    case ETHERTYPE_ARP:
        fprintf(fp, "(Address resolution)\n");
        break;
    case ETHERTYPE_REVARP:
        fprintf(fp, "(Reverse ARP)\n");
        break;
    case ETHERTYPE_IPV6:
        fprintf(fp, "(IPv6)\n");
        break;
    default:
        fprintf(fp, "(unknown)\n");
        break;
    }
    fprintf(fp, "arp_hln=%u\n", eth_arp->arp_hln);
    fprintf(fp, "arp_pln=%u\n", eth_arp->arp_pln);
    fprintf(fp, "arp_op=%u", ntohs(eth_arp->arp_op));
    if (ntohs(eth_arp->arp_op) <= 10) {
        fprintf(fp, "(%s)\n", op[ntohs(eth_arp->arp_op)]);
    }
    else {
        fprintf(fp, "(undefine)\n");
    }
    fprintf(fp, "arp_sha=%s\n", my_ether_ntoa_r(eth_arp->arp_sha, buf, sizeof(buf)));
    fprintf(fp, "arp_spa=%s\n", arp_ip2str(eth_arp->arp_spa, buf, sizeof(buf)));
    fprintf(fp, "arp_tha=%s\n", my_ether_ntoa_r(eth_arp->arp_tha, buf, sizeof(buf)));
    fprintf(fp, "arp_tpa=%s\n", arp_ip2str(eth_arp->arp_spa, buf, sizeof(buf)));

    return 0;
}

//! プロトコル
static char *Proto[] = {
    "undefined",
    "ICMP",
    "IGMP",
    "undefined",
    "IPIP",
    "undefined",
    "TCP",
    "undefined",
    "EGP",
    "undefined",
    "undefined",
    "undefined",
    "PUP",
    "undefined",
    "undefined",
    "undefined",
    "undefined",
    "UDP"};

/**
 * @brief IPヘッダを表示する
 * @param ip_hdr : IPヘッダ情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintIpHeader(struct iphdr *ip_hdr, u_char *option, int optionLen, FILE *fp)
{
    int i = 0;
    char buf[80] = {'\0'};

    fprintf(fp, "\nip--------------------------------------\n");
    fprintf(fp, "version=%u\n", ip_hdr->version);
    fprintf(fp, "ihl=%u\n", ip_hdr->ihl);
    fprintf(fp, "tos=%x\n", ip_hdr->tos);
    fprintf(fp, "tot_len=%u\n", ntohs(ip_hdr->tot_len));
    fprintf(fp, "id=%u\n", ntohs(ip_hdr->id));
    fprintf(fp, "frag_off=%x,%u\n", (ntohs(ip_hdr->frag_off) >> 13) & 0x07, ntohs(ip_hdr->frag_off) & 0x1FFF);
    fprintf(fp, "ttl=%u\n", ip_hdr->ttl);
    fprintf(fp, "protocol=%u", ip_hdr->protocol);
    if (ip_hdr->protocol <= 17) {
        fprintf(fp, "(%s)\n", Proto[ip_hdr->protocol]);
    }
    else {
        fprintf(fp, "(undefined)\n");
    }
    fprintf(fp, "check=%x\n", ip_hdr->check);
    fprintf(fp, "saddr=%s\n", ip_ip2str(ip_hdr->saddr, buf, sizeof(buf)));
    fprintf(fp, "daddr=%s\n", ip_ip2str(ip_hdr->daddr, buf, sizeof(buf)));
    if (optionLen > 0) {
        fprintf(fp, "option:");
        for (i = 0; i < optionLen; i++) {
            if (i != 0) {
                fprintf(fp, ":%02x", option[i]);
            }
            else {
                fprintf(fp, "%02x", option[i]);
            }
        }
    }

    return 0;
}

/**
 * @brief IPv6ヘッダを表示する
 * @param ip6_hdr : IPv6ヘッダ情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintIp6Header(struct ip6_hdr *ip6_hdr, FILE *fp)
{
    char buf[80];

    fprintf(fp, "\nip6-------------------------------------\n");

    fprintf(fp, "ip6_flow=%x\n", ip6_hdr->ip6_flow);
    fprintf(fp, "ip6_plen=%d\n", ntohs(ip6_hdr->ip6_plen));
    fprintf(fp, "ip6_nxt=%u", ip6_hdr->ip6_nxt);
    if (ip6_hdr->ip6_nxt <= 17) {
        fprintf(fp, "(%s)\n", Proto[ip6_hdr->ip6_nxt]);
    }
    else {
        fprintf(fp, "(undefined\n");
    }
    fprintf(fp, "ip6_hlim=%d\n", ip6_hdr->ip6_hlim);

    fprintf(fp, "ip6_src=%s\n", inet_ntop(AF_INET6, &ip6_hdr->ip6_src, buf, sizeof(buf)));
    fprintf(fp, "ip6_dst=%s\n", inet_ntop(AF_INET6, &ip6_hdr->ip6_dst, buf, sizeof(buf)));

    return 0;
}

/**
 * @brief ICMP情報を表示する
 * @param icmp : ICMP情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintIcmp(struct icmp *icmp, FILE *fp)
{
    static char *icmp_type[] = {
        "Echo Reply",
        "undefined",
        "undefined",
        "Destination Unreachable",
        "Source Quench",
        "Redirect",
        "undefined",
        "undefined",
        "Echo Request",
        "Router Adverisement",
        "Router Selection",
        "Time Exceeded for Datagram",
        "Parameter Problem on Datagram",
        "Timestamp Request",
        "Timestamp Reply",
        "Information Request",
        "Information Reply",
        "Address Mask Request",
        "Address Mask Reply"};

    fprintf(fp, "\nicmp------------------------------------\n");

    fprintf(fp, "icmp_type=%u", icmp->icmp_type);
    if (icmp->icmp_type <= 18) {
        fprintf(fp, "(%s)\n", icmp_type[icmp->icmp_type]);
    }
    else {
        fprintf(fp, "(undefined)\n");
    }
    fprintf(fp, "icmp_code=%u\n", icmp->icmp_code);
    fprintf(fp, "icmp_cksum=%u\n", ntohs(icmp->icmp_cksum));

    if (icmp->icmp_type == 0 || icmp->icmp_type == 8) {
        fprintf(fp, "icmp_id=%u\n", ntohs(icmp->icmp_id));
        fprintf(fp, "icmp_seq=%u\n", ntohs(icmp->icmp_seq));
    }

    return 0;
}

/**
 * @brief ICMP6情報を表示する
 * @param icmp6 : ICMP6情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintIcmp6(struct icmp6_hdr *icmp6, FILE *fp)
{
    fprintf(fp, "\nicmp6-----------------------------------\n");

    fprintf(fp, "icmp6_type=%u", icmp6->icmp6_type);
    if (icmp6->icmp6_type == 1) {
        fprintf(fp, "(Destination Unreachable)\n");
    }
    else if (icmp6->icmp6_type == 2) {
        fprintf(fp, "(Packet too Big)\n");
    }
    else if (icmp6->icmp6_type == 3) {
        fprintf(fp, "(Time Exceeded)\n");
    }
    else if (icmp6->icmp6_type == 4) {
        fprintf(fp, "(Parameter Problem)\n");
    }
    else if (icmp6->icmp6_type == 128) {
        fprintf(fp, "(Echo Request)\n");
    }
    else if (icmp6->icmp6_type == 129) {
        fprintf(fp, "(Echo Reply)\n");
    }
    else {
        fprintf(fp, "(undefined)\n");
    }
    fprintf(fp, "icmp6_code=%u\n", icmp6->icmp6_code);
    fprintf(fp, "icmp6_cksum=%u\n", ntohs(icmp6->icmp6_cksum));

    if (icmp6->icmp6_type == 128 || icmp6->icmp6_type == 129) {
        fprintf(fp, "icmp6_id=%u\n", ntohs(icmp6->icmp6_id));
        fprintf(fp, "icmp6_seq=%u\n", ntohs(icmp6->icmp6_seq));
    }

    return 0;
}

/**
 * @brief TCP情報を表示する
 * @param tcp : TCP情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintTcp(struct tcphdr *tcp, FILE *fp)
{
    fprintf(fp, "\ntcp-------------------------------------\n");

    fprintf(fp, "source=%u\n", ntohs(tcp->source));
    fprintf(fp, "dest=%u\n", ntohs(tcp->dest));
    fprintf(fp, "seq=%u\n", ntohl(tcp->seq));
    fprintf(fp, "ack_seq=%u\n", ntohl(tcp->ack_seq));
    fprintf(fp, "doff=%u\n", tcp->doff);
    fprintf(fp, "urg=%u\n", tcp->urg);
    fprintf(fp, "ack=%u\n", tcp->ack);
    fprintf(fp, "psh=%u\n", tcp->psh);
    fprintf(fp, "rst=%u\n", tcp->rst);
    fprintf(fp, "syn=%u\n", tcp->syn);
    fprintf(fp, "fin=%u\n", tcp->fin);
    fprintf(fp, "th_win=%u\n", ntohs(tcp->window));
    fprintf(fp, "th_sum=%u\n", ntohs(tcp->check));
    fprintf(fp, "th_urp=%u\n", ntohs(tcp->urg_ptr));

    return 0;
}

/**
 * @brief UDP情報を表示する
 * @param udp : UDP情報
 * @param fp : 出力先ファイルポインタ
 * @return 成功
 */
int PrintUdp(struct udphdr *udp, FILE *fp)
{
    fprintf(fp, "\nudp-------------------------------------\n");

    fprintf(fp, "source=%u\n", ntohs(udp->source));
    fprintf(fp, "dest=%u\n", ntohs(udp->dest));
    fprintf(fp, "len=%u\n", ntohs(udp->len));
    fprintf(fp, "check=%x\n", ntohs(udp->check));

    return 0;
}
