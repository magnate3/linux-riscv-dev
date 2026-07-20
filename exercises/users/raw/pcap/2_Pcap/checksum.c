/**
 * @file checksum.c
 * @brief チェックサムをチャックするための関数群の実装ファイル
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

// IPアドレスの疑似ヘッダ
struct pseudo_ip {
    struct in_addr ip_src;
    struct in_addr ip_dst;
    unsigned char dummy;
    unsigned char ip_p;
    unsigned short ip_len;
};

// IPv6アドレスの疑似ヘッダ
struct pseudo_ip6_hdr {
    struct in6_addr src;
    struct in6_addr dst;
    unsigned long plen;
    unsigned short dmy1;
    unsigned char dmy2;
    unsigned char nxt;
};

/**
 * @brief チェックサムを計算する
 * @param data : データ
 * @param len : データの長さ
 * @return チェックサム
 */
u_int16_t checksum(u_char *data, int len)
{
    register u_int32_t sum = 0;
    register u_int16_t *ptr = NULL;
    register int c = 0;
    u_int16_t val = 0;

    ptr = (u_int16_t *)data;

    for (c = len; c > 1; c -= 2) {
        sum += (*ptr);
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ptr++;
    }
    if (c == 1) {
        memcpy(&val, ptr, sizeof(u_int8_t));
        sum += val;
    }

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

/**
 * @brief 2つのデータのチェックサムを計算する
 * @param data1 : データ1
 * @param len1 : データ1の長さ
 * @param data2 : データ2
 * @param len2 : データ2の長さ
 * @return チェックサム
 */
u_int16_t checksum2(u_char *data1, int len1, u_char *data2, int len2)
{
    register u_int32_t sum = 0;
    register u_int16_t *ptr = NULL;
    register int c = 0;
    u_int16_t val = 0;

    sum = 0;
    ptr = (u_int16_t *)data1;
    for (c = len1; c > 1; c -= 2) {
        sum += (*ptr);
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ptr++;
    }

    if (c == 1) {
        val = ((*ptr) << 8) + (*data2);
        sum += val;
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ptr = (u_int16_t *)(data2 + 1);
        len2--;
    }
    else {
        ptr = (u_int16_t *)data2;
    }

    for (c = len2; c > 1; c -= 2) {
        sum += (*ptr);
        if (sum & 0x80000000) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        ptr++;
    }
    if (c == 1) {
        val = 0;
        memcpy(&val, ptr, sizeof(u_int8_t));
        sum += val;
    }

    while (sum >> 16) {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }

    return ~sum;
}

/**
 * @brief IPヘッダのチェックサムを確認する
 * @param ip_hdr : IPヘッダ
 * @param option : オプション
 * @param optionLen : オプションの長さ
 * @return 1 : チェックサムが 0 or 65535
 *         0 : ↑以外
 */
int checkIPchecksum(struct iphdr *ip_hdr, u_char *option, int optionLen)
{
    unsigned short sum = 0;

    if (optionLen == 0) {
        sum = checksum((u_char *)ip_hdr, sizeof(struct iphdr));
        if (sum == 0 || sum == 0xFFFF) {
            return 1;
        }
        else {
            return 0;
        }
    }
    else {
        sum = checksum2((u_char *)ip_hdr, sizeof(struct iphdr), option, optionLen);
        if (sum == 0 || sum == 0xFFFF) {
            return 1;
        }
        else {
            return 0;
        }
    }
}

/**
 * @brief IPのTCP, UDPのチェックサムを確認する
 * @param ip_hdr : IPヘッダ
 * @param data : TCP or UDP のデータ
 * @param len : データの長さ
 * @return 1 : チェックサムが 0 or 65535
 *         0 : ↑以外
 */
int checkIPDATAchecksum(struct iphdr *ip_hdr, unsigned char *data, int len)
{
    struct pseudo_ip p_ip;
    unsigned short sum = 0;

    memset(&p_ip, 0, sizeof(struct pseudo_ip));
    p_ip.ip_src.s_addr = ip_hdr->saddr;
    p_ip.ip_dst.s_addr = ip_hdr->daddr;
    p_ip.ip_p = ip_hdr->protocol;
    p_ip.ip_len = htons(len);

    sum = checksum2((unsigned char *)&p_ip, sizeof(struct pseudo_ip), data, len);
    if (sum == 0 || sum == 0xFFFF) {
        return 1;
    }
    else {
        return 0;
    }
}

/**
 * @brief IPv6のTCP, UDP, ICMPのチェックサムを確認する
 * @param ip_hdr : IPヘッダ
 * @param data : TCP or UDP のデータ
 * @param len : データの長さ
 * @return 1 : チェックサムが 0 or 65535
 *         0 : ↑以外
 */
int checkIP6DATAchecksum(struct ip6_hdr *ip, unsigned char *data, int len)
{
    struct pseudo_ip6_hdr p_ip;
    unsigned short sum = 0;

    memset(&p_ip, 0, sizeof(struct pseudo_ip6_hdr));

    memcpy(&p_ip.src, &ip->ip6_src, sizeof(struct in6_addr));
    memcpy(&p_ip.dst, &ip->ip6_dst, sizeof(struct in6_addr));
    p_ip.plen = ip->ip6_plen;
    p_ip.nxt = ip->ip6_nxt;

    sum = checksum2((unsigned char *)&p_ip, sizeof(struct pseudo_ip6_hdr), data, len);
    if (sum == 0 || sum == 0xFFFF) {
        return 1;
    }
    else {
        return 0;
    }
}
