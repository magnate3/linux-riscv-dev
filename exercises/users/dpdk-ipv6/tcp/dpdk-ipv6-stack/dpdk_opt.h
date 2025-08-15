#ifndef __DPVS_OPT__
#define __DPVS_OPT__
#ifdef __KERNEL__
#include <asm/byteorder.h>
#else
#include <endian.h>
#endif
/**
 *  *  "Option Protocol": IPPROTO_OPT
 *  *
 *  *   0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7 0 1 2 3 4 5 6 7
 *  *  +---------------+---------------+---------------+--------------+
 *  *  |  Ver. | Rsvd. |    Protocol   |            Length            |
 *  *  +---------------+---------------+---------------+--------------+
 *  *  :                           Options                            :
 *  *  +---------------+---------------+---------------+--------------+
 *  *
 *  *  Ve.     Version, now 0x1 (1) for ipv4 address family, OPPHDR_IPV4
 *  *                       0x2 (2) for ipv6 address family, OPPHDR_IPV6
 *  *  Rsvd.   Reserved bits, must be zero.
 *  *  Protocol    Next level protocol, e.g., IPPROTO_UDP.
 *  *  Length    Length of fixed header and options, not include payloads.
 *  *  Options    Compatible with IPv4 options, including IPOPT_UOA.
 *  */

#define IPPROTO_OPT    0xf8 /* 248 */

#define OPPHDR_IPV6 0x02
#define OPPHDR_IPV4 0x01

/* OPtion Protocol header */
struct opphdr {
#if defined(__LITTLE_ENDIAN_BITFIELD) || (__BYTE_ORDER == __LITTLE_ENDIAN)
    unsigned int rsvd0:4;
    unsigned int version:4;
#elif defined (__BIG_ENDIAN_BITFIELD) || (__BYTE_ORDER == __BIG_ENDIAN)
    unsigned int version:4;
    unsigned int rsvd0:4;
#else
#ifndef __KERNEL__
# error    "Please fix <bits/endian.h>"
#else
# error    "Please fix <asm/byteorder.h>"
#endif
#endif
    __u8    protocol;    /* IPPROTO_XXX */
    __be16    length;        /* length of fixed header and options */
    __u8    options[0];
} __attribute__((__packed__));
#endif
