/*
 *  tayga.h -- main header file
 *
 *  part of TAYGA <http://www.litech.org/tayga/>
 *  Copyright (C) 2010  Nathan Lutchansky <lutchann@litech.org>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 */

#include <linux/in.h>
#include <linux/in6.h>
#include <stdio.h>
//#include <linux/list.h>
#include <linux/netdevice.h>
//#include <linux/proc_fs.h>

typedef __u8 u8;
typedef __u16 u16;
typedef __u32 u32;

#ifndef IN6_ARE_ADDR_EQUAL
#define IN6_ARE_ADDR_EQUAL(a,b)  (__extension__  \
	({ __const struct in6_addr *__a = (__const struct in6_addr *) (a); \
		__const struct in6_addr *__b = (__const struct in6_addr *) (b); \
		__a->s6_addr32[0] == __b->s6_addr32[0]  \
		&& __a->s6_addr32[1] == __b->s6_addr32[1]  \
		&& __a->s6_addr32[2] == __b->s6_addr32[2]  \
		&& __a->s6_addr32[3] == __b->s6_addr32[3]; }))
#endif

/* Number of seconds between dynamic pool ageing passes */
#define POOL_CHECK_INTERVAL	(5)


/* Protocol structures */

struct ip4 {
	u8 ver_ihl; /* 7-4: ver==4, 3-0: IHL */
	u8 tos;
	u16 length;
	u16 ident;
	u16 flags_offset; /* 15-13: flags, 12-0: frag offset */
	u8 ttl;
	u8 proto;
	u16 cksum;
	struct in_addr src;
	struct in_addr dest;
} __attribute__ ((__packed__));

#define IP4_F_DF	0x4000
#define IP4_F_MF	0x2000
#define IP4_F_MASK	0x1fff

struct ip6 {
	u32 ver_tc_fl; /* 31-28: ver==6, 27-20: traf cl, 19-0: flow lbl */
	u16 payload_length;
	u8 nexthdr;
	u8 hop_limit;
	struct in6_addr src;
	struct in6_addr dest;
} __attribute__ ((__packed__));

struct ip6_frag {
	u8 next_header;
	u8 reserved;
	u16 offset_flags; /* 15-3: frag offset, 2-0: flags */
	u32 ident;
} __attribute__ ((__packed__));

#define IP6_F_MF	0x0001
#define IP6_F_MASK	0xfff8

struct icmp {
	u8 type;
	u8 code;
	u16 cksum;
	u32 word;
} __attribute__ ((__packed__));

#define	WKPF	(htonl(0x0064ff9b))

/* Adjusting the MTU by 20 does not leave room for the IP6 fragmentation
   header, for fragments with the DF bit set.  Follow up with BEHAVE on this.

   (See http://www.ietf.org/mail-archive/web/behave/current/msg08499.html)
 */
#define MTU_ADJ		20



/* utilities */

static inline char *simple_inet6_ntoa(const struct in6_addr *a, char *s)
{
	sprintf(s, "%x:%x:%x:%x:%x:%x:%x:%x",
			ntohs(a->s6_addr16[0]), ntohs(a->s6_addr16[1]), 
			ntohs(a->s6_addr16[2]), ntohs(a->s6_addr16[3]), 
			ntohs(a->s6_addr16[4]), ntohs(a->s6_addr16[5]), 
			ntohs(a->s6_addr16[6]), ntohs(a->s6_addr16[7]));
	return s;
}

static inline char *simple_inet_ntoa(const struct in_addr *a, char *s)
{
	unsigned char *ap = (unsigned char *)&a->s_addr;
	sprintf(s, "%u.%u.%u.%u", ap[0], ap[1], ap[2], ap[3]);
	return s;
}


