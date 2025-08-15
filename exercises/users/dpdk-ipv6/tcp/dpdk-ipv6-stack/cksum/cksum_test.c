#include<stdio.h>
#include<stdint.h>
#include<string.h>
#include <netinet/ip6.h>
#include <arpa/inet.h>
typedef uint16_t __sum16;
typedef uint32_t u32;
typedef uint32_t __u32;
typedef unsigned char __u8;
typedef unsigned int __wsum;
//typedef uint16_t __bitwise __sum16;
//arch/arm64/include/asm/checksum.h
static inline __sum16 csum_fold(__wsum csum)
{
        u32 sum = ( u32)csum;
        sum += (sum >> 16) | (sum << 16);
        return ~( __sum16)(sum >> 16);
}
__sum16 csum_ipv6_magic(const struct in6_addr *saddr,
                        const struct in6_addr *daddr,
                        __u32 len, __u8 proto, __wsum csum)
{

        int carry;
        __u32 ulen;
        __u32 uproto;
        __u32 sum = ( u32)csum;

        sum += ( u32)saddr->s6_addr32[0];
        carry = (sum < ( u32)saddr->s6_addr32[0]);
        sum += carry;

        sum += ( u32)saddr->s6_addr32[1];
        carry = (sum < ( u32)saddr->s6_addr32[1]);
        sum += carry;

        sum += ( u32)saddr->s6_addr32[2];
        carry = (sum < ( u32)saddr->s6_addr32[2]);
        sum += carry;

        sum += ( u32)saddr->s6_addr32[3];
        carry = (sum < ( u32)saddr->s6_addr32[3]);
        sum += carry;

        sum += ( u32)daddr->s6_addr32[0];
        carry = (sum < ( u32)daddr->s6_addr32[0]);
        sum += carry;

        sum += ( u32)daddr->s6_addr32[1];
        carry = (sum < ( u32)daddr->s6_addr32[1]);
        sum += carry;

        sum += ( u32)daddr->s6_addr32[2];
        carry = (sum < ( u32)daddr->s6_addr32[2]);
        sum += carry;

        sum += ( u32)daddr->s6_addr32[3];
        carry = (sum < ( u32)daddr->s6_addr32[3]);
        sum += carry;

        ulen = ( u32)htonl((__u32) len);
        sum += ulen;
        carry = (sum < ulen);
        sum += carry;

        uproto = ( u32)htonl(proto);
        sum += uproto;
        carry = (sum < uproto);
        sum += carry;

        return csum_fold(( __wsum)sum);
}
int main()
 {
	  
   return 0;  
  }
