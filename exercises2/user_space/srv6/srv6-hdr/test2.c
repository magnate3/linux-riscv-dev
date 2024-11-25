#include<stdio.h>
#include<stdint.h>
#define SRV6_NEXT_HDR 43	/* Routing header. */
#define SRV6_HDR_EXT_LEN 0	/* Routing header extension length. */
#define SRV6_ROUTING_TYPE 4 /* SRv6 routing type. */
#define IPV6_PAYLOAD_LEN 128
struct in6_addr
{
        union 
        {
                uint8_t  u6_addr8[16];
                uint16_t u6_addr16[8];
                uint32_t  u6_addr32[4];
        } in6_u;
#define s6_addr                  in6_u.u6_addr8
#define s6_addr16                in6_u.u6_addr16
#define s6_addr32                in6_u.u6_addr32
};
struct srh {
	uint8_t next_hdr;
	uint8_t hdr_ext_len;
	uint8_t routing_type;
	uint8_t segments_left;
	uint8_t last_entry;
	uint8_t flags;
	uint16_t tag;
	struct in6_addr segments[0];
} __attribute__((packed));
static __always_inline int srh_get_hdr_len(struct srh *hdr)
{
	return (hdr->hdr_ext_len + 1) * 8;
}

static __always_inline int srh_get_segment_list_len(struct srh *hdr)
{
	return hdr->last_entry + 1;
}

int main()
{
    uint8_t sidlist_size = 1;
    int hdr_ext_len =
		(sizeof(struct srh) +
		 sizeof(struct in6_addr) *sidlist_size  - 8) /
		8;
    uint32_t  ipv6_payload_len =  128;
    struct srh srh = {
		.next_hdr = 0x11,
		.hdr_ext_len = hdr_ext_len,
		.routing_type = SRV6_ROUTING_TYPE,
		.segments_left = sidlist_size - 1,
		.last_entry = sidlist_size - 1,
    };
    ipv6_payload_len = htons(ipv6_payload_len + sizeof(struct srh) + sizeof(struct in6_addr) * sidlist_size);
    printf("hdr_ext_len: %u, srh_get_hdr_len: %u \n",hdr_ext_len, srh_get_hdr_len(&srh));
    printf("sizeof(struct srh): %u \n",sizeof(struct srh));
    printf("sizeof(struct srh) + sizeof(struct in6_addr) * sidlist_size : %u, equal: %u \n",sizeof(struct srh) + sizeof(struct in6_addr) * sidlist_size,\
    srh_get_hdr_len(&srh));
    printf("ipv6_payload_len change to : %u, equal: %u \n",ntohs(ipv6_payload_len), IPV6_PAYLOAD_LEN + srh_get_hdr_len(&srh));
    return 0;
}
