

#  sizeof(struct srh)


```
#include<stdio.h>
#include<stdint.h>
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
int main()
{
    printf("sizeof(struct srh): %u \n",sizeof(struct srh));
    return 0;
}
```

```
sizeof(struct srh): 8 
```


```
static __always_inline int add_srh(struct __sk_buff *skb, void *data,
								   void *data_end,
								   struct sidlist_data *sidlist_data)
{
	struct ipv6hdr *ipv6 = data + sizeof(struct ethhdr);
	if (data + sizeof(struct ethhdr) + sizeof(struct ipv6hdr) > data_end)
		return -1;

	int hdr_ext_len =
		(sizeof(struct srh) +
		 sizeof(struct in6_addr) * sidlist_data->sidlist_size - 8) /
		8;

	struct srh srh = {
		.next_hdr = ipv6->nexthdr,
		.hdr_ext_len = hdr_ext_len,
		.routing_type = SRV6_ROUTING_TYPE,
		.segments_left = sidlist_data->sidlist_size - 1,
		.last_entry = sidlist_data->sidlist_size - 1,
	};
}
```

+ ipv6  payload_len变化    

```
ipv6->payload_len =
		bpf_htons(bpf_ntohs(ipv6->payload_len) + sizeof(struct srh) +
				  sizeof(struct in6_addr) * sidlist_data->sidlist_size);
```
+ ipv6->nexthdr     
```
#define SRV6_NEXT_HDR 43	/* Routing header. */
	ipv6->nexthdr = SRV6_NEXT_HDR;
```

# sizeof(struct srh) + sizeof(struct in6_addr) * sidlist_size == srh_get_hdr_len(struct srh *hdr)

```
[root@centos7 ~]# ./test2 
hdr_ext_len: 2, srh_get_hdr_len: 24 
sizeof(struct srh): 8 
sizeof(struct srh) + sizeof(struct in6_addr) * sidlist_size : 24, equal: 24 
ipv6_payload_len change to : 152, equal: 152 
```