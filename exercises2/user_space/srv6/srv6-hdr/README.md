

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