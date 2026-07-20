

大多数计算机都采用小端字节序（little-endian），即将低位字节存储在内存的低地址处，高位字节存储在内存的高地址处。`而网络通信使用的则是大端字节序（big-endian）`，即将高位字节存储在内存的低地址处，低位字节存储在内存的高地址处。

```Text
例如，在小端字节序下，整数值0x12345678被表示为以下四个连续的8位二进制数：

 0x78  0x56  0x34  0x12
而在大端字节序下，则应该按照以下顺序进行排列：

 0x12  0x34  0x56  0x78
```

```
#include <stdio.h>
#include <stdint.h>

#include <string.h>



int
main(int argc, char **argv)
{
    rte_be16_t rte_value  = 0x1214;
    uint16_t value = 0x1214;
    char *byte = (char*)&value;
    printf("%d,,%d\n",byte[0], byte[1]);
    byte = (char*)&rte_value;
    printf("%d,,%d\n",byte[0], byte[1]);
    return 0;
}
```


```
[root@centos7 test]# gcc mem.c  -o mem
[root@centos7 test]# ./mem
20,,18
20,,18
[root@centos7 test]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 test]# 
```


```

/*
 * The following types should be used when handling values according to a
 * specific byte ordering, which may differ from that of the host CPU.
 *
 * Libraries, public APIs and applications are encouraged to use them for
 * documentation purposes.
 */
typedef uint16_t rte_be16_t; /**< 16-bit big-endian value. */
typedef uint32_t rte_be32_t; /**< 32-bit big-endian value. */
typedef uint64_t rte_be64_t; /**< 64-bit big-endian value. */
typedef uint16_t rte_le16_t; /**< 16-bit little-endian value. */
typedef uint32_t rte_le32_t; /**< 32-bit little-endian value. */
typedef uint64_t rte_le64_t; /**< 64-bit little-endian value. */
```


```
/**
 * IPv4 Header
 */
struct rte_ipv4_hdr {
        uint8_t  version_ihl;           /**< version and header length */
        uint8_t  type_of_service;       /**< type of service */
        rte_be16_t total_length;        /**< length of packet */
        rte_be16_t packet_id;           /**< packet ID */
        rte_be16_t fragment_offset;     /**< fragmentation offset */
        uint8_t  time_to_live;          /**< time to live */
        uint8_t  next_proto_id;         /**< protocol ID */
        rte_be16_t hdr_checksum;        /**< header checksum */
        rte_be32_t src_addr;            /**< source address */
        rte_be32_t dst_addr;            /**< destination address */
} __attribute__((__packed__));
```


```
librte_eal/common/include/arch/ppc_64/rte_byteorder.h:86:#define rte_cpu_to_be_32(x) rte_bswap32(x)
librte_eal/common/include/arch/ppc_64/rte_byteorder.h:104:#define rte_cpu_to_be_32(x) (x)
librte_eal/common/include/arch/x86/rte_byteorder.h:78:#define rte_cpu_to_be_32(x) rte_bswap32(x)
librte_eal/common/include/arch/arm/rte_byteorder.h:47:#define rte_cpu_to_be_32(x) rte_bswap32(x)
librte_eal/common/include/arch/arm/rte_byteorder.h:65:#define rte_cpu_to_be_32(x) (x)
librte_eal/common/include/generic/rte_byteorder.h:195:static rte_be32_t rte_cpu_to_be_32(uint32_t x);
```

##   arm/rte_byteorder.h

```

/* ARM architecture is bi-endian (both big and little). */
#if RTE_BYTE_ORDER == RTE_LITTLE_ENDIAN

#define rte_cpu_to_le_16(x) (x)
#define rte_cpu_to_le_32(x) (x)
#define rte_cpu_to_le_64(x) (x)

#define rte_cpu_to_be_16(x) rte_bswap16(x)
#define rte_cpu_to_be_32(x) rte_bswap32(x)
#define rte_cpu_to_be_64(x) rte_bswap64(x)

#define rte_le_to_cpu_16(x) (x)
#define rte_le_to_cpu_32(x) (x)
#define rte_le_to_cpu_64(x) (x)

#define rte_be_to_cpu_16(x) rte_bswap16(x)
#define rte_be_to_cpu_32(x) rte_bswap32(x)
#define rte_be_to_cpu_64(x) rte_bswap64(x)

#else /* RTE_BIG_ENDIAN */

#define rte_cpu_to_le_16(x) rte_bswap16(x)
#define rte_cpu_to_le_32(x) rte_bswap32(x)
#define rte_cpu_to_le_64(x) rte_bswap64(x)

#define rte_cpu_to_be_16(x) (x)
#define rte_cpu_to_be_32(x) (x)
#define rte_cpu_to_be_64(x) (x)

#define rte_le_to_cpu_16(x) rte_bswap16(x)
#define rte_le_to_cpu_32(x) rte_bswap32(x)
#define rte_le_to_cpu_64(x) rte_bswap64(x)

#define rte_be_to_cpu_16(x) (x)
#define rte_be_to_cpu_32(x) (x)
#define rte_be_to_cpu_64(x) (x)
#endif
```