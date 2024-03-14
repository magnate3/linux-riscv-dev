
#ifndef _IP6_FRAG_H_
#define _IP6_FRAG_H_
#define RTE_IPV6_MIN_MTU 1280 /**< Minimum MTU for IPv6, see RFC 8200. */
/** IPv6 fragment extension header. */
#define RTE_IPV6_EHDR_MF_SHIFT  0
#define RTE_IPV6_EHDR_MF_MASK   1
#define RTE_IPV6_EHDR_FO_SHIFT  3
#define RTE_IPV6_EHDR_FO_MASK   (~((1 << RTE_IPV6_EHDR_FO_SHIFT) - 1))
#define RTE_IPV6_EHDR_FO_ALIGN  (1 << RTE_IPV6_EHDR_FO_SHIFT)

#define RTE_IPV6_FRAG_USED_MASK (RTE_IPV6_EHDR_MF_MASK | RTE_IPV6_EHDR_FO_MASK)

#define RTE_IPV6_GET_MF(x)      ((x) & RTE_IPV6_EHDR_MF_MASK)
#define RTE_IPV6_GET_FO(x)      ((x) >> RTE_IPV6_EHDR_FO_SHIFT)

#define RTE_IPV6_SET_FRAG_DATA(fo, mf)  \
        (((fo) & RTE_IPV6_EHDR_FO_MASK) | ((mf) & RTE_IPV6_EHDR_MF_MASK))
struct rte_ipv6_fragment_ext {
        uint8_t next_header;    /**< Next header type */
        uint8_t reserved;       /**< Reserved */
        rte_be16_t frag_data;   /**< All fragmentation data */
        rte_be32_t id;          /**< Packet ID */
} __rte_packed;
#endif
