/*
 * Work-around of a compilation error with ICC on invocations of the
 * rte_be_to_cpu_16() function.
 */
#ifdef __GCC__
#define RTE_BE_TO_CPU_16(be_16_v)  rte_be_to_cpu_16((be_16_v))
#define RTE_CPU_TO_BE_16(cpu_16_v) rte_cpu_to_be_16((cpu_16_v))
#else
#if RTE_BYTE_ORDER == RTE_BIG_ENDIAN
#define RTE_BE_TO_CPU_16(be_16_v)  (be_16_v)
#define RTE_CPU_TO_BE_16(cpu_16_v) (cpu_16_v)
#else
#define RTE_BE_TO_CPU_16(be_16_v) \
        (uint16_t) ((((be_16_v) & 0xFF) << 8) | ((be_16_v) >> 8))
#define RTE_CPU_TO_BE_16(cpu_16_v) \
        (uint16_t) ((((cpu_16_v) & 0xFF) << 8) | ((cpu_16_v) >> 8))
#endif
#define DEV_TX_OFFLOAD_MBUF_FAST_FREE       0x00010000
#define RTE_ETHER_ADDR_LEN  6
#define RTE_ETHER_ADDR_FMT_SIZE         18
/* Ethernet frame types */
#define RTE_ETHER_TYPE_IPV4 0x0800 /**< IPv4 Protocol. */
#define RTE_ETHER_TYPE_IPV6 0x86DD /**< IPv6 Protocol. */
#define RTE_ETHER_TYPE_ARP  0x0806 /**< Arp Protocol. */
#define RTE_ETHER_TYPE_RARP 0x8035 /**< Reverse Arp Protocol. */
#define RTE_ETHER_TYPE_VLAN 0x8100 /**< IEEE 802.1Q VLAN tagging. */
typedef uint16_t rte_be16_t; /**< 16-bit big-endian value. */
typedef uint32_t rte_be32_t; /**< 32-bit big-endian value. */
typedef uint64_t rte_be64_t; /**< 64-bit big-endian value. */
typedef uint16_t rte_le16_t; /**< 16-bit little-endian value. */
typedef uint32_t rte_le32_t; /**< 32-bit little-endian value. */
typedef uint64_t rte_le64_t; /**< 64-bit little-endian value. */
struct rte_ether_addr {
        uint8_t addr_bytes[RTE_ETHER_ADDR_LEN]; /**< Addr bytes in tx order */
} __attribute__((aligned(2)));
struct rte_arp_ipv4 {
        struct rte_ether_addr arp_sha;  /**< sender hardware address */
        uint32_t          arp_sip;  /**< sender IP address */
        struct rte_ether_addr arp_tha;  /**< target hardware address */
        uint32_t          arp_tip;  /**< target IP address */
} __attribute__((__packed__)) __attribute__((aligned(2)));

struct rte_arp_hdr {
        uint16_t arp_hardware;    /* format of hardware address */
#define RTE_ARP_HRD_ETHER     1  /* ARP Ethernet address format */

        uint16_t arp_protocol;    /* format of protocol address */
        uint8_t  arp_hlen;    /* length of hardware address */
        uint8_t  arp_plen;    /* length of protocol address */
        uint16_t arp_opcode;     /* ARP opcode (command) */
#define RTE_ARP_OP_REQUEST    1 /* request to resolve address */
#define RTE_ARP_OP_REPLY      2 /* response to previous request */
#define RTE_ARP_OP_REVREQUEST 3 /* request proto addr given hardware */
#define RTE_ARP_OP_REVREPLY   4 /* response giving protocol address */
#define RTE_ARP_OP_INVREQUEST 8 /* request to identify peer */
#define RTE_ARP_OP_INVREPLY   9 /* response identifying peer */

        struct rte_arp_ipv4 arp_data;
} __attribute__((__packed__)) __attribute__((aligned(2)));
/**
 * ICMP Header
 */
struct rte_icmp_hdr {
        uint8_t  icmp_type;     /* ICMP packet type. */
        uint8_t  icmp_code;     /* ICMP packet code. */
        rte_be16_t icmp_cksum;  /* ICMP packet checksum. */
        rte_be16_t icmp_ident;  /* ICMP packet identifier. */
        rte_be16_t icmp_seq_nb; /* ICMP packet sequence number. */
} __attribute__((__packed__));

/* ICMP packet types */
#define RTE_IP_ICMP_ECHO_REPLY   0
#define RTE_IP_ICMP_ECHO_REQUEST 8
struct rte_ether_hdr {
        struct rte_ether_addr d_addr; /**< Destination address. */
        struct rte_ether_addr s_addr; /**< Source address. */
        uint16_t ether_type;      /**< Frame type. */
} __attribute__((aligned(2)));

/**
 * Ethernet VLAN Header.
 * Contains the 16-bit VLAN Tag Control Identifier and the Ethernet type
 * of the encapsulated frame.
 */
struct rte_vlan_hdr {
        uint16_t vlan_tci; /**< Priority (3) + CFI (1) + Identifier Code (12) */
        uint16_t eth_proto;/**< Ethernet type of encapsulated frame. */
} __attribute__((__packed__));
/** Create IPv4 address */
#define RTE_IPV4(a, b, c, d) ((uint32_t)(((a) & 0xff) << 24) | \
                                           (((b) & 0xff) << 16) | \
                                           (((c) & 0xff) << 8)  | \
                                           ((d) & 0xff))
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

static inline void rte_ether_addr_copy(const struct rte_ether_addr *ea_from,
                                   struct rte_ether_addr *ea_to)
{
#ifdef __INTEL_COMPILER
        uint16_t *from_words = (uint16_t *)(ea_from->addr_bytes);
        uint16_t *to_words   = (uint16_t *)(ea_to->addr_bytes);

        to_words[0] = from_words[0];
        to_words[1] = from_words[1];
        to_words[2] = from_words[2];
#else
        /*
         * Use the common way, because of a strange gcc warning.
         */
        *ea_to = *ea_from;
#endif
}
typedef void (*buffer_tx_error_fn)(struct rte_mbuf **unsent, uint16_t count,
                void *userdata);

/**
 * Structure used to buffer packets for future TX
 * Used by APIs rte_eth_tx_buffer and rte_eth_tx_buffer_flush
 */
struct rte_eth_dev_tx_buffer {
        buffer_tx_error_fn error_callback;
        void *error_userdata;
        uint16_t size;           /**< Size of buffer for buffered tx */
        uint16_t length;         /**< Number of packets in the array */
        struct rte_mbuf *pkts[];
        /**< Pending packets to be sent on explicit flush or when full */
};

/**
 * Calculate the size of the tx buffer.
 *
 * @param sz
 *   Number of stored packets.
 */
#define RTE_ETH_TX_BUFFER_SIZE(sz) \
        (sizeof(struct rte_eth_dev_tx_buffer) + (sz) * sizeof(struct rte_mbuf *))
#if 0
uint16_t
rte_eth_dev_count_avail(void)
{
        uint16_t p;
        uint16_t count;

        count = 0;

        RTE_ETH_FOREACH_DEV(p)
                count++;

        return count;
}
#endif
#endif /* __GCC__ */
