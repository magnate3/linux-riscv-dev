#ifndef __SENDER_H_
#define __SENDER_H_

#include <stdlib.h>
#include <stdint.h>

uint8_t SRC_MAC[] = {0xb0, 0x6e, 0xbf, 0xc4, 0xb1, 0x1a};
uint8_t DST_MAC[] = {0xb0, 0x6e, 0xbf, 0xc4, 0xb4, 0x79};
char* dest_ip = "10.0.0.200";
char* src_ip = "10.0.0.15";
struct eth_header {
    uint8_t dmac[6];
    uint8_t smac[6];
    uint16_t ethType;
};

struct ipv4_hdr {
	uint8_t  version_ihl;		/**< version and header length */
	uint8_t  type_of_service;	/**< type of service */
	uint16_t total_length;		/**< length of packet */
	uint16_t packet_id;		/**< packet ID */
	uint16_t fragment_offset;	/**< fragmentation offset */
	uint8_t  time_to_live;		/**< time to live */
	uint8_t  next_proto_id;		/**< protocol ID */
	uint16_t hdr_checksum;		/**< header checksum */
	uint32_t src_addr;		/**< source address */
	uint32_t dst_addr;		/**< destination address */
} __attribute__((__packed__));

struct worker_hdr {
	uint32_t round_number;
	uint32_t egress_port;
	uint32_t qid;
	uint32_t round_index;
} __attribute__((__packed__));

struct udp_hdr {
	uint16_t src_port;    /**< UDP source port. */
	uint16_t dst_port;    /**< UDP destination port. */
	uint16_t dgram_len;   /**< UDP datagram length */
	uint16_t dgram_cksum; /**< UDP datagram checksum */
} __attribute__((__packed__));

struct dune_header
{

} __attribute__((__packed__));

#endif