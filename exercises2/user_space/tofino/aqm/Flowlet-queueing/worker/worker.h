#ifndef __WORKER_H_
#define __WORKER_H_

#include <stdlib.h>
#include <stdint.h>

#define INTERFACE_NAME "eth8"

uint8_t SRC_MAC[] = {0x40, 0xa6, 0xb7, 0x6f, 0x21, 0xd9};
uint8_t DST_MAC[] = {0x40, 0xa6, 0xb7, 0x6f, 0x21, 0xda};
char* dest_ip = "10.0.0.200";
char* src_ip = "10.0.0.1";
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

struct udp_hdr {
	uint16_t src_port;    /**< UDP source port. */
	uint16_t dst_port;    /**< UDP destination port. */
	uint16_t dgram_len;   /**< UDP datagram length */
	uint16_t dgram_cksum; /**< UDP datagram checksum */
} __attribute__((__packed__));

struct worker_hdr {
	uint32_t queue_length;
	uint32_t qid;
	uint32_t round;
	uint32_t round_index;
	uint32_t counter;
} __attribute__((__packed__));

#endif