#ifndef _COMMON_H
#define _COMMON_H

/* 1) Includes */
#include <stdint.h>
#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <linux/udp.h>
#include "../libbpf/src/bpf_helpers.h"
#include "../libbpf/src/bpf_endian.h"

/* 2) Defines */
#define MAX_QUERY_LENGTH 50
#define MAX_ALLOWED_QUERY_LENGTH 30
#define BLOCK_TIME 5000000000 //5 secs
#define NUM_OF_SERVERS 2
#define INITIAL_FWD_PORT 60000
#define MAC_SIZE 6

struct forwarding_server{
	uint32_t ip_addr;
	char mac[MAC_SIZE];
};

struct _dns_query{
	uint16_t qtype;
	uint16_t qclass;
	uint16_t qlength;
	char qname[MAX_QUERY_LENGTH];		
};

//extracted from internet - known header
struct _dns_hdr{
	uint16_t id;

	uint8_t rd	:1;//recursion desired
	uint8_t tc	:1;//truncated message
	uint8_t aa	:1;//authoritive answer
	uint8_t opcode	:4;//purpose of message
	uint8_t qr	:1;//query/response flag

	uint8_t rcode	:4;//response code
	uint8_t cd	:1;//checking desiabled
	uint8_t ad	:1;//authenticated data
	uint8_t z	:1;// reserved
	uint8_t ra	:1;//recursion available

	uint16_t qdcount;//number of question entries
	uint16_t ancount;//number of answer entries
	uint16_t nscount;//number of authority entries
	uint16_t arcount;//number of resource entries
};

struct _packets_counters{
	uint64_t dropped_packets_name;
	uint64_t dropped_packets_length;
	uint64_t already_blocked;
	uint64_t passed_packets;	
};


/* 4) Maps */ 
struct {
	__uint(type, BPF_MAP_TYPE_PROG_ARRAY);
	__uint(key_size, sizeof(uint32_t));
	__uint(value_size, sizeof(uint32_t));
	__uint(max_entries, 3);
}map_xdp_progs SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(key_size, sizeof(uint32_t));
	__uint(value_size, sizeof(uint32_t));
	__uint(max_entries, 2);//TODO #define
	__uint(map_flags, 1);
}map_illegal_domains SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(key_size, sizeof(uint32_t)); //IP address
	__uint(value_size, sizeof(struct _dns_query)); 
	__uint(max_entries, 100);//TODO #define
	__uint(map_flags, 1);
}query SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(key_size, sizeof(uint32_t));
	__uint(value_size, sizeof(struct _packets_counters));
	__uint(max_entries, 1);
}map_packets_counters SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(key_size, sizeof(uint32_t)); //IP addr
	__uint(value_size, sizeof(uint64_t)); //time in ns
	__uint(max_entries, 1000);	
}map_blocked_requesters SEC(".maps");

#endif



