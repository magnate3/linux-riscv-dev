/* Includes */
#include "common.h"

/* Defines */
#define XDP_PROG_VALIDATION 1
#define XDP_LOAD_BALANCER 2
#define DNS_PORT 53
#define DNS_QUERY_REQUEST 0
#define PACKETS_COUNTERS_MAP_KEY 0
#define bpf_printk2(fmt, ...)                            \
({                                                      \
        char ____fmt[] = fmt;                           \
        bpf_trace_printk(____fmt, sizeof(____fmt),      \
                         ##__VA_ARGS__);                \
})

/*
	struct xdp_md 
	{
	    __u32 data;
	    __u32 data_end;
	    __u32 data_meta;
	    __u32 ingress_ifindex;
	    __u32 rx_queue_index;  
	    __u32 egress_ifindex;
	};
	The packet contents lie between xdp_md.data and xdp_md.data_end
*/

/* Forward Declarations */

/* parse_query function:
 * takes a pointer to query start and returns query in dns_query struct
 * inputs: xdp_md - for sanity check - cursor can't be > data_end
 * 	   query_starts - pointer to the beggining of the query	
 * outputs:
 * 	returns FAILURE in case of sanity check failure
 * 		SUCCESS in case of success
 * 		dns_query *q - contains query 
 * */
static int parse_query(struct xdp_md *ctx, void *query_start, struct _dns_query *q);
/*
 * parse_host_domain function: our filter function - changeable according to NS requirments
 * takes a dns_query and returns domain according to filter spec
 * inputs: dns_query *query - contains the query
 * outputs: domain contains extracted data according to filter
 * */
static int parse_host_domain(struct _dns_query *query, char *domain);
/* get_ip_addr function: 
 * extracts ip address from packet
 * inputs: xdp_md for data and data_end (packet bounds)
 * output: returns src ip_addrs
 * */
static uint32_t get_ip_addr(struct xdp_md *ctx);
/* checksum function: 
 * returns ip(v4)_header checksum
 * */
static unsigned short checksum(unsigned short *ip, int iphdr_size);

static uint32_t forwarding_server = 0;
static uint32_t fwd_port = INITIAL_FWD_PORT;
static struct forwarding_server fwd_servers[NUM_OF_SERVERS] = 
{
	{.ip_addr = 134744072,
	 .mac = {0x00, 0x50, 0x56, 0xfb, 0x58, 0x8a}
	},
	{.ip_addr = 134743044,
	 .mac = {0x00, 0x50, 0x56, 0xfb, 0x58, 0x8a}
	},
}; 

/* Implementations */

/*xdp-receive-packet - our main XDP program
 * extracts query from the network packet
 * input: xdp_md to get packet content
 * output: bpf_tail_call next XDP program in case of success, XDP_DROP otherwise.
*/
SEC("xdp-packet-preprocess")
int  dns_extract_query(struct xdp_md *ctx)
{
	//packet is between data & data_end starting from ethernet
	void *data = (void *)(long)ctx->data; 
	void *data_end = (void *)(long)ctx->data_end;
	struct ethhdr *eth_header = NULL;
	struct iphdr *ip_header = NULL;
	struct udphdr *udp_header = NULL;
	uint32_t saddr = 0;
	uint32_t eth_header_size = 0;
	uint32_t ip_header_size = 0;
	uint32_t udp_header_size = 0;
	uint32_t dns_header_size = 0;

	/* 1) Validate ETH header size*/
	eth_header = data;
	eth_header_size = sizeof(*eth_header);
	if((void *)eth_header + eth_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 2) Validate IP header size*/
	ip_header = (void *)eth_header + eth_header_size;
	ip_header_size = sizeof(*ip_header);
	if((void *)ip_header + ip_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 3) Validate UDP protocol (DNS request)*/
	if(ip_header->protocol != IPPROTO_UDP)
	{
		return XDP_PASS;
	}

	/* 4) Validate UDP header size*/
	udp_header = (void *)ip_header + ip_header_size;
	udp_header_size = sizeof(*udp_header);
	if((void *)udp_header + udp_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 5) Validate DNS port */
	if(udp_header->dest != __bpf_htons(DNS_PORT))
	{
		return XDP_PASS;//XDP_DROP causing trouble
	}

	/* 6) Validate DNS header size */ 
	struct _dns_hdr *dns_header = (void *)udp_header + udp_header_size;
	dns_header_size = sizeof(*dns_header);
	if((void *)dns_header + dns_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 7) Analyze DNS query */
	if(dns_header->qr == DNS_QUERY_REQUEST)
	{
		saddr = ip_header->saddr;
		void *query_start = (void *)dns_header + dns_header_size; 
		struct _dns_query q;
		int q_length;

		/* 7.1) Extract domain from DNS query */
		q_length = parse_query(ctx, query_start, &q);

		/* 7.2) Validate domain */
		if(q_length != -1)
		{
			bpf_map_update_elem(&query, &saddr, &q, BPF_ANY);
			bpf_printk2("domain: %s", q.qname);
			//call tail function
			bpf_tail_call(ctx, &map_xdp_progs, XDP_PROG_VALIDATION);
		}
		return XDP_DROP;
	}
	return XDP_PASS;
}



/* dns_legal_domain
 * checks if dns query is legal
 * returns XDP_DROP and blocks host for 5 secs in case dns isn't legal
 * otherwise jumps to load-balancer XDP program
 * */
SEC("xdp-packet-validation")
int  dns_legal_domain(struct xdp_md *ctx)
{
	uint32_t saddr = -1;
        uint32_t to_drop = 0;
        uint32_t counters_key = PACKETS_COUNTERS_MAP_KEY;

	/* 1) Get requester IP address */
	if((saddr = get_ip_addr(ctx)) == -1)
	{
		return XDP_DROP;
	}

	/* 2) Get Packets Counters */
	struct _packets_counters *pkts_counters = bpf_map_lookup_elem(&map_packets_counters,&counters_key);
	if(!pkts_counters)
	{
		return XDP_DROP;
	}

	/* 3) Check if host is being blocked */
	uint64_t *exists_time = NULL, curr_time=0;
	if((exists_time = bpf_map_lookup_elem(&map_blocked_requesters, &saddr)))
	{
		curr_time = bpf_ktime_get_ns();
		if(curr_time - *exists_time < BLOCK_TIME)
		{
			pkts_counters->already_blocked = pkts_counters->already_blocked+1;
			bpf_map_update_elem(&map_packets_counters, &counters_key, pkts_counters, BPF_ANY);
			return XDP_DROP;	
		}
		else
		{
			bpf_map_delete_elem(&map_blocked_requesters, &saddr);
		}
	}

	/* 4) Get Dns Query */
	struct _dns_query *q = NULL;
       	if(!(q = bpf_map_lookup_elem(&query, &saddr)))
	{
		return XDP_DROP;
	}

	/* 5) Parse Dns Query */
	char curr_domain[MAX_QUERY_LENGTH];
	__builtin_memset(curr_domain, 0, sizeof(curr_domain));
	//get domain from query
	if(parse_host_domain(q, curr_domain))
	{
		return XDP_DROP;
	}
	/* 4) Validate requested domain */
	/* 4.2) Validate domain*/
	uint32_t *is_illegal = NULL;
	if((is_illegal=bpf_map_lookup_elem(&map_illegal_domains, curr_domain)))
	{
		pkts_counters->dropped_packets_name = pkts_counters->dropped_packets_name + 1;
		to_drop = 1;
	/* 4.2) Validate domain length*/
	}else if(q->qlength>=MAX_ALLOWED_QUERY_LENGTH)
	{
		pkts_counters->dropped_packets_length = pkts_counters->dropped_packets_length + 1;
		to_drop = 1;
	}
	if(to_drop == 1)
	{
		bpf_map_update_elem(&map_packets_counters, &counters_key, pkts_counters, BPF_ANY);
		uint64_t first_query_time = bpf_ktime_get_ns();
		bpf_map_update_elem(&map_blocked_requesters, &saddr, &first_query_time, BPF_ANY);
		return XDP_DROP;
	}
	/* 5) Packet Validated */
	pkts_counters->passed_packets = pkts_counters->passed_packets + 1;
	bpf_map_update_elem(&map_packets_counters, &counters_key, pkts_counters, BPF_ANY);

}
/* load balancer
 * Passes queries to DNS servers (round robin) using XDP_TX 
 * */
SEC("xdp-load-balancer")
int load_balancer(struct xdp_md *ctx){
	void *data = (void *)(long)ctx->data; 
	void *data_end = (void *)(long)ctx->data_end;
	struct ethhdr *eth_header = NULL;
	struct iphdr *ip_header = NULL;
	struct udphdr *udp_header = NULL;
	uint32_t eth_header_size = 0;
	uint32_t ip_header_size = 0;
	uint32_t udp_header_size = 0;
	uint32_t dst_addr = 0;
	
	/* 1) Validate ETH header size*/
	eth_header = data;
	eth_header_size = sizeof(*eth_header);
	if((void *)eth_header + eth_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 2) Validate IP header size*/
	ip_header = (void *)eth_header + eth_header_size;
	ip_header_size = sizeof(*ip_header);
	if((void *)ip_header + ip_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 3) Validate UDP protocol (DNS request)*/
	if(ip_header->protocol != IPPROTO_UDP)
	{
		return XDP_PASS;
	}

	/* 4) Validate UDP header size*/
	udp_header = (void *)ip_header + ip_header_size;
	udp_header_size = sizeof(*udp_header);
	if((void *)udp_header + udp_header_size > data_end)
	{
		return XDP_DROP;
	}

	/* 5) Validate DNS port */
	if(udp_header->dest != __bpf_htons(DNS_PORT))
	{
		return XDP_PASS;//XDP_DROP causing trouble
	}

	/* Sanity check for forwarding_server static variable */
	if(forwarding_server >= NUM_OF_SERVERS) return XDP_DROP;
	
	/* Assign packets to ns servers in a round robin fashion */
	if(forwarding_server == NUM_OF_SERVERS - 1){
		char *mac = fwd_servers[forwarding_server].mac;
		uint32_t new_dst_addr = fwd_servers[forwarding_server].ip_addr;

		//Change MAC
		eth_header->h_dest[0] = mac[0];
		eth_header->h_dest[1] = mac[1];
		eth_header->h_dest[2] = mac[2] ;
		eth_header->h_dest[3] = mac[3] ;
		eth_header->h_dest[4] = mac[4];
		eth_header->h_dest[5] = mac[5];
		
		//Changes in ip header
		dst_addr = bpf_ntohs(*(unsigned short *)&ip_header->daddr);
		ip_header->daddr = bpf_htonl(new_dst_addr);
		ip_header->check = checksum((unsigned short *)ip_header, sizeof(struct iphdr));
		
		//Changes in udp header; checksum = 0 => ignoring checking
		udp_header->source = bpf_ntohs(fwd_port);
		udp_header->check = 0;
		forwarding_server = 0;
		fwd_port = INITIAL_FWD_PORT;
	}else{
		char *mac = fwd_servers[forwarding_server].mac;
		uint32_t new_dst_addr = fwd_servers[forwarding_server].ip_addr;

		//Change MAC
		eth_header->h_dest[0] = mac[0];
		eth_header->h_dest[1] = mac[1];
		eth_header->h_dest[2] = mac[2] ;
		eth_header->h_dest[3] = mac[3] ;
		eth_header->h_dest[4] = mac[4];
		eth_header->h_dest[5] = mac[5];

		//Changes in ip header		
		ip_header->daddr = bpf_htonl(new_dst_addr);
		//ip_header->daddr = bpf_htonl(fwd_servers[forwarding_server].ip_addr);
		ip_header->check = checksum((unsigned short *)ip_header, sizeof(struct iphdr));
		
		//Changes in udp header; checksum = 0 => ignoring checking
		udp_header->source = bpf_ntohs(fwd_port);
		udp_header->check = 0;
		++forwarding_server;
		++fwd_port;
	}
	//bpf_printk("leaving load_balancer\n");
	return XDP_TX;
}

/**************************************** checksum ******************************************************/
static unsigned short checksum(unsigned short *ip, int iphdr_size){
	unsigned short s = 0;
	while(iphdr_size > 1){
		s += *ip;
		ip++;
		iphdr_size -= 2;
	}
	if(iphdr_size == 1)
		s += *(unsigned char *)ip;
	s = (s & 0xffff) + (s >> 16);
	s = (s & 0xffff) + (s >> 16);
	return ~s;
}

/**************************************** parse_query ******************************************************/
static int parse_query(struct xdp_md *ctx, void *query_start, struct _dns_query *q){
	void *data_end = (void *)(long)ctx->data_end;
	void *cursor = query_start;
	uint16_t pos = 0, i=0;
	__builtin_memset(&q->qname[0], 0, sizeof(q->qname));
	q->qclass = 0; q->qtype = 0;
	for(i=0; i<MAX_QUERY_LENGTH; ++i){
		if(cursor + 1 > data_end)
			break;
		if(*(char *)(cursor) == 0){
			if(cursor + 5 > data_end)//can't get record and type
				break;
			else{
				q->qtype = bpf_htons(*(uint16_t *)(cursor+1));
				q->qclass = bpf_htons(*(uint16_t *)(cursor+3));
				q->qlength = pos;
			}
			q->qname[pos] = *(char *)(cursor);
			return pos + 1 + 2 + 2;
		}
		q->qname[pos] = *(char *)(cursor);
		++pos;
		++cursor;
	}
	return -1;
}

/**************************************** parse_host_domain ******************************************************/
static int parse_host_domain(struct _dns_query *q, char *curr_domain){
	uint32_t k=0;
	for(int i=0; i<MAX_QUERY_LENGTH; ++i){
		if(q->qname[i] == '\x03'){// \0x3 is ETX - end of text
			__builtin_memset(curr_domain, 0, MAX_QUERY_LENGTH);
			k=0;
			continue;
		}	
		else if(q->qname[i] == '\0'){
			break;
		}
		curr_domain[k++] = q->qname[i];
	}
	return 0;
}

/**************************************** get_ip_addr ******************************************************/
static uint32_t get_ip_addr(struct xdp_md *ctx){
	void *data = (void *)(long)ctx->data; 
	void *data_end = (void *)(long)ctx->data_end;
	struct ethhdr *eth_header = data;
	struct iphdr *ip_header;
	
	if((void *)eth_header + sizeof(*eth_header) > data_end){
		return -1;
	}
	ip_header = data + sizeof(*eth_header);
	if((void *)ip_header + sizeof(*ip_header) > data_end){
		return XDP_DROP;
	}
	if(ip_header->protocol != IPPROTO_UDP)
		return -1;
	return ip_header->saddr;

}

/* SPDX-License-Identifier: GPL-2.0 */
char LICENSE[] SEC("license") = "GPL";

