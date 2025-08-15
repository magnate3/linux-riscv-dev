#include "tcp_util.h"
#include "dpdk.h"
#include "tcp.h"
#define IP_BUF_LEN 16
// Shuffle the TCP source port array
#define roundup(x, y) (					\
{							\
	const typeof(y) __y = y;			\
	(((x) + (__y - 1)) / __y) * __y;		\
}							\
)
void shuffle(uint16_t* arr, uint32_t n) {
	if(n < 2) {
		return;
	}

        uint32_t i;
	for(i = 0; i < n - 1; i++) {
		uint32_t j = i + rte_rand() / (UINT64_MAX / (n - i) + 1);
		uint16_t tmp = arr[j];
		arr[j] = arr[i];
		arr[i] = tmp;
	}
}
static void ip_format_addr(char *buf, uint16_t size,const uint32_t ip_addr)
{
    snprintf(buf, size, "%" PRIu8 ".%" PRIu8 ".%" PRIu8 ".%" PRIu8 ,
             (uint8_t)((ip_addr >> 24) & 0xff),
             (uint8_t)((ip_addr >> 16) & 0xff),
             (uint8_t)((ip_addr >> 8) & 0xff),
             (uint8_t)((ip_addr)&0xff));
}
void dpdk_dump_iph(const struct rte_ipv4_hdr *ip_hdr)
{
    char buf[IP_BUF_LEN] = {0};
    ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->src_addr)); 
    printf("src ip : %s, ",buf);
    memset(buf,IP_BUF_LEN,0);
    ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(ip_hdr->dst_addr)); 
    printf("dst ip : %s ",buf);
    printf("\tip total len %u \n", rte_be_to_cpu_16(ip_hdr->total_length));
}

void dpdk_dump_tcph(struct rte_tcp_hdr *tcp_hdr, unsigned int l4_len)
{
     unsigned int          tcp_hdr_len;
     tcp_hdr_len = (tcp_hdr->data_off >> 4) << 2;
     printf("sport=%u, dport=%u, hdrlen=%u, flags=%c%c%c%c%c%c, data_len=%u",
              rte_be_to_cpu_16(tcp_hdr->src_port),
              rte_be_to_cpu_16(tcp_hdr->dst_port),
              tcp_hdr_len,
              (tcp_hdr->tcp_flags & RTE_TCP_URG_FLAG) == 0 ? '-' : 'u',
              (tcp_hdr->tcp_flags & RTE_TCP_ACK_FLAG) == 0 ? '-' : 'a',
              (tcp_hdr->tcp_flags & RTE_TCP_PSH_FLAG) == 0 ? '-' : 'p',
              (tcp_hdr->tcp_flags & RTE_TCP_RST_FLAG) == 0 ? '-' : 'r',
              (tcp_hdr->tcp_flags & RTE_TCP_SYN_FLAG) == 0 ? '-' : 's',
              (tcp_hdr->tcp_flags & RTE_TCP_FIN_FLAG) == 0 ? '-' : 'f',
              l4_len - tcp_hdr_len);

    printf("  seq=%u, ack=%u, window=%u, urgent=%u \n",
              rte_be_to_cpu_32(tcp_hdr->sent_seq),
              rte_be_to_cpu_32(tcp_hdr->recv_ack),
              rte_be_to_cpu_16(tcp_hdr->rx_win),
              rte_be_to_cpu_16(tcp_hdr->tcp_urp));

}
static inline uint8_t
add_syn_opts(struct rte_tcp_hdr *tcp_hdr)
{
	uint8_t *to;
	struct tcpopt *opt;

	to = (uint8_t *)(tcp_hdr +1);

	/* setup MSS*/
	opt = (struct tcpopt *)to;
	opt->kl.raw = TCP_OPT_KL_MSS;
	opt->mss = rte_cpu_to_be_16(TCP4_OP_MSS);

	to += TCP_OPT_LEN_MSS;
	opt = (struct tcpopt *)to;

	/* setup TMS*/
	 {

		opt->kl.raw = TCP_OPT_KL_TMS;
		opt->ts.val = rte_cpu_to_be_32(rte_rdtsc());
		opt->ts.ecr = rte_cpu_to_be_32(rte_rdtsc());

		to += TCP_OPT_LEN_TMS;
		opt = (struct tcpopt *)to;
	}

	/* setup TMS*/
	{

		opt->kl.raw = TCP_OPT_KL_WSC;
		opt->wscale = 4;

		to += TCP_OPT_LEN_WSC;
		opt = (struct tcpopt *)to;
	}

	//to[0] = TCP_OPT_KIND_EOL;
        return TCP_OPT_LEN_MSS + TCP_OPT_LEN_TMS + TCP_OPT_LEN_WSC;
}
uint8_t add_mss_option(struct rte_tcp_hdr *tcp_hdr, uint16_t mss_value)
{
  	uint8_t *to;
	struct tcpopt *opt;

	to = (uint8_t *)(tcp_hdr+1);

	/* setup MSS*/
	opt = (struct tcpopt *)to;
	opt->kl.raw = TCP_OPT_KL_MSS;
	opt->mss = rte_cpu_to_be_16(mss_value);
	to += TCP_OPT_LEN_MSS;
        return TCP_OPT_LEN_MSS;
}

uint8_t pad_option(struct rte_tcp_hdr *tcp_hdr, uint8_t option_len)
{
    uint8_t align = sizeof(uint32_t);
    uint8_t *to;
    uint8_t *tcp_options = (uint8_t*)(tcp_hdr+1);
    uint8_t len;
    to = (uint8_t *)(tcp_options+option_len);
    to[0] = TCP_OPT_KIND_EOL;
    option_len += 1;
    len = roundup(option_len,align); 
    printf("option_len %u, align option len %u\n",option_len, len);
    return len;
}
void reset_tcp_blocks(tcp_control_block_t * block) {
     rte_atomic16_set(&block->tcb_state, TCP_INIT);
     rte_atomic16_set(&block->tcb_rwin, 0xFFFF);
     uint32_t seq = rte_rand();
     block->tcb_seq_ini = seq;
     block->tcb_next_seq = seq;
}
// Create and initialize the TCP Control Blocks for all flows
void init_tcp_blocks() {
	// allocate the all control block structure previosly
	tcp_control_blocks = (tcp_control_block_t *) rte_zmalloc("tcp_control_blocks", nr_flows * sizeof(tcp_control_block_t), RTE_CACHE_LINE_SIZE);

	// choose TCP source port for all flows
	uint16_t src_tcp_port;
	uint16_t ports[nr_flows];
        
        uint32_t i;
	for(i = 0; i < nr_flows; i++) {
		ports[i] = rte_cpu_to_be_16((i % (nr_flows/nr_servers)) + 1);
	}
	// shuffle port array
	shuffle(ports, nr_flows);

	for(i = 0; i < nr_flows; i++) {
		rte_atomic16_init(&tcp_control_blocks[i].tcb_state);
		rte_atomic16_set(&tcp_control_blocks[i].tcb_state, TCP_INIT);
		rte_atomic16_set(&tcp_control_blocks[i].tcb_rwin, 0xFFFF);

		src_tcp_port = ports[i];

		tcp_control_blocks[i].src_addr = src_ipv4_addr;
		tcp_control_blocks[i].dst_addr = dst_ipv4_addr;

		tcp_control_blocks[i].src_port = src_tcp_port;
		tcp_control_blocks[i].dst_port = rte_cpu_to_be_16(dst_tcp_port + (i % nr_servers));
		
		printf("dst_port: %u\n", dst_tcp_port + (i % nr_servers));
		uint32_t seq = rte_rand();
		tcp_control_blocks[i].tcb_seq_ini = seq;
		tcp_control_blocks[i].tcb_next_seq = seq;

		tcp_control_blocks[i].flow_mark_action.id = i;
		tcp_control_blocks[i].flow_queue_action.index = i % nr_queues;
		tcp_control_blocks[i].flow_eth.type = ETH_IPV4_TYPE_NETWORK;
		tcp_control_blocks[i].flow_eth_mask.type = 0xFFFF;
		tcp_control_blocks[i].flow_ipv4.hdr.src_addr = tcp_control_blocks[i].dst_addr;
		tcp_control_blocks[i].flow_ipv4.hdr.dst_addr = tcp_control_blocks[i].src_addr;
		tcp_control_blocks[i].flow_ipv4_mask.hdr.src_addr = 0xFFFFFFFF;
		tcp_control_blocks[i].flow_ipv4_mask.hdr.dst_addr = 0xFFFFFFFF;
		tcp_control_blocks[i].flow_tcp.hdr.src_port = tcp_control_blocks[i].dst_port;
		tcp_control_blocks[i].flow_tcp.hdr.dst_port = tcp_control_blocks[i].src_port;
		tcp_control_blocks[i].flow_tcp_mask.hdr.src_port = 0xFFFF;
		tcp_control_blocks[i].flow_tcp_mask.hdr.dst_port = 0xFFFF;
	}
}
tcp_control_block_t * get_tcp_control_block(const struct rte_ipv4_hdr * iph, const struct rte_tcp_hdr * tcp_hdr,uint32_t *idx)
{
        tcp_control_block_t * item;
        uint32_t i;
        //dpdk_dump_iph(iph);
        //char buf[IP_BUF_LEN] = {0};
	for(i = 0; i < nr_flows; i++) {
            item =  &tcp_control_blocks[i];
#if 0
            ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(item->src_addr)); 
            printf("item src ip : %s, ",buf);
            memset(buf,IP_BUF_LEN,0);
            ip_format_addr(buf,IP_BUF_LEN,rte_be_to_cpu_32(item->dst_addr)); 
            printf("item dst ip : %s ",buf);
            printf("tcp hdr src port : %u, dst port : %u ,",rte_be_to_cpu_16(tcp_hdr->src_port), rte_be_to_cpu_16(tcp_hdr->dst_port));
            printf("item src port : %u, dst port : %u \n",item->src_port, item->dst_port);
#endif
#if 0
            if(item->src_addr == rte_be_to_cpu_32(iph->dst_addr) && item->dst_addr == rte_be_to_cpu_32(iph->src_addr)\
                && item->src_port == rte_be_to_cpu_16(tcp_hdr->dst_port) && item->dst_port == rte_be_to_cpu_16(tcp_hdr->src_port)){
#else
            if(item->src_addr == iph->dst_addr && item->dst_addr == iph->src_addr\
                && item->src_port == tcp_hdr->dst_port && item->dst_port == tcp_hdr->src_port){
#endif
                *idx = i;
                return item;
            }      
        }
        return NULL;
}
#if 1
// Create the TCP SYN packet
struct rte_mbuf* create_syn_packet(uint16_t i) {
	uint8_t option_len=0,tcp_len=0;
	// allocate TCP SYN packet in the hugepages
	struct rte_mbuf* pkt = rte_pktmbuf_alloc(pktmbuf_pool);
	if(pkt == NULL) {
		rte_exit(EXIT_FAILURE, "Error to alloc a rte_mbuf.\n");
	}

	// ensure that IP/TCP checksum offloadings
	pkt->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM);

	// get control block for the flow
	tcp_control_block_t *block = &tcp_control_blocks[i];

	// fill Ethernet information
	struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *) rte_pktmbuf_mtod(pkt, struct ether_hdr*);
	eth_hdr->d_addr = dst_eth_addr;
	eth_hdr->s_addr = src_eth_addr;
	eth_hdr->ether_type = ETH_IPV4_TYPE_NETWORK;

	// fill TCP information
	struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
#if 0
	tcp_hdr->dst_port = htons(block->dst_port);
	tcp_hdr->src_port = htons(block->src_port);
#else
	tcp_hdr->dst_port = block->dst_port;
	tcp_hdr->src_port = block->src_port;
#endif
        tcp_len = sizeof(struct rte_tcp_hdr);
#if 0
        option_len += add_mss_option(tcp_hdr,TCP4_OP_MSS);
#else
        option_len = add_syn_opts(tcp_hdr);
        option_len = pad_option(tcp_hdr,option_len);
#endif
        //tcp_hdr->data_off = 0x50;
        //tcp_hdr->data_off = 0x60;
        //tcp_hdr->data_off = ((sizeof(struct rte_tcp_hdr) / sizeof(uint32_t)) << 4);
        tcp_hdr->data_off = ((tcp_len + option_len) / sizeof(uint32_t)) << 4;
	tcp_hdr->sent_seq = block->tcb_seq_ini;
	tcp_hdr->recv_ack = 0;
	tcp_hdr->rx_win = 0xFFFF;
	tcp_hdr->tcp_flags = RTE_TCP_SYN_FLAG;
	tcp_hdr->tcp_urp = 0;
	tcp_hdr->cksum = 0;
        //dpdk_dump_tcph(tcp_hdr,tcp_len + option_len);
	// fill IPv4 information
	struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	ipv4_hdr->version_ihl = 0x45;
	ipv4_hdr->total_length = rte_cpu_to_be_16(sizeof(struct rte_ipv4_hdr) + tcp_len + option_len);
	//ipv4_hdr->total_length = rte_cpu_to_be_16(sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr));
	ipv4_hdr->time_to_live = 255;
	ipv4_hdr->packet_id = 0;
	ipv4_hdr->next_proto_id = IPPROTO_TCP;
	//ipv4_hdr->fragment_offset = 0;
	ipv4_hdr->fragment_offset =   htons(RTE_IPV4_HDR_DF_FLAG);
#if 0
	ipv4_hdr->src_addr = htonl(block->src_addr);
	ipv4_hdr->dst_addr = htonl(block->dst_addr);
#else
	ipv4_hdr->src_addr = block->src_addr;
	ipv4_hdr->dst_addr = block->dst_addr;
#endif
	ipv4_hdr->hdr_checksum = 0;

        dpdk_dump_iph(ipv4_hdr);
        pkt->l4_len = tcp_len + option_len;
        //pkt->l4_len = sizeof(struct rte_tcp_hdr);
        pkt->l3_len = sizeof(struct rte_ipv4_hdr);
        pkt->l2_len = RTE_ETHER_HDR_LEN;
	//pkt->ol_flags |= RTE_MBUF_F_TX_TCP_CKSUM|RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_IPV4;
	// fill the packet size
	pkt->data_len = sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr) + option_len;
	pkt->pkt_len = pkt->data_len;

	return pkt;
}
#else
// Create the TCP SYN packet
struct rte_mbuf* create_syn_packet(uint16_t i) {
	// allocate TCP SYN packet in the hugepages
	struct rte_mbuf* pkt = rte_pktmbuf_alloc(pktmbuf_pool);
	if(pkt == NULL) {
		rte_exit(EXIT_FAILURE, "Error to alloc a rte_mbuf.\n");
	}

	// ensure that IP/TCP checksum offloadings
	pkt->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM);

	// get control block for the flow
	tcp_control_block_t *block = &tcp_control_blocks[i];

	// fill Ethernet information
	struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *) rte_pktmbuf_mtod(pkt, struct ether_hdr*);
	eth_hdr->d_addr = dst_eth_addr;
	eth_hdr->s_addr = src_eth_addr;
	eth_hdr->ether_type = ETH_IPV4_TYPE_NETWORK;

	// fill IPv4 information
	struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	ipv4_hdr->version_ihl = 0x45;
	ipv4_hdr->total_length = rte_cpu_to_be_16(sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr));
	ipv4_hdr->time_to_live = 255;
	ipv4_hdr->packet_id = 0;
	ipv4_hdr->next_proto_id = IPPROTO_TCP;
	//ipv4_hdr->fragment_offset = 0;
	ipv4_hdr->fragment_offset =   htons(RTE_IPV4_HDR_DF_FLAG);
	ipv4_hdr->src_addr = block->src_addr;
	ipv4_hdr->dst_addr = block->dst_addr;
	ipv4_hdr->hdr_checksum = 0;
        dpdk_dump_iph(ipv4_hdr);
	// fill TCP information
	struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
	tcp_hdr->dst_port = block->dst_port;
	tcp_hdr->src_port = block->src_port;
        //tcp_hdr->data_off = 0x50;
        tcp_hdr->data_off = ((sizeof(struct rte_tcp_hdr) / sizeof(uint32_t)) << 4);
	tcp_hdr->sent_seq = block->tcb_seq_ini;
	tcp_hdr->recv_ack = 0;
	tcp_hdr->rx_win = 0xFFFF;
	tcp_hdr->tcp_flags = RTE_TCP_SYN_FLAG;
	tcp_hdr->tcp_urp = 0;
	tcp_hdr->cksum = 0;


        pkt->l4_len = sizeof(struct rte_tcp_hdr);
        pkt->l3_len = sizeof(struct rte_ipv4_hdr);
        pkt->l2_len = RTE_ETHER_HDR_LEN;
	//pkt->ol_flags |= RTE_MBUF_F_TX_TCP_CKSUM|RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_IPV4;
	// fill the packet size
	pkt->data_len = sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr);
	pkt->pkt_len = pkt->data_len;

	return pkt;
}
#endif
// Create the TCP ACK packet
struct rte_mbuf *create_ack_packet(uint16_t i) {
	// allocate TCP ACK packet in the hugepages
	// allocate TCP ACK packet in the hugepages
	struct rte_mbuf* pkt = rte_pktmbuf_alloc(pktmbuf_pool);
	if(pkt == NULL) {
		rte_exit(EXIT_FAILURE, "Error to alloc a rte_mbuf.\n");
	}

	// ensure that IP/TCP checksum offloadings
	pkt->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM);

	// get control block for the flow
	tcp_control_block_t *block = &tcp_control_blocks[i];

	// fill Ethernet information
	struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *) rte_pktmbuf_mtod(pkt, struct ether_hdr*);
	eth_hdr->d_addr = dst_eth_addr;
	eth_hdr->s_addr = src_eth_addr;
	eth_hdr->ether_type = ETH_IPV4_TYPE_NETWORK;

	// fill IPv4 information
	struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	ipv4_hdr->version_ihl = 0x45;
	ipv4_hdr->total_length = rte_cpu_to_be_16(sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr));
	ipv4_hdr->time_to_live = 255;
	ipv4_hdr->packet_id = 0;
	ipv4_hdr->next_proto_id = IPPROTO_TCP;
	ipv4_hdr->fragment_offset = 0;
	ipv4_hdr->src_addr = block->src_addr;
	ipv4_hdr->dst_addr = block->dst_addr;
	ipv4_hdr->hdr_checksum = 0;

	// set the TCP SEQ number
	uint32_t newseq = rte_cpu_to_be_32(rte_be_to_cpu_32(block->tcb_next_seq) + 1);
	block->tcb_next_seq = newseq;

	// fill TCP information
	struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
	tcp_hdr->dst_port = block->dst_port;
	tcp_hdr->src_port = block->src_port;
	tcp_hdr->data_off = 0x50;
	tcp_hdr->sent_seq = newseq;
	tcp_hdr->recv_ack = rte_atomic32_read(&block->tcb_next_ack);
	tcp_hdr->rx_win = 0xFFFF;
	tcp_hdr->tcp_flags = RTE_TCP_ACK_FLAG;
	tcp_hdr->tcp_urp = 0;
	tcp_hdr->cksum = 0;

        pkt->l4_len = sizeof(struct rte_tcp_hdr);
        pkt->l3_len = sizeof(struct rte_ipv4_hdr);
        pkt->l2_len = RTE_ETHER_HDR_LEN;
	// fill the packet size
	pkt->data_len = sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_tcp_hdr);
	pkt->pkt_len = pkt->data_len;

	return pkt;
}

// Process the TCP SYN+ACK packet and return the TCP ACK
struct rte_mbuf* process_syn_ack_packet(struct rte_mbuf* pkt) {
	// process only IPv4 packets
	struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *) rte_pktmbuf_mtod(pkt, struct ether_hdr*);
	if(eth_hdr->ether_type != ETH_IPV4_TYPE_NETWORK) {
		return NULL;
	}

	// process only TCP packets
	struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	if(ipv4_hdr->next_proto_id != IPPROTO_TCP) {
		return NULL;
	}

	// get TCP header
	struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *, sizeof(struct rte_ether_hdr) + (ipv4_hdr->version_ihl & 0x0f)*4);

#if     NIC_SUPPPORT_FLOW_OFFLOAD
	// retrieve the index of the flow from the NIC (NIC tags the packet according the 5-tuple using DPDK rte_flow)
	uint32_t idx = pkt->hash.fdir.hi;

	// get control block for the flow
	tcp_control_block_t *block = &tcp_control_blocks[idx];
#else
	uint32_t idx = 0;
	tcp_control_block_t *block = get_tcp_control_block(ipv4_hdr, tcp_hdr,&idx);
        if(NULL == block) {
             printf("find tcp control block fail \n");
             return NULL;
        }
#endif
	// get the TCP control block state
	uint8_t state = rte_atomic16_read(&block->tcb_state);

	// process only in SYN_SENT state and SYN+ACK packet
	if((state == TCP_SYN_SENT) && (tcp_hdr->tcp_flags == (RTE_TCP_SYN_FLAG|RTE_TCP_ACK_FLAG))) {
		// update the TCP state to ESTABLISHED
		rte_atomic16_set(&block->tcb_state, TCP_ESTABLISHED);

		// get the TCP SEQ number
		uint32_t seq = rte_be_to_cpu_32(tcp_hdr->sent_seq);
		block->last_seq_recv = seq;

		// update TCP SEQ and ACK numbers
		rte_atomic32_set(&block->tcb_next_ack, rte_cpu_to_be_32(seq + 1));
		block->tcb_ack_ini = tcp_hdr->sent_seq;

		// return TCP ACK packet
		return create_ack_packet(idx);
	}

	return NULL;
}

// Fill the TCP packets from TCP Control Block data
void fill_tcp_packet(uint16_t i, struct rte_mbuf *pkt, uint8_t flag) {
	uint16_t tcp_total = 0;
	uint16_t tcp_hdr_len = 0;
	// get control block for the flow
	tcp_control_block_t *block = &tcp_control_blocks[i];

	// ensure that IP/TCP checksum offloadings
	pkt->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM | RTE_MBUF_F_TX_TCP_CKSUM);

	// fill Ethernet information
	struct rte_ether_hdr *eth_hdr = (struct rte_ether_hdr *) rte_pktmbuf_mtod(pkt, struct ether_hdr*);
	eth_hdr->d_addr = dst_eth_addr;
	eth_hdr->s_addr = src_eth_addr;
	eth_hdr->ether_type = ETH_IPV4_TYPE_NETWORK;


	// set the TCP SEQ number
	uint32_t sent_seq = block->tcb_next_seq;

	// fill TCP information
	struct rte_tcp_hdr *tcp_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_tcp_hdr *, sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
	tcp_hdr->dst_port = block->dst_port;
	tcp_hdr->src_port = block->src_port;
	tcp_hdr->data_off = 0x50;
	tcp_hdr->sent_seq = sent_seq;
	tcp_hdr->recv_ack = rte_atomic32_read(&block->tcb_next_ack);
	tcp_hdr->rx_win = 0xFFFF;
	tcp_hdr->tcp_flags = flag;
	tcp_hdr->tcp_urp = 0;
	tcp_hdr->cksum = 0;

	tcp_hdr_len = (tcp_hdr->data_off >> 4) << 2;
	// updates the TCP SEQ number
	sent_seq = rte_cpu_to_be_32(rte_be_to_cpu_32(sent_seq) + tcp_payload_size);
	block->tcb_next_seq = sent_seq;

	// fill the payload of the packet
	uint8_t *payload = ((uint8_t*)tcp_hdr) + tcp_hdr_len;
	fill_tcp_payload(payload, tcp_payload_size);

	tcp_total = tcp_payload_size + tcp_hdr_len;
	// fill IPv4 information
	struct rte_ipv4_hdr *ipv4_hdr = rte_pktmbuf_mtod_offset(pkt, struct rte_ipv4_hdr *, sizeof(struct rte_ether_hdr));
	ipv4_hdr->version_ihl = 0x45;
	ipv4_hdr->total_length = rte_cpu_to_be_16(tcp_total + sizeof(struct rte_ipv4_hdr));
	ipv4_hdr->time_to_live = 255;
	ipv4_hdr->packet_id = rte_cpu_to_be_16(rte_rand());
	ipv4_hdr->next_proto_id = IPPROTO_TCP;
	ipv4_hdr->fragment_offset = 0;
	ipv4_hdr->src_addr = block->src_addr;
	ipv4_hdr->dst_addr = block->dst_addr;
	ipv4_hdr->hdr_checksum = 0;
#if 1
//#if DBUG_TCP
        dpdk_dump_tcph(tcp_hdr,tcp_total);
#endif
        pkt->l4_len = tcp_hdr_len;
        //pkt->l4_len = tcp_total;
        pkt->l3_len = sizeof(struct rte_ipv4_hdr);
        pkt->l2_len = RTE_ETHER_HDR_LEN;
#if DBUG_TCP_GSO
        if(tcp_payload_size >  RTE_ETHER_MTU)
        {
            pkt->ol_flags |= PKT_TX_TCP_SEG;
            pkt->tso_segsz = 1024;
        }
#endif
#if DBUG_TCP
        dpdk_dump_iph(ipv4_hdr);
#endif
	// fill the packet size
	pkt->data_len = pkt->l2_len + pkt->l3_len + tcp_total;
	pkt->pkt_len = pkt->data_len;
}

// Fill the payload of the TCP packet
void fill_tcp_payload(uint8_t *payload, uint32_t length) {
        uint32_t i = 0;
	for(i = 0; i < length; i++) {
		payload[i] = 'A';
	}
}
