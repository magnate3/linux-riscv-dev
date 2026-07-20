
#include <rte_eal.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <stdio.h>
#include <arpa/inet.h>


#define NUM_MBUFS (4096-1)
#define CACHE_SIZE 0
#define PRIV_SIZE 0
#define BURST_SIZE	32


#define gPortId 0

#define TEST_CHECKSUM 1
#if TEST_CHECKSUM
#include "dpdk.h"
#endif
static struct rte_ether_addr tx_src_mac;
static struct rte_ether_addr tx_dst_mac;
static uint32_t tx_src_ipaddr;
static uint32_t tx_dst_ipaddr;
static uint16_t tx_src_udp_port;
static uint16_t tx_dst_udp_port;



static const struct rte_eth_conf port_conf_default = {
	.rxmode = {.max_rx_pkt_len = RTE_ETHER_MAX_LEN }
};


static void init_port(struct rte_mempool *mbuf_pool) {
    //判断是否存在可用的eth
    uint16_t nb_sys_ports = rte_eth_dev_count_avail();
    if(nb_sys_ports == 0 ) {
        rte_exit(EXIT_FAILURE, "Error not eth support\n");
    }

	struct rte_eth_dev_info dev_info;
	rte_eth_dev_info_get(gPortId, &dev_info);

    //配置接口属性
    const uint16_t nb_rx_queues = 1;
    const uint16_t nb_tx_queues = 1;

    const struct rte_eth_conf eth_port_conf = port_conf_default;

    if(rte_eth_dev_configure(gPortId, nb_rx_queues, nb_tx_queues, &eth_port_conf) != 0) {
        rte_exit(EXIT_FAILURE, "Error configure eth dev\n");
    }

    //为网卡分配接收队列
    if(rte_eth_rx_queue_setup(gPortId, 0, 1024, rte_eth_dev_socket_id(gPortId), NULL, mbuf_pool) != 0) {
        rte_exit(EXIT_FAILURE, "Error RX queue setup\n");
    }

    //为网卡分配发送队列
    struct rte_eth_txconf txq_conf = dev_info.default_txconf;
    txq_conf.offloads = eth_port_conf.rxmode.offloads;
    if (rte_eth_tx_queue_setup(gPortId, 0 , 1024, 
        rte_eth_dev_socket_id(gPortId), &txq_conf) < 0) {
		
        rte_exit(EXIT_FAILURE, "Could not setup TX queue\n");
		
    }
	
    //启动网卡
    if(rte_eth_dev_start(gPortId) != 0) {
        rte_exit(EXIT_FAILURE, "Error Start eth dev\n");
    }

    printf("Success start eth dev: %d\n", gPortId);
}

static inline uint32_t 
csum32_add(uint32_t a, uint32_t b) {
	a += b;
	return a + (a < b);
}

static inline uint16_t 
csum16_add(uint16_t a, uint16_t b) {
	a += b;
	return a + (a < b);
}
#if !TEST_CHECKSUM
static int encode_udp_pkt(uint8_t *pkt_data, unsigned int total_len, uint8_t *udp_data, unsigned int len){
    //eth
    struct rte_ether_hdr *ehdr = (struct rte_ether_hdr *)pkt_data;
    rte_memcpy(&ehdr->s_addr, &tx_src_mac, sizeof(struct rte_ether_addr));
    rte_memcpy(&ehdr->d_addr, &tx_dst_mac, sizeof(struct rte_ether_addr));
    ehdr->ether_type = htons(RTE_ETHER_TYPE_IPV4);

    //ip
    struct rte_ipv4_hdr *iphdr = (struct rte_ipv4_hdr *)(pkt_data + sizeof(struct rte_ether_hdr));
    iphdr->version_ihl = 0x45;
    iphdr->type_of_service = 0;
    iphdr->total_length = htons(total_len - sizeof(struct rte_ether_hdr));
    iphdr->packet_id = 0;
    iphdr->fragment_offset = 0;
    iphdr->time_to_live = 64;
    iphdr->next_proto_id = IPPROTO_UDP;
    iphdr->src_addr = tx_src_ipaddr;
    iphdr->dst_addr = tx_dst_ipaddr;
	
    iphdr->hdr_checksum = 0;
    iphdr->hdr_checksum = rte_ipv4_cksum(iphdr);


    //udp
    struct rte_udp_hdr *udphdr = (struct rte_udp_hdr *)(pkt_data + sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
    udphdr->dgram_len = htons(total_len - sizeof(struct rte_ipv4_hdr) - sizeof(struct rte_ether_hdr));
    
    udphdr->dst_port = tx_dst_udp_port;
    udphdr->src_port = tx_src_udp_port;
    rte_memcpy((uint8_t*)(udphdr) + sizeof(struct rte_udp_hdr), udp_data, len);
	
    udphdr->dgram_cksum = 0;
    udphdr->dgram_cksum = rte_ipv4_udptcp_cksum(iphdr, udphdr);

    printf("Sending: udp src port: %d -> %d\n", ntohs(tx_src_udp_port), ntohs(tx_dst_udp_port));

    return 0;
    
}
#else
static int encode_udp_pkt2(struct rte_mbuf *mbuf,  uint8_t *udp_data, unsigned int len){
    //eth
    uint8_t *pkt_data = rte_pktmbuf_mtod(mbuf, uint8_t*);
    struct rte_ether_hdr *ehdr = (struct rte_ether_hdr *)pkt_data;
    unsigned int total_len = sizeof(struct rte_ether_addr) + sizeof(struct rte_ether_hdr) +  sizeof(struct rte_udp_hdr) + len;
    rte_memcpy(&ehdr->s_addr, &tx_src_mac, sizeof(struct rte_ether_addr));
    rte_memcpy(&ehdr->d_addr, &tx_dst_mac, sizeof(struct rte_ether_addr));
    ehdr->ether_type = htons(RTE_ETHER_TYPE_IPV4);
    //ip
    struct rte_ipv4_hdr *iphdr = (struct rte_ipv4_hdr *)(pkt_data + sizeof(struct rte_ether_hdr));
    iphdr->version_ihl = 0x45;
    iphdr->type_of_service = 0;
    iphdr->total_length = htons(total_len - sizeof(struct rte_ether_hdr));
    iphdr->packet_id = 0;
    iphdr->fragment_offset = 0;
    iphdr->time_to_live = 64;
    iphdr->next_proto_id = IPPROTO_UDP;
    iphdr->src_addr = tx_src_ipaddr;
    iphdr->dst_addr = tx_dst_ipaddr;
	
    iphdr->hdr_checksum = 0;
    iphdr->hdr_checksum = rte_ipv4_cksum(iphdr);


    //udp
    struct rte_udp_hdr *udphdr = (struct rte_udp_hdr *)(pkt_data + sizeof(struct rte_ether_hdr) + sizeof(struct rte_ipv4_hdr));
    udphdr->dgram_len = htons(total_len - sizeof(struct rte_ipv4_hdr) - sizeof(struct rte_ether_hdr));
    
    udphdr->dst_port = tx_dst_udp_port;
    udphdr->src_port = tx_src_udp_port;
    rte_memcpy((uint8_t*)(udphdr) + sizeof(struct rte_udp_hdr), udp_data, len);
	
    udphdr->dgram_cksum = 0;
    udphdr->dgram_cksum = rte_ipv4_udptcp_cksum(iphdr, udphdr);
    printf("ip cksum %x, udp cksum %x \n", iphdr->hdr_checksum , udphdr->dgram_cksum);
#if 0
    iphdr->hdr_checksum = 0;
    udphdr->dgram_cksum = 0;

	/* Must be set to offload checksum. */
	mbuf->l2_len = sizeof(struct rte_ether_hdr);
	mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
	//mbuf->l4_len = sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr);
	mbuf->l4_len =  sizeof(struct rte_udp_hdr);
	mbuf->pkt_len = mbuf->l2_len +  mbuf->l3_len  + mbuf->l4_len + len ;
	mbuf->data_len = mbuf->pkt_len;

	/* Enable IPV4 CHECKSUM OFFLOAD */
	mbuf->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_IP_CKSUM);

	/* Enable UDP TX CHECKSUM OFFLOAD */
	mbuf->ol_flags |= (RTE_MBUF_F_TX_IPV4 | RTE_MBUF_F_TX_UDP_CKSUM);
#if 0
	uint32_t pseudo_cksum = csum32_add(
		csum32_add(iphdr->src_addr, iphdr->dst_addr),
		(iphdr->next_proto_id << 24) + udphdr->dgram_len
	);
    udphdr->dgram_cksum = csum16_add(pseudo_cksum & 0xFFFF, pseudo_cksum >> 16);
#endif
#else
	/* Must be set to offload checksum. */
	mbuf->l2_len = sizeof(struct rte_ether_hdr);
	mbuf->l3_len = sizeof(struct rte_ipv4_hdr);
	//mbuf->l4_len = sizeof(struct rte_ipv4_hdr) + sizeof(struct rte_udp_hdr);
	mbuf->l4_len =  sizeof(struct rte_udp_hdr);
	mbuf->pkt_len = mbuf->l2_len +  mbuf->l3_len  + mbuf->l4_len + len ;
	mbuf->data_len = mbuf->pkt_len;
#endif
    printf("Sending: udp src port: %d -> %d\n, ", ntohs(tx_src_udp_port), ntohs(tx_dst_udp_port));

    return 0;
    
}
#endif

static struct rte_mbuf *create_tx_mbuf(struct rte_mempool *mbuf_pool, uint8_t *udp_data, int data_len){
    struct rte_mbuf *mbuf = rte_pktmbuf_alloc(mbuf_pool);
    if(!mbuf){
        rte_exit(EXIT_FAILURE, "rte_pktmbuf_alloc\n");
    }


    
#if TEST_CHECKSUM
    encode_udp_pkt2(mbuf,  udp_data, data_len);
#else
    unsigned total_len = data_len + 36;

    mbuf->data_len = total_len;
    mbuf->pkt_len = total_len;
    uint8_t *pkt_buf = rte_pktmbuf_mtod(mbuf, uint8_t*);
    encode_udp_pkt(pkt_buf, total_len, udp_data, data_len);
#endif
    return mbuf;
}


int main(int argc, char * argv[])
{
    //初始化网卡设备
    if( rte_eal_init(argc, argv) < 0 ) {
        rte_exit(EXIT_FAILURE, "Error with EAL init\n");
    }

    //创建内存池
    struct rte_mempool *mbuf_pool = rte_pktmbuf_pool_create("mbuf pool", NUM_MBUFS, CACHE_SIZE, PRIV_SIZE, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id());
    if(mbuf_pool == NULL) {
        rte_exit(EXIT_FAILURE, "Error create mbuf pool");
    }

    init_port(mbuf_pool);
	
	rte_eth_macaddr_get(gPortId, &tx_src_mac);


    while(1) {
        struct rte_mbuf *mbufs[BURST_SIZE];
		unsigned num_recvd = rte_eth_rx_burst(gPortId, 0, mbufs, BURST_SIZE);
		if (num_recvd > BURST_SIZE) {
			rte_exit(EXIT_FAILURE, "Error receiving from eth\n");
		}

		unsigned i = 0;
		for (i = 0; i < num_recvd; i++) {

			struct rte_ether_hdr *ehdr = rte_pktmbuf_mtod(mbufs[i], struct rte_ether_hdr*);
			if (ehdr->ether_type != rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4)) {
			    rte_pktmbuf_free(mbufs[i]);
				continue;
			}

			struct rte_ipv4_hdr *iphdr =  rte_pktmbuf_mtod_offset(mbufs[i], struct rte_ipv4_hdr *, 
				sizeof(struct rte_ether_hdr));
			
			if (iphdr->next_proto_id == IPPROTO_UDP) {

				struct rte_udp_hdr *udphdr = (struct rte_udp_hdr *)(iphdr + 1);

				uint16_t length = ntohs(udphdr->dgram_len);
				*((char*)udphdr + length) = '\0';

				struct in_addr addr;
				rte_memcpy(&addr.s_addr, &iphdr->src_addr, sizeof(uint32_t));
				printf("src: %s:%d, ", inet_ntoa(addr), ntohs(udphdr->src_port));

				rte_memcpy(&addr.s_addr, &iphdr->dst_addr, sizeof(uint32_t));
				printf("dst: %s:%d, %s\n", inet_ntoa(addr), ntohs(udphdr->dst_port), 
					(char *)(udphdr+1));	


				rte_memcpy(&tx_dst_mac, &ehdr->s_addr, sizeof(struct rte_ether_addr));
				tx_src_ipaddr = iphdr->dst_addr;
				tx_dst_ipaddr = iphdr->src_addr;
				tx_src_udp_port = udphdr->dst_port;
				tx_dst_udp_port = udphdr->src_port;

				struct rte_mbuf *tx_mbuf = create_tx_mbuf(mbuf_pool, (uint8_t *)(udphdr+1), length);
				if(rte_eth_tx_burst(gPortId, 0, &tx_mbuf, 1) <= 0) {
					printf("Error Sending to eth\n");
				}
    			
				rte_pktmbuf_free(tx_mbuf);
			}

			rte_pktmbuf_free(mbufs[i]);
		}

    }

    return 0;
}



