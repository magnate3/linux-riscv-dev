#ifndef __FLOW_BLOCK__
#define __FLOW_BLOCK__
#include <rte_flow.h>
struct rte_flow *
generate_ipv4_flow(uint16_t port_id, uint16_t rx_q,
		uint32_t src_ip, uint32_t src_mask,
		uint32_t dest_ip, uint32_t dest_mask,
		struct rte_flow_error *error);
int ipv4_l4_flow_add(uint16_t port_id,uint16_t queue_id ,rte_be32_t dst_ip_be,
	rte_be16_t src_port_be, rte_be16_t src_port_mask_be,
	rte_be16_t dst_port_be, rte_be16_t dst_port_mask_be,
	uint8_t proto);
int ipv4_flow_action_mark_add(uint16_t port_id,uint16_t queue_id, uint16_t nr_queues, bool is_ip,
		uint32_t src_ip, uint32_t src_mask,
		uint32_t dest_ip, uint32_t dest_mask);
int flow_new(uint8_t port_id, uint16_t rxq, uint32_t sip, uint32_t dip);
void i40eDeviceSetRSSHashFunction(uint64_t *rss_hf);
int i40eDeviceSetRSSFlowIPv4(int port_id, const char *port_name, struct rte_eth_rss_conf *rss_conf);
int rss_config_port(struct rte_eth_conf *conf, struct rte_eth_dev_info *dev_info);
void rss_init(void);
#endif
