/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2017 Mellanox Technologies, Ltd
 */
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdbool.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include "flow_blocks.h"
#define MAX_PATTERN_NUM		3
#define MAX_ACTION_NUM		2
#define STR_NOIP "NO IP"


/**
 * create a flow rule that sends packets with matching src and dest ip
 * to selected queue.
 *
 * @param port_id
 *   The selected port.
 * @param rx_q
 *   The selected target queue.
 * @param src_ip
 *   The src ip value to match the input packet.
 * @param src_mask
 *   The mask to apply to the src ip.
 * @param dest_ip
 *   The dest ip value to match the input packet.
 * @param dest_mask
 *   The mask to apply to the dest ip.
 * @param[out] error
 *   Perform verbose error reporting if not NULL.
 *
 * @return
 *   A flow if the rule could be created else return NULL.
 */
struct rte_flow *
generate_ipv4_flow(uint16_t port_id, uint16_t rx_q,
		uint32_t src_ip, uint32_t src_mask,
		uint32_t dest_ip, uint32_t dest_mask,
		struct rte_flow_error *error)
{
	struct rte_flow_attr attr;
	struct rte_flow_item pattern[MAX_PATTERN_NUM];
	struct rte_flow_action action[MAX_ACTION_NUM];
	struct rte_flow *flow = NULL;
	struct rte_flow_action_queue queue = { .index = rx_q };
	struct rte_flow_item_ipv4 ip_spec;
	struct rte_flow_item_ipv4 ip_mask;
	int res;

	memset(pattern, 0, sizeof(pattern));
	memset(action, 0, sizeof(action));

	/*
	 * set the rule attribute.
	 * in this case only ingress packets will be checked.
	 */
	memset(&attr, 0, sizeof(struct rte_flow_attr));
	attr.ingress = 1;

	/*
	 * create the action sequence.
	 * one action only,  move packet to queue
	 */
	action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
	action[0].conf = &queue;
	action[1].type = RTE_FLOW_ACTION_TYPE_END;

	/*
	 * set the first level of the pattern (ETH).
	 * since in this example we just want to get the
	 * ipv4 we set this level to allow all.
	 */
	pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;

	/*
	 * setting the second level of the pattern (IP).
	 * in this example this is the level we care about
	 * so we set it according to the parameters.
	 */
	memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
	memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));
	ip_spec.hdr.dst_addr = htonl(dest_ip);
	ip_mask.hdr.dst_addr = dest_mask;
	ip_spec.hdr.src_addr = htonl(src_ip);
	ip_mask.hdr.src_addr = src_mask;
	pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
	pattern[1].spec = &ip_spec;
	pattern[1].mask = &ip_mask;

	/* the final level must be always type end */
	pattern[2].type = RTE_FLOW_ITEM_TYPE_END;

	res = rte_flow_validate(port_id, &attr, pattern, action, error);
	if (!res)
		flow = rte_flow_create(port_id, &attr, pattern, action, error);

	return flow;
}


int ipv4_flow_action_mark_add(uint16_t port_id,uint16_t queue_id, uint16_t nr_queues, bool is_ip,
		uint32_t src_ip, uint32_t src_mask,
		uint32_t dest_ip, uint32_t dest_mask)
{
    int res;
    struct rte_flow *flow;
    struct rte_flow_attr attr;
    struct rte_flow_action action[3];
    struct rte_flow_action_mark normal_mark = {.id = 0xbef};
    struct rte_flow_action_rss rss;
    struct rte_flow_item pattern[2];
    struct rte_flow_item_ipv4 ip_spec;
    struct rte_flow_item_ipv4 ip_mask;
    struct rte_flow_error error;
    char str_flow[256];
    memset(pattern, 0, sizeof(pattern));
    memset(action, 0, sizeof(action));
    memset(&rss, 0, sizeof(rss));
    action[0].type = RTE_FLOW_ACTION_TYPE_MARK;
    action[0].conf = &normal_mark;
    /*
    * set the rule attribute.
    * in this case only ingress packets will be checked.
    */
    memset(&attr, 0, sizeof(struct rte_flow_attr));
    attr.ingress = 1;
    if (!is_ip) {
            struct rte_flow_action_queue queue;
            action[1].type = RTE_FLOW_ACTION_TYPE_QUEUE;
            queue.index = queue_id;
            action[1].conf = &queue;
        } else {
            
#if 1
            action[1].type = RTE_FLOW_ACTION_TYPE_RSS;
            uint8_t rss_key[40];
            uint16_t queue[RTE_MAX_QUEUES_PER_PORT];
            uint16_t i  = 0;
            for(; i < nr_queues; ++i) {
                queue[i] = i;
            }
            struct rte_eth_rss_conf rss_conf;
            rss_conf.rss_key = rss_key;
            rss_conf.rss_key_len = 40;
            rte_eth_dev_rss_hash_conf_get(port_id, &rss_conf);
            rss.types = rss_conf.rss_hf;
            rss.key_len = rss_conf.rss_key_len;
            rss.queue_num = nr_queues;
            rss.key = rss_key;
            rss.queue = queue;
            rss.level = 0;
            rss.func = RTE_ETH_HASH_FUNCTION_DEFAULT;
            action[1].conf = &rss;
#endif
        }
     action[2].type = RTE_FLOW_ACTION_TYPE_END;
     /*
      * setting the second level of the pattern (IP).
      * in this example this is the level we care about
      * so we set it according to the parameters.
      */
     memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
     memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));
     ip_spec.hdr.dst_addr = htonl(dest_ip);
     ip_mask.hdr.dst_addr = dest_mask;
     ip_spec.hdr.src_addr = htonl(src_ip);
     ip_mask.hdr.src_addr = src_mask;
     pattern[0].type = RTE_FLOW_ITEM_TYPE_IPV4;
     pattern[0].spec = &ip_spec;
     pattern[0].mask = &ip_mask;

     /* the final level must be always type end */
     pattern[1].type = RTE_FLOW_ITEM_TYPE_END;
    
     res = snprintf(str_flow, sizeof(str_flow),
		"DstIP=%u SrcIp=%u",
		rte_be_to_cpu_32(dest_ip), rte_be_to_cpu_32(src_ip));
    RTE_VERIFY(res > 0 && res < (int)sizeof(str_flow));
    res = rte_flow_validate(port_id, &attr, pattern, action, &error);
    if (!res)
	flow = rte_flow_create(port_id, &attr, pattern, action, &error);
    else {
        goto err1;
    }
    if (flow == NULL) {
        goto err1;
    }
    return 0;
err1:
	/* rte_errno is set to a positive errno value. */
	printf("%s(%u): cannot create IPv4 flow, errno=%i (%s), rte_flow_error_type=%i: %s\n",
		__func__,port_id , rte_errno, strerror(rte_errno),
			error.type, error.message);
	return -1;
}
int
ipv4_l4_flow_add(uint16_t port_id,uint16_t queue_id ,rte_be32_t dst_ip_be,
	rte_be16_t src_port_be, rte_be16_t src_port_mask_be,
	rte_be16_t dst_port_be, rte_be16_t dst_port_mask_be,
	uint8_t proto)
{
	struct rte_flow_attr attr = { .ingress = 1, };
	struct rte_flow_action_queue queue = { .index = queue_id };
	struct rte_flow_action action[] = {
		{
			.type = RTE_FLOW_ACTION_TYPE_QUEUE,
			.conf = &queue,
	       	},
		{
			.type = RTE_FLOW_ACTION_TYPE_END,
		}
	};
	struct rte_flow_item_eth eth_spec = {
		.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4),
	};
	struct rte_flow_item_eth eth_mask = {
		.type = 0xFFFF,
	};
	struct rte_flow_item_ipv4 ip_spec = {
		.hdr = {
			.dst_addr = dst_ip_be,
			.next_proto_id = proto,
		}
	};
	struct rte_flow_item_ipv4 ip_mask = {
		.hdr = {
			.dst_addr = 0xFFFFFFFF,
			.next_proto_id = 0xFF,
		}
	};
	struct rte_flow_item pattern[] = {
		{
			.type = RTE_FLOW_ITEM_TYPE_ETH,
			.spec = &eth_spec,
			.mask = &eth_mask,
		},
		{
			.type = RTE_FLOW_ITEM_TYPE_IPV4,
			.spec = &ip_spec,
			.mask = &ip_mask,
		},
		{
		         .type = RTE_FLOW_ITEM_TYPE_TCP,
                },
		{
			.type = RTE_FLOW_ITEM_TYPE_END,
		},
	};
	struct rte_flow *flow;
	struct rte_flow_item_tcp tcp_spec;
	struct rte_flow_item_tcp tcp_mask;
	struct rte_flow_item_udp udp_spec;
	struct rte_flow_item_udp udp_mask;
	struct rte_flow_error error;
	int ret;
	const char *str_proto = "NO PROTO";
	char str_dst_ip[INET_ADDRSTRLEN], str_flow[256];

#if 0
	if (!iface->rss) {
		/*
		 * IPv4 flows can only be used if supported by the NIC
		 * (to steer matching packets) and if RSS is supported
		 * (to steer non-matching packets elsewhere).
		 */
		printf("%s(%u): cannot use IPv4 flows when RSS is not supported\n",
			__func__, port_id);
		return -1;
	}
#endif
	if (proto == IPPROTO_TCP) {
		memset(&tcp_spec, 0, sizeof(tcp_spec));
		memset(&tcp_mask, 0, sizeof(tcp_mask));
		tcp_spec.hdr.src_port = src_port_be;
		tcp_mask.hdr.src_port = src_port_mask_be;
		tcp_spec.hdr.dst_port = dst_port_be;
		tcp_mask.hdr.dst_port = dst_port_mask_be;
		pattern[2].type = RTE_FLOW_ITEM_TYPE_TCP;
		pattern[2].spec = &tcp_spec;
		pattern[2].mask = &tcp_mask;
		str_proto = "TCP";
	} else if (proto == IPPROTO_UDP) {
		memset(&udp_spec, 0, sizeof(udp_spec));
		memset(&udp_mask, 0, sizeof(udp_mask));
		udp_spec.hdr.src_port = src_port_be;
		udp_mask.hdr.src_port = src_port_mask_be;
		udp_spec.hdr.dst_port = dst_port_be;
		udp_mask.hdr.dst_port = dst_port_mask_be;
		pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
		pattern[2].spec = &udp_spec;
		pattern[2].mask = &udp_mask;
		str_proto = "UDP";
	} else {
		printf("%s(%u): unexpected L4 protocol %hu for IPv4 flow\n",
			__func__, port_id, proto);
		return -1;
	}

	/* Get a human-readable description of the flow. */
	if (unlikely(inet_ntop(AF_INET, &dst_ip_be,
			str_dst_ip, sizeof(str_dst_ip)) == NULL)) {
		printf("%s(%u): inet_ntop() failed, errno=%i: %s\n",
			__func__, port_id, errno, strerror(errno));
		RTE_BUILD_BUG_ON(sizeof(STR_NOIP) > sizeof(str_dst_ip));
		strcpy(str_dst_ip, STR_NOIP);
	}
	ret = snprintf(str_flow, sizeof(str_flow),
		"DstIP=%s %s SrcPort=%i/0x%x DstPort=%i/0x%x",
		str_dst_ip, str_proto,
		rte_be_to_cpu_16(src_port_be),
		rte_be_to_cpu_16(src_port_mask_be),
		rte_be_to_cpu_16(dst_port_be),
		rte_be_to_cpu_16(dst_port_mask_be));
	RTE_VERIFY(ret > 0 && ret < (int)sizeof(str_flow));

	ret = rte_flow_validate(port_id, &attr, pattern, action, &error);
	if (ret < 0) {
		/*
		 * A negative errno value was returned
		 * (and also put in rte_errno).
		 */
		printf("%s(%u, %s): cannot validate IPv4 flow, errno=%i (%s), rte_flow_error_type=%i: %s\n",
			__func__, port_id, str_flow,
			-ret, strerror(-ret),
			error.type, error.message);
		return -1;
	}

	flow = rte_flow_create(port_id, &attr, pattern, action, &error);
	if (flow == NULL) {
		/* rte_errno is set to a positive errno value. */
		printf("%s(%u, %s): cannot create IPv4 flow, errno=%i (%s), rte_flow_error_type=%i: %s\n",
			__func__,port_id , str_flow,
			rte_errno, strerror(rte_errno),
			error.type, error.message);
		return -1;
	}

	printf("%s(%u, %s): IPv4 flow supported\n",
		__func__, port_id, str_flow);
	return 0;
}
