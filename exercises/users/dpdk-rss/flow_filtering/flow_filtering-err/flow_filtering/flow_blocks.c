/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright 2017 Mellanox Technologies, Ltd
 */
#include <sys/socket.h>
#include <arpa/inet.h>
#include <errno.h>
#include <stdbool.h>
#include <rte_errno.h>
#include <rte_ethdev.h>
#include <rte_version.h>
#include <rte_thash.h>
#include "flow_blocks.h"
#define MAX_PATTERN_NUM		3
#define MAX_ACTION_NUM		2
#define MAX_PATTERN_NUM4        4	
#define MAX_ACTION_NUM2		2
#define STR_NOIP "NO IP"
extern uint16_t nr_queues;
#define RSS_HASH_KEY_LENGTH 40
#define RSS_NONE            0
#define RSS_L3              1
#define RSS_L3L4            2
static uint8_t rss_hash_key_symmetric_be[RSS_HASH_KEY_LENGTH];
static uint8_t rss_hash_key_symmetric[RSS_HASH_KEY_LENGTH] = {
    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
    0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A, 0x6D, 0x5A,
};
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
#if 0
	struct rte_flow_item_eth eth_spec = {
		.type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4),
	};
	struct rte_flow_item_eth eth_mask = {
		//.type = 0xFFFF,
                //.dst.addr_bytes = "\xff\xff\xff\xff\xff\xff",
	};
#else
          /*
           *    * set the first level of the pattern (ETH).
           *       * since in this example we just want to get the
           *          * ipv4 we set this level to allow all.
           **/
        struct rte_flow_item_eth eth_spec;
        struct rte_flow_item_eth eth_mask;
        memset(&eth_spec, 0, sizeof(struct rte_flow_item_eth));
        memset(&eth_mask, 0, sizeof(struct rte_flow_item_eth));
        eth_spec.type = rte_cpu_to_be_16(0x8100);
        eth_mask.type = 0xffff;
        memcpy(&eth_spec.dst, "F41D6BF7BF96", sizeof(struct rte_ether_addr));
        //memset(&eth_spec.dst, 0x00, RTE_ETHER_ADDR_LEN);
        memset(&eth_mask.dst, 0xff, RTE_ETHER_ADDR_LEN);
        memset(&eth_spec.src, 0x00, RTE_ETHER_ADDR_LEN);
        memset(&eth_mask.src, 0x00, RTE_ETHER_ADDR_LEN);
#endif
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
#if 1
void generate_udp_fdir_rule(uint16_t port_id, uint16_t rx_queue_id,uint32_t ip_dst, uint16_t dst_port) {
     struct rte_flow_attr attr;
     struct rte_flow_item pattern[MAX_PATTERN_NUM4];
     struct rte_flow_action action[MAX_ACTION_NUM2];
     struct rte_flow_action_queue queue;
     struct rte_flow_item_ipv4 ip_spec, ip_mask;
     queue.index = rx_queue_id;
     struct rte_flow_item_udp udp_spec;
     struct rte_flow_item_udp udp_mask;

     // only check ingress packet
     memset(&attr, 0, sizeof(struct rte_flow_attr));
     attr.ingress = 1;

     memset(pattern, 0, sizeof(pattern));
     memset(action, 0, sizeof(action));
     // place into the specific queue
     action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
     action[0].conf = &queue;
     action[1].type = RTE_FLOW_ACTION_TYPE_END;

     // first-level pattern: allow all ethernet header
     pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
     // second-level pattern: allow all ipv4 header
     memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
     memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));
     ip_spec.hdr.dst_addr = htonl(ip_dst);
     ip_mask.hdr.dst_addr = htonl(0xffffffff);
     pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
#if  1
     //ip_spec.hdr.next_proto_id = IPPROTO_UDP;
     //ip_mask.hdr.next_proto_id = 0xff;
     ip_spec.hdr.src_addr = 0;
     ip_mask.hdr.src_addr = 0;
     pattern[1].spec = &ip_spec;
     pattern[1].mask = &ip_mask;
#endif
	// third-level pattern: match udp.dstport
	pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
	memset(&udp_spec, 0, sizeof(struct rte_flow_item_udp));
	memset(&udp_mask, 0, sizeof(struct rte_flow_item_udp));
	udp_spec.hdr.src_port = 0;
	udp_mask.hdr.src_port = 0; // allow all src ports
#if 0
	udp_spec.hdr.dst_port = rte_cpu_to_le_16(dst_port);
	udp_mask.hdr.dst_port = rte_cpu_to_le_16(0xFFFF); // only allow specific destination port
#else
	udp_spec.hdr.dst_port = htons(dst_port);
	udp_mask.hdr.dst_port = htons(0xFFFF); // only allow specific destination port
#endif
	pattern[2].spec = &udp_spec;
	pattern[2].mask = &udp_mask;
	pattern[2].last = 0; // disable range match
	// last-level pattern: end of pattern list
	pattern[3].type = RTE_FLOW_ITEM_TYPE_END;

	struct rte_flow_error error;
	int res = rte_flow_validate(port_id, &attr, pattern, action, &error);
	if (res != 0) {
		printf("[%s] flow validate error: type %d, message %s\n", __func__,error.type, error.message ? error.message : "(no stated reason)");
    	rte_exit(EXIT_FAILURE, "flow validate error");
	}

	struct rte_flow *flow = NULL;
	flow = rte_flow_create(port_id, &attr, pattern, action, &error);
	if (flow == NULL) {
		printf("[%s] flow create error: type %d, message %s\n",__func__, error.type, error.message ? error.message : "(no stated reason)");
    	rte_exit(EXIT_FAILURE, "flow create error");
	}
        printf("[%s] flow create success \n",__func__);
}
#else
static struct rte_flow_item eth_item = {
	RTE_FLOW_ITEM_TYPE_ETH,
	0, 0, 0
};
static struct rte_flow_item end_item = {
	RTE_FLOW_ITEM_TYPE_END,
	0, 0, 0
};
static struct rte_flow_action end_action = {
	RTE_FLOW_ACTION_TYPE_END,
	0
};
void generate_udp_fdir_rule(uint16_t port_id, uint16_t rx_queue_id,uint32_t ip_dst, uint16_t dst_port) {
     struct rte_flow_attr attr;
     struct rte_flow_item pattern[MAX_PATTERN_NUM4];
     struct rte_flow_action action[MAX_ACTION_NUM2];
     struct rte_flow_action_queue queue;
     struct rte_flow_item_ipv4 ip_spec, ip_mask;
     struct rte_flow_item_udp udp_spec;
     struct rte_flow_item_udp udp_mask;

     // only check ingress packet
     memset(&attr, 0, sizeof(struct rte_flow_attr));
     attr.ingress = 1;

     queue.index = rx_queue_id;
     memset(pattern, 0, sizeof(pattern));
     memset(action, 0, sizeof(action));
     // place into the specific queue
     action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
     action[0].conf = &queue;
     action[1].type = RTE_FLOW_ACTION_TYPE_END;

     // first-level pattern: allow all ethernet header
     pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
     // second-level pattern: allow all ipv4 header
     memset(&ip_spec, 0, sizeof(struct rte_flow_item_ipv4));
     memset(&ip_mask, 0, sizeof(struct rte_flow_item_ipv4));
#if 1
     ip_spec.hdr.next_proto_id = IPPROTO_UDP;
     ip_mask.hdr.next_proto_id = 0xff;
#endif
     ip_spec.hdr.src_addr = 0;
     ip_mask.hdr.src_addr = 0;
     printf("[%s] flow dst ip<%u> \n",__func__,ip_dst);
     ip_spec.hdr.dst_addr = htonl(ip_dst);
     ip_mask.hdr.dst_addr = 0xffffffff;
     pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
     pattern[1].spec = &ip_spec;
     pattern[1].mask = &ip_mask;
     //pattern[1].last = 0; // disable range match
     // last-level pattern: end of pattern list
     pattern[2].type = RTE_FLOW_ITEM_TYPE_END;

     struct rte_flow_error error;
     int res = rte_flow_validate(port_id, &attr, pattern, action, &error);
     if (res != 0) {
		printf("[%s] flow validate error: type %d, message %s\n", __func__,error.type, error.message ? error.message : "(no stated reason)");
    	rte_exit(EXIT_FAILURE, "flow validate error");
     }

     struct rte_flow *flow = NULL;
     flow = rte_flow_create(port_id, &attr, pattern, action, &error);
     if (flow == NULL) {
		printf("[%s] flow create error: type %d, message %s\n",__func__, error.type, error.message ? error.message : "(no stated reason)");
    	rte_exit(EXIT_FAILURE, "flow create error");
     }
     printf("[%s] flow create success \n",__func__);
}
#endif
static void flow_pattern_init_eth(struct rte_flow_item *pattern, struct rte_flow_item_eth *spec,
    struct rte_flow_item_eth *mask)
{
    memset(spec, 0, sizeof(struct rte_flow_item_eth));
    memset(mask, 0, sizeof(struct rte_flow_item_eth));
#if 1
    spec->type = 0;
    mask->type = 0;
#else
    spec->type = rte_cpu_to_be_16(RTE_ETHER_TYPE_IPV4);
    mask->type = 0xFFFF;
#endif
    memset(pattern, 0, sizeof(struct rte_flow_item));
    pattern->type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern->spec = spec;
    pattern->mask = mask;
}

static void flow_pattern_init_ipv4(struct rte_flow_item *pattern, struct rte_flow_item_ipv4 *spec,
    struct rte_flow_item_ipv4 *mask, uint32_t sip, uint32_t dip)
{
    uint32_t smask = 0;
    uint32_t dmask = 0;

    if (sip != 0) {
        smask = 0xffffffff;
    }

    if (dip != 0) {
        //dmask = 0xffffffff;
        dmask = 0xffffff00;
    }

    memset(spec, 0, sizeof(struct rte_flow_item_ipv4));
    spec->hdr.dst_addr = dip;
    spec->hdr.src_addr = sip;

    memset(mask, 0, sizeof(struct rte_flow_item_ipv4));
    mask->hdr.dst_addr = htons(dmask);
    mask->hdr.src_addr = smask;

#if 0
    spec->hdr.type_of_service = 0x01;
    mask->hdr.type_of_service = 0x01; 
#endif
    memset(pattern, 0, sizeof(struct rte_flow_item));
    pattern->type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern->spec = spec;
    pattern->mask = mask;
}
static void flow_pattern_init_end(struct rte_flow_item *pattern)
{
    memset(pattern, 0, sizeof(struct rte_flow_item));
    pattern->type = RTE_FLOW_ITEM_TYPE_END;
}

static void flow_action_init(struct rte_flow_action *action, struct rte_flow_action_queue *queue, uint16_t rxq)
{
    memset(action, 0, sizeof(struct rte_flow_action) * 2);
    memset(queue, 0, sizeof(struct rte_flow_action_queue));

    queue->index = rxq;
    action[0].type = RTE_FLOW_ACTION_TYPE_QUEUE;
    action[0].conf = queue;
    action[1].type = RTE_FLOW_ACTION_TYPE_END;
}

static int flow_create(uint8_t port_id, struct rte_flow_item *pattern, struct rte_flow_action *action)
{
    int ret = 0;
    struct rte_flow *flow = NULL;
    struct rte_flow_error err;
    struct rte_flow_attr attr;

    memset(&attr, 0, sizeof(struct rte_flow_attr));
    attr.ingress = 1;
    attr.group = 0;

    ret = rte_flow_validate(port_id, &attr, pattern, action, &err);
    if (ret < 0) {
        printf("Error: Interface dose not support FDIR. Please use 'rss'!\n");
        return -1;
    }

    flow = rte_flow_create(port_id, &attr, pattern, action, &err);
    if (flow == NULL) {
        printf("Error: Flow create error\n");
        return -1;
    }
    printf("rte flow create successfully");
    return 0;
}
static void flow_pattern_init_udp(struct rte_flow_item *pattern,struct rte_flow_item_udp  *udp_spec, struct rte_flow_item_udp  *udp_mask, uint16_t dst_port)
{
    uint32_t dmask = 0;
    if (dst_port != 0) {
        dmask = 0xffff;
    }
    memset(udp_spec, 0, sizeof(struct rte_flow_item_udp));
    memset(udp_mask, 0, sizeof(struct rte_flow_item_udp));
    udp_spec->hdr.dst_port = htons(dst_port);
    udp_mask->hdr.dst_port = dmask;
    pattern->type = RTE_FLOW_ITEM_TYPE_UDP;
    pattern->spec = udp_spec;
    pattern->mask = udp_mask;
}
int flow_new(uint8_t port_id, uint16_t rxq, uint32_t sip, uint32_t dip,uint16_t dst_port)
{
    struct rte_flow_action_queue queue;
    struct rte_flow_item_eth eth_spec, eth_mask;
    struct rte_flow_item_ipv4 ip_spec, ip_mask;
    struct rte_flow_item_udp  udp_spec;
    struct rte_flow_item_udp  udp_mask;
    struct rte_flow_item pattern[MAX_PATTERN_NUM4];
    struct rte_flow_action action[MAX_PATTERN_NUM4];

    flow_action_init(action, &queue, rxq);
    flow_pattern_init_eth(&pattern[0], &eth_spec, &eth_mask);
    flow_pattern_init_ipv4(&pattern[1], &ip_spec, &ip_mask, sip, dip);
#if 1
    flow_pattern_init_udp(&pattern[2],&udp_spec,&udp_mask,dst_port);
    flow_pattern_init_end(&pattern[3]);
#else
    flow_pattern_init_end(&pattern[2]);
#endif
    return flow_create(port_id, pattern, action);
}
static int i40eDeviceCreateRSSFlow(int port_id, const char *port_name,
        struct rte_eth_rss_conf *rss_conf, uint64_t rss_type, struct rte_flow_item *pattern)
{
    struct rte_flow_action_rss rss_action_conf = { 0 };
    struct rte_flow_attr attr = { 0 };
    struct rte_flow_action action[] = { { 0 }, { 0 } };
    struct rte_flow *flow;
    struct rte_flow_error flow_error = { 0 };
    uint16_t queue_offset = 2;
    uint16_t i, queue_aval = nr_queues-queue_offset;
    uint16_t queues[RTE_MAX_QUEUES_PER_PORT]={3,4};
    for (i = 0; i < queue_aval; ++i) {
        queues[i] = i+ queue_offset;
    }
    rss_action_conf.func = RTE_ETH_HASH_FUNCTION_SYMMETRIC_TOEPLITZ;
    rss_action_conf.level = 0;
    rss_action_conf.types = rss_type;
    rss_action_conf.key_len = rss_conf->rss_key_len;
    rss_action_conf.key = rss_conf->rss_key;
    rss_action_conf.queue_num = queue_aval;
    rss_action_conf.queue = queues;

    attr.ingress = 1;
    action[0].type = RTE_FLOW_ACTION_TYPE_RSS;
    action[0].conf = &rss_action_conf;
    action[1].type = RTE_FLOW_ACTION_TYPE_END;
    int ret = rte_flow_validate(port_id, &attr, pattern, action, &flow_error);
    if (ret < 0) {
        printf("Error on rte_flow validation for port %s: %s errmsg: %s \n",
                    port_name, rte_strerror(-ret), flow_error.message);
        return ret;
    }
    flow = rte_flow_create(port_id, &attr, pattern, action, &flow_error);
    if (flow == NULL) {
        printf("Error when creating rte_flow rule on %s: %s", port_name,
                flow_error.message);
    } else {
        printf("RTE_FLOW flow rule created for port %s", port_name);
    }
    return 0;
}
int i40eDeviceSetRSSFlowIPv4(
        int port_id, const char *port_name, struct rte_eth_rss_conf *rss_conf)
{
    int ret = 0;
    struct rte_flow_item pattern[] = { { 0 }, { 0 }, { 0 }, { 0 } };

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_END;
    ret |= i40eDeviceCreateRSSFlow(
            port_id, port_name, rss_conf, RTE_ETH_RSS_NONFRAG_IPV4_OTHER, pattern);
    memset(pattern, 0, sizeof(pattern));

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_UDP;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;
    ret |= i40eDeviceCreateRSSFlow(port_id, port_name, rss_conf, RTE_ETH_RSS_NONFRAG_IPV4_UDP, pattern);
    memset(pattern, 0, sizeof(pattern));

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_TCP;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;
    ret |= i40eDeviceCreateRSSFlow(port_id, port_name, rss_conf, RTE_ETH_RSS_NONFRAG_IPV4_TCP, pattern);
    memset(pattern, 0, sizeof(pattern));

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_SCTP;
    pattern[3].type = RTE_FLOW_ITEM_TYPE_END;
    ret |= i40eDeviceCreateRSSFlow(
            port_id, port_name, rss_conf, RTE_ETH_RSS_NONFRAG_IPV4_SCTP, pattern);
    memset(pattern, 0, sizeof(pattern));

    pattern[0].type = RTE_FLOW_ITEM_TYPE_ETH;
    pattern[1].type = RTE_FLOW_ITEM_TYPE_IPV4;
    pattern[2].type = RTE_FLOW_ITEM_TYPE_END;
    ret |= i40eDeviceCreateRSSFlow(port_id, port_name, rss_conf, RTE_ETH_RSS_FRAG_IPV4, pattern);
    return ret;
}
void i40eDeviceSetRSSHashFunction(uint64_t *rss_hf)
{
    #if RTE_VERSION < RTE_VERSION_NUM(21, 0, 0, 0)
        *rss_hf = ETH_RSS_FRAG_IPV4 | ETH_RSS_NONFRAG_IPV4_TCP | ETH_RSS_NONFRAG_IPV4_UDP |
                  ETH_RSS_NONFRAG_IPV4_SCTP | ETH_RSS_NONFRAG_IPV4_OTHER | ETH_RSS_FRAG_IPV6 |
                  ETH_RSS_NONFRAG_IPV6_TCP | ETH_RSS_NONFRAG_IPV6_UDP | ETH_RSS_NONFRAG_IPV6_SCTP |
                  ETH_RSS_NONFRAG_IPV6_OTHER | ETH_RSS_SCTP;
    #else
        *rss_hf = RTE_ETH_RSS_FRAG_IPV4 | RTE_ETH_RSS_NONFRAG_IPV4_OTHER | RTE_ETH_RSS_FRAG_IPV6 |
                  RTE_ETH_RSS_NONFRAG_IPV6_OTHER;
   #endif
}
static uint64_t rss_get_rss_hf(struct rte_eth_dev_info *dev_info, uint8_t rss, bool ipv6)
{
    uint64_t offloads = 0;
    uint64_t ipv4_flags = 0;
    uint64_t ipv6_flags = 0;

    offloads = dev_info->flow_type_rss_offloads;
    if (rss == RSS_L3) {
        ipv4_flags = RTE_ETH_RSS_IPV4 | RTE_ETH_RSS_FRAG_IPV4;
        ipv6_flags = RTE_ETH_RSS_IPV6 | RTE_ETH_RSS_FRAG_IPV6;
    } else if (rss == RSS_L3L4) {
        ipv4_flags = RTE_ETH_RSS_NONFRAG_IPV4_UDP | RTE_ETH_RSS_NONFRAG_IPV4_TCP;
        ipv6_flags = RTE_ETH_RSS_NONFRAG_IPV6_UDP | RTE_ETH_RSS_NONFRAG_IPV6_TCP;
    }

    if (ipv6) {
        if ((offloads & ipv6_flags) == 0) {
            return 0;
        }
    } else {
        if ((offloads & ipv4_flags) == 0) {
            return 0;
        }
    }

    return (offloads & (ipv4_flags | ipv6_flags));
}
int rss_config_port(struct rte_eth_conf *conf, struct rte_eth_dev_info *dev_info)
{
    uint64_t rss_hf = 0;
    struct rte_eth_rss_conf *rss_conf = NULL;

    rss_conf = &conf->rx_adv_conf.rss_conf;
#if 0
    if (g_config.rss == RSS_AUTO) {
        if (g_config.mq_rx_rss) {
            conf->rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;
            rss_conf->rss_hf = rss_get_rss_hf(dev_info, g_config.rss_auto);
        }
        return 0;
    }
#endif
    rss_hf = rss_get_rss_hf(dev_info, RSS_L3L4,false);
    //rss_hf = rss_get_rss_hf(dev_info, RSS_L3,false);
    if (rss_hf == 0) {
        return -1;
    }

    conf->rxmode.mq_mode = RTE_ETH_MQ_RX_RSS;
#if DEBUG_I40E
#else
    rss_conf->rss_key = rss_hash_key_symmetric;
    rss_conf->rss_key_len = RSS_HASH_KEY_LENGTH,
#endif
    rss_conf->rss_hf = rss_hf;

    return 0;
}
void rss_init(void)
{
    rte_convert_rss_key((const uint32_t *)rss_hash_key_symmetric,
                        (uint32_t *)rss_hash_key_symmetric_be, RSS_HASH_KEY_LENGTH);
}
