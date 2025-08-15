// SPDX-License-Identifier: GPL-2.0-or-later
/*
 *  SR-IPv6 implementation
 *
 *  Authors:
 *  David Lebrun <david.lebrun@uclouvain.be>
 *  eBPF support: Mathieu Xhonneux <m.xhonneux@gmail.com>
 */

#include <linux/types.h>
#include <linux/skbuff.h>
#include <linux/net.h>
#include <linux/module.h>
#include <linux/netfilter.h>
#include <net/ip.h>
#include <net/lwtunnel.h>
#include <net/netevent.h>
#include <net/netns/generic.h>
#include <net/ip6_fib.h>
#include <net/route.h>
#include <net/seg6.h>
#include <linux/seg6.h>
#include <linux/seg6_local.h>
#include <net/addrconf.h>
#include <net/ip6_route.h>
#include <net/dst_cache.h>
#include <net/ip_tunnels.h>
#ifdef CONFIG_IPV6_SEG6_HMAC
#include <net/seg6_hmac.h>
#endif
#include <net/seg6_local.h>
#include <linux/etherdevice.h>
#include <linux/bpf.h>

#define LEGACY_DT4 1
#define LEGACY_DT6 1

struct seg6_local_lwt;

static struct sk_buff *end_dt_vrf_core(struct sk_buff *skb, struct seg6_local_lwt *slwt, u16 family);

struct seg6_action_desc {
	int action;
	unsigned long attrs;
	int (*input)(struct sk_buff *skb, struct seg6_local_lwt *slwt);
	int static_headroom;
};

struct bpf_lwt_prog {
	struct bpf_prog *prog;
	char *name;
};

struct seg6_local_lwt {
	int action;
	struct ipv6_sr_hdr *srh;
	int table;
	struct in_addr nh4;
	struct in6_addr nh6;
	int iif;
	int oif;
	struct bpf_lwt_prog bpf;

	struct sr6_usid_info    usid_info;

	int loc_len;
	int src_pos;

	int headroom;
	struct seg6_action_desc *desc;
};

int seg6_lookup_nexthop(struct sk_buff *skb, struct in6_addr *nhaddr, u32 tbl_id);

static int seg6_local_deliver (struct sk_buff *skb)
{
	struct ipv6hdr ip6;
	struct ipv6hdr *hdr;
	struct ipv6_sr_hdr *srh = NULL;
	unsigned int srhoff = 0;
	unsigned int off = 0;
	int proto;

	if (ipv6_find_hdr(skb, &srhoff, IPPROTO_ROUTING, NULL, NULL) >= 0) {
		hdr = ipv6_hdr(skb);
		memcpy(&ip6, hdr, sizeof(struct ipv6hdr));

		srh = (struct ipv6_sr_hdr *)(skb->data + srhoff);
	}

	/* find ICMP or UDP header */
	if (ipv6_find_hdr(skb, &off, IPPROTO_ICMPV6, NULL, NULL) >= 0) {
		proto = IPPROTO_ICMPV6;
	} else if (ipv6_find_hdr(skb, &off, IPPROTO_UDP, NULL, NULL) >= 0) {
		proto = IPPROTO_UDP;
	} else {
		goto drop;
	}

	if (srh) {
		/* Strip off SRH */
		if (!pskb_pull(skb, off)) {
			goto drop;
		}

		skb_postpull_rcsum(skb, skb_network_header(skb), off);

		ip6.nexthdr = proto;

		/* Push new IPv6 header */
		skb_push(skb, sizeof(struct ipv6hdr));
		memcpy(skb->data, &ip6, sizeof(struct ipv6hdr));

		skb_postpush_rcsum(skb, skb->data, sizeof(struct ipv6hdr));

		skb_reset_network_header(skb);
        skb_reset_transport_header(skb);
        skb->encapsulation = 0;
	}

	skb->transport_header = skb->network_header + sizeof(struct ipv6hdr);
	IP6CB(skb)->nhoff = offsetof(struct ipv6hdr, nexthdr);

    /*
     * RTF_LOCAL should be set when the destination is matched to local SID.
     */
    //ipv6_unicast_destination_set(skb);

    return ip6_input(skb);

drop:
	return -EINVAL;
}

static struct seg6_local_lwt *seg6_local_lwtunnel(struct lwtunnel_state *lwt)
{
	return (struct seg6_local_lwt *)lwt->data;
}

static struct ipv6_sr_hdr *get_srh(struct sk_buff *skb)
{
	struct ipv6_sr_hdr *srh;
	int len, srhoff = 0;

	if (ipv6_find_hdr(skb, &srhoff, IPPROTO_ROUTING, NULL, NULL) < 0)
		return NULL;

	if (!pskb_may_pull(skb, srhoff + sizeof(*srh)))
		return NULL;

	srh = (struct ipv6_sr_hdr *)(skb->data + srhoff);

	len = (srh->hdrlen + 1) << 3;

	if (!pskb_may_pull(skb, srhoff + len))
		return NULL;

	/* note that pskb_may_pull may change pointers in header;
	 * for this reason it is necessary to reload them when needed.
	 */
	srh = (struct ipv6_sr_hdr *)(skb->data + srhoff);

	if (!seg6_validate_srh(srh, len))
		return NULL;

	return srh;
}

static struct ipv6_sr_hdr *get_and_validate_srh(struct sk_buff *skb)
{
	struct ipv6_sr_hdr *srh;

	srh = get_srh(skb);
	if (!srh)
		return NULL;

	if (srh->segments_left == 0)
		return NULL;

#ifdef CONFIG_IPV6_SEG6_HMAC
	if (!seg6_hmac_validate_skb(skb))
		return NULL;
#endif

	return srh;
}

static bool check_usid_block (struct in6_addr *daddr, struct seg6_local_lwt *slwt)
{
	if (slwt->usid_info.usid_block_len != 0) {
		return true;
	}

	return false;
}

static u32 get_next_usid (struct in6_addr *daddr, struct seg6_local_lwt *slwt)
{
	int pos;
	u32 *usid;

	pos = slwt->usid_info.usid_block_len + slwt->usid_info.usid_len;

	usid = (u32 *)&daddr->s6_addr[pos / 8];

	return *usid;
}

static int advance_nextusid (struct in6_addr *daddr, struct seg6_local_lwt *slwt)
{
	int base = slwt->usid_info.usid_block_len / 8;
	int pos = slwt->usid_info.usid_len / 8;
	int index;

	for (index = base; index < 16 - pos; index++) {
		daddr->s6_addr[index] = daddr->s6_addr[index+pos];
	}

	for (index = 16 - pos; index < 16; index++) {
		daddr->s6_addr[index] = 0;
	}

	return 0;
}

static bool decap_and_validate(struct sk_buff *skb, int proto)
{
	struct ipv6_sr_hdr *srh;
	unsigned int off = 0;

	srh = get_srh(skb);
	if (srh && srh->segments_left > 0)
		return false;

#ifdef CONFIG_IPV6_SEG6_HMAC
	if (srh && !seg6_hmac_validate_skb(skb))
		return false;
#endif

	if (ipv6_find_hdr(skb, &off, proto, NULL, NULL) < 0)
		return false;

	if (!pskb_pull(skb, off))
		return false;

	skb_postpull_rcsum(skb, skb_network_header(skb), off);

	skb_reset_network_header(skb);
	skb_reset_transport_header(skb);
	if (iptunnel_pull_offloads(skb))
		return false;

	if (proto == IPPROTO_IPV6)
		skb_set_transport_header(skb, sizeof(struct ipv6hdr));
	else if (proto == IPPROTO_IPIP)
		skb_set_transport_header(skb, sizeof(struct iphdr));
 
	return true;
}

static void advance_nextseg(struct ipv6_sr_hdr *srh, struct in6_addr *daddr)
{
	struct in6_addr *addr;

	srh->segments_left--;
	addr = srh->segments + srh->segments_left;
	*daddr = *addr;
}

int seg6_lookup_nexthop(struct sk_buff *skb, struct in6_addr *nhaddr,
			u32 tbl_id)
{
	struct net *net = dev_net(skb->dev);
	struct ipv6hdr *hdr = ipv6_hdr(skb);
	int flags = RT6_LOOKUP_F_HAS_SADDR;
	struct dst_entry *dst = NULL;
	struct rt6_info *rt;
	struct flowi6 fl6;

	fl6.flowi6_iif = skb->dev->ifindex;
	fl6.daddr = nhaddr ? *nhaddr : hdr->daddr;
	fl6.saddr = hdr->saddr;
	fl6.flowlabel = ip6_flowinfo(hdr);
	fl6.flowi6_mark = skb->mark;
	fl6.flowi6_proto = hdr->nexthdr;

	if (nhaddr)
		fl6.flowi6_flags = FLOWI_FLAG_KNOWN_NH;

	if (!tbl_id) {
		dst = ip6_route_input_lookup(net, skb->dev, &fl6, skb, flags);
	} else {
		struct fib6_table *table;

		table = fib6_get_table(net, tbl_id);
		if (!table)
			goto out;

		rt = ip6_pol_route(net, table, 0, &fl6, skb, flags);
		dst = &rt->dst;
	}

out:
	if (!dst) {
		rt = net->ipv6.ip6_blk_hole_entry;
		dst = &rt->dst;
		dst_hold(dst);
	}

	skb_dst_drop(skb);
	skb_dst_set(skb, dst);

	return dst->error;
}

int seg6_lookup_nexthop_iface(struct sk_buff *skb, struct in6_addr *nhaddr,
			u32 iif, u32 tbl_id)
{
	struct net *net = dev_net(skb->dev);
	struct ipv6hdr *hdr = ipv6_hdr(skb);
	int flags = RT6_LOOKUP_F_HAS_SADDR;
	struct dst_entry *dst = NULL;
	struct rt6_info *rt;
	struct flowi6 fl6;

	fl6.flowi6_iif = iif;
	fl6.daddr = nhaddr ? *nhaddr : hdr->daddr;
	fl6.saddr = hdr->saddr;
	fl6.flowlabel = ip6_flowinfo(hdr);
	fl6.flowi6_mark = skb->mark;
	fl6.flowi6_proto = hdr->nexthdr;

	if (nhaddr)
		fl6.flowi6_flags = FLOWI_FLAG_KNOWN_NH;

	if (!tbl_id) {
		dst = ip6_route_input_lookup(net, skb->dev, &fl6, skb, flags);
	} else {
		struct fib6_table *table;

		table = fib6_get_table(net, tbl_id);
		if (!table)
			goto out;

		rt = ip6_pol_route(net, table, iif, &fl6, skb, flags);
		dst = &rt->dst;
	}

out:
	if (!dst) {
		rt = net->ipv6.ip6_blk_hole_entry;
		dst = &rt->dst;
		dst_hold(dst);
	}

	skb_dst_drop(skb);
	skb_dst_set(skb, dst);
	return dst->error;
}

/* regular endpoint function */
static int input_action_end(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	struct ipv6_sr_hdr *srh;
	u32 next_usid;

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	if (check_usid_block(&ip6->daddr, slwt)) {
		/* uSID carrier */
		next_usid = get_next_usid(&ip6->daddr, slwt);
		if (next_usid != 0) {
			advance_nextusid(&ip6->daddr, slwt);
			seg6_lookup_nexthop(skb, NULL, 0);
			return dst_input(skb);
		}
	}

	srh = get_and_validate_srh(skb);
	if (!srh)
		goto drop;

	advance_nextseg(srh, &ipv6_hdr(skb)->daddr);

	seg6_lookup_nexthop(skb, NULL, 0);

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

/* regular endpoint, and forward to specified nexthop */
static int input_action_end_x(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	struct ipv6_sr_hdr *srh;
	u32 next_usid;

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	if (check_usid_block(&ip6->daddr, slwt)) {
		/* uSID carrier */
		next_usid = get_next_usid(&ip6->daddr, slwt);
		if (next_usid != 0) {
			advance_nextusid(&ip6->daddr, slwt);
			seg6_lookup_nexthop_iface(skb, &slwt->nh6, slwt->oif, 0);
			return dst_input(skb);
		}
	}

	srh = get_and_validate_srh(skb);
	if (!srh)
		goto drop;

	advance_nextseg(srh, &ipv6_hdr(skb)->daddr);

	//seg6_lookup_nexthop(skb, &slwt->nh6, 0);
	seg6_lookup_nexthop_iface(skb, &slwt->nh6, slwt->oif, 0);

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

static int input_action_end_t(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct ipv6_sr_hdr *srh;

	srh = get_and_validate_srh(skb);
	if (!srh)
		goto drop;

	advance_nextseg(srh, &ipv6_hdr(skb)->daddr);

	seg6_lookup_nexthop(skb, NULL, slwt->table);

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

/* decapsulate and forward inner L2 frame on specified interface */
static int input_action_end_dx2(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct net *net = dev_net(skb->dev);
	struct net_device *odev;
	struct ethhdr *eth;

	if (!decap_and_validate(skb, NEXTHDR_NONE))
		goto drop;

	if (!pskb_may_pull(skb, ETH_HLEN))
		goto drop;

	skb_reset_mac_header(skb);
	eth = (struct ethhdr *)skb->data;

	/* To determine the frame's protocol, we assume it is 802.3. This avoids
	 * a call to eth_type_trans(), which is not really relevant for our
	 * use case.
	 */
	if (!eth_proto_is_802_3(eth->h_proto))
		goto drop;

	odev = dev_get_by_index_rcu(net, slwt->oif);
	if (!odev)
		goto drop;

	/* As we accept Ethernet frames, make sure the egress device is of
	 * the correct type.
	 */
	if (odev->type != ARPHRD_ETHER)
		goto drop;

	if (!(odev->flags & IFF_UP) || !netif_carrier_ok(odev))
		goto drop;

	skb_orphan(skb);

	if (skb_warn_if_lro(skb))
		goto drop;

	skb_forward_csum(skb);

	if (skb->len - ETH_HLEN > odev->mtu)
		goto drop;

	skb->dev = odev;
	skb->protocol = eth->h_proto;

	return dev_queue_xmit(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

/* decapsulate and forward to specified nexthop */
static int input_action_end_dx6(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct in6_addr *nhaddr = NULL;
	struct ipv6hdr *ip6;
	u32 next_usid;
	struct net *net;
	struct net_device *dev;

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	if (check_usid_block(&ip6->daddr, slwt)) {
		next_usid = get_next_usid(&ip6->daddr, slwt);
		if (next_usid != 0) {
			goto drop;
		}
	}

	/* this function accepts IPv6 encapsulated packets, with either
	 * an SRH with SL=0, or no SRH.
	 */

	if (!decap_and_validate(skb, IPPROTO_IPV6))
		goto drop;

	if (!pskb_may_pull(skb, sizeof(struct ipv6hdr)))
		goto drop;

	/* The inner packet is not associated to any local interface,
	 * so we do not call netif_rx().
	 *
	 * If slwt->nh6 is set to ::, then lookup the nexthop for the
	 * inner packet's DA. Otherwise, use the specified nexthop.
	 */

	if (slwt->table) {
		net = dev_net(skb_dst(skb)->dev);

		dev = l3mdev_master_by_table(net, slwt->table);
		if (dev) {
			skb->dev = dev;
			IP6CB(skb)->iif = skb->skb_iif = dev->ifindex;
		}

		IP6CB(skb)->flags |= IP6SKB_L3SLAVE;
	}

	if (!ipv6_addr_any(&slwt->nh6))
		nhaddr = &slwt->nh6;

	skb->protocol = htons(ETH_P_IPV6);
	skb_set_transport_header(skb, sizeof(struct ipv6hdr));

	skb_dst_drop(skb);

	seg6_lookup_nexthop_iface(skb, nhaddr, slwt->oif, slwt->table);

	return dst_input(skb);
drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

static int input_action_end_dx4(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	struct iphdr *iph;
	__be32 nhaddr;
	int err;
	u32 next_usid;
#ifdef LEGACY_DT4
	struct net *net;
	struct net_device *dev;
#endif

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	if (check_usid_block(&ip6->daddr, slwt)) {
		next_usid = get_next_usid(&ip6->daddr, slwt);
		if (next_usid != 0) {
			goto drop;
		}
	}

	if (!decap_and_validate(skb, IPPROTO_IPIP))
		goto drop;

	if (!pskb_may_pull(skb, sizeof(struct iphdr)))
		goto drop;

#ifdef LEGACY_DT4
	if (slwt->table) {
		net = dev_net(skb_dst(skb)->dev);

		dev = l3mdev_master_by_table(net, slwt->table);
		if (dev) {
			skb->dev = dev;
			IPCB(skb)->iif = skb->skb_iif = dev->ifindex;
		}

		IPCB(skb)->flags |= IPSKB_L3SLAVE;
	}

	skb->protocol = htons(ETH_P_IP);
	skb_dst_drop(skb);
	skb_set_transport_header(skb, sizeof(struct iphdr));

	iph = ip_hdr(skb);

	nhaddr = slwt->nh4.s_addr ? slwt->nh4.s_addr : iph->daddr;

	err = ip_route_input_table(skb, nhaddr, iph->saddr, 0, skb->dev,
			slwt->table, RTF_KNOWN_NH);
	if (err)
		goto drop;
#else
	skb = end_dt_vrf_core(skb, slwt, AF_INET);
	if (!skb)
		return 0;

	if (IS_ERR(skb)) {
		goto drop;
	}

	iph = ip_hdr(skb);

	nhaddr = slwt->nh4.s_addr ? slwt->nh4.s_addr : iph->daddr;

	err = ip_route_input(skb, nhaddr, iph->saddr, 0, skb->dev);
	if (unlikely(err))
		goto drop;
#endif

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

#if defined(LEGACY_DT4) && defined( LEGACY_DT6)
#else
static struct sk_buff *end_dt_vrf_rcv(struct sk_buff *skb,
								      u16 family,
									  struct net_device *dev)
{
	/* based on l3mdev_ip_rcv; we are only interested in the master */
	if (unlikely(!netif_is_l3_master(dev) && !netif_has_l3_rx_handler(dev))) {
		goto drop;
	}

	if (unlikely(!dev->l3mdev_ops->l3mdev_l3_rcv)) {
		goto drop;
	}

	/* the decap packet IPv4/IPv6 does not come with any mac header info.
	 * We must unset the mac header to allow the VRF device to rebuild it,
	 * just in case there is a sniffer attached on the device.
	 */
	skb_reset_mac_header(skb);

	skb = dev->l3mdev_ops->l3mdev_l3_rcv(dev, skb, family);
	if (!skb)
		/* the skb buffer was consumed by the handler */
		return NULL;

	/* when a packet is received by a VRF or by one of its slaves, the
 	 * master device reference is set into the skb.
	 */
	if (unlikely(skb->dev != dev || skb->skb_iif != dev->ifindex)) {
		goto drop;
	}

	return skb;

drop:
	kfree_skb(skb);
	return NULL;
}

static struct sk_buff *end_dt_vrf_core(struct sk_buff *skb,
                                       struct seg6_local_lwt *slwt,
									   u16 family)
{
	struct net *net = dev_net(skb->dev);
	struct net_device *vrf;

	vrf = l3mdev_master_by_table(net, slwt->table);
	if (! vrf) {
		return NULL;
	}

	if (family == AF_INET) {
		skb->protocol = htons(ETH_P_IP);
		skb_set_transport_header(skb, sizeof(struct iphdr));
	} else if (family == AF_INET6) {
		skb->protocol = htons(ETH_P_IPV6);
		skb_set_transport_header(skb, sizeof(struct ipv6hdr));
	} else {
		goto drop;
	}

	skb_dst_drop(skb);

	nf_reset_ct(skb);

	return end_dt_vrf_rcv (skb, family, vrf);

drop:
	kfree_skb(skb);
	return NULL;
}
#endif

static int input_action_end_dt4(struct sk_buff *skb,
                                struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	struct iphdr *iph;
	int err;
	u32 next_usid;
#ifdef LEGACY_DT4
	struct net *net;
	struct net_device *dev;
#endif

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	if (check_usid_block(&ip6->daddr, slwt)) {
		next_usid = get_next_usid(&ip6->daddr, slwt);
		if (next_usid != 0) {
			goto drop;
		}
	}

	if (!decap_and_validate(skb, IPPROTO_IPIP))
		goto drop;

	if (!pskb_may_pull(skb, sizeof(struct iphdr)))
		goto drop;

#ifdef LEGACY_DT4
	if (slwt->table) {
		net = dev_net(skb_dst(skb)->dev);

		dev = l3mdev_master_by_table(net, slwt->table);
		if (dev) {
			skb->dev = dev;
			IPCB(skb)->iif = skb->skb_iif = dev->ifindex;
		}

		IPCB(skb)->flags |= IPSKB_L3SLAVE;
	}

	skb->protocol = htons(ETH_P_IP);
	skb_dst_drop(skb);
	skb_set_transport_header(skb, sizeof(struct iphdr));

	iph = ip_hdr(skb);

	err = ip_route_input_table(skb, iph->daddr, iph->saddr, 0, skb->dev,
						slwt->table, 0);
	if (unlikely(err)) {
		goto drop;
	}
#else
	skb = end_dt_vrf_core(skb, slwt, AF_INET);
	if (!skb)
		return 0;

	if (IS_ERR(skb)) {
		goto drop;
	}

	iph = ip_hdr(skb);

	err = ip_route_input(skb, iph->daddr, iph->saddr, 0, skb->dev);
	if (unlikely(err)) {
		goto drop;
	}
#endif

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

static int input_action_end_dt6(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	u32 next_usid;
#ifdef LEGACY_DT6
	struct net *net;
	struct net_device *dev;
#endif

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	if (check_usid_block(&ip6->daddr, slwt)) {
		next_usid = get_next_usid(&ip6->daddr, slwt);
		if (next_usid != 0) {
			goto drop;
		}
	}

	if (!decap_and_validate(skb, IPPROTO_IPV6))
		goto drop;

	if (!pskb_may_pull(skb, sizeof(struct ipv6hdr)))
		goto drop;

#ifdef LEGACY_DT6
	if (slwt->table) {
		net = dev_net(skb_dst(skb)->dev);

		dev = l3mdev_master_by_table(net, slwt->table);
		if (dev) {
			skb->dev = dev;
			skb->skb_iif = dev->ifindex;
		}

		IP6CB(skb)->flags |= IP6SKB_L3SLAVE;
	}

	skb_set_transport_header(skb, sizeof(struct ipv6hdr));
	skb->protocol = htons(ETH_P_IPV6);

	skb_dst_drop(skb);

	seg6_lookup_nexthop(skb, NULL, slwt->table);
#else
	skb = end_dt_vrf_core(skb, slwt, AF_INET6);
	if (!skb)
		return 0;

	if (IS_ERR(skb))
		return PTR_ERR(skb);

	seg6_lookup_nexthop(skb, NULL, slwt->table);
#endif

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

/* push an SRH on top of the current one */
static int input_action_end_b6(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct ipv6_sr_hdr *srh;
	int err = -EINVAL;

	srh = get_and_validate_srh(skb);
	if (!srh)
		goto drop;

	err = seg6_do_srh_inline(skb, slwt->srh);
	if (err)
		goto drop;

	ipv6_hdr(skb)->payload_len = htons(skb->len - sizeof(struct ipv6hdr));
	skb_set_transport_header(skb, sizeof(struct ipv6hdr));

	seg6_lookup_nexthop(skb, NULL, 0);

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return err;
}

/* encapsulate within an outer IPv6 header and a specified SRH */
static int input_action_end_b6_encap(struct sk_buff *skb,
				     struct seg6_local_lwt *slwt)
{
	struct ipv6_sr_hdr *srh;
	int err = -EINVAL;

	srh = get_and_validate_srh(skb);
	if (!srh)
		goto drop;

	advance_nextseg(srh, &ipv6_hdr(skb)->daddr);

	skb_reset_inner_headers(skb);
	skb->encapsulation = 1;

	err = seg6_do_srh_encap(skb, slwt->srh, IPPROTO_IPV6);
	if (err)
		goto drop;

	ipv6_hdr(skb)->payload_len = htons(skb->len - sizeof(struct ipv6hdr));
	skb_set_transport_header(skb, sizeof(struct ipv6hdr));

	seg6_lookup_nexthop(skb, NULL, 0);

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return err;
}

DEFINE_PER_CPU(struct seg6_bpf_srh_state, seg6_bpf_srh_states);

bool seg6_bpf_has_valid_srh(struct sk_buff *skb)
{
	struct seg6_bpf_srh_state *srh_state =
		this_cpu_ptr(&seg6_bpf_srh_states);
	struct ipv6_sr_hdr *srh = srh_state->srh;

	if (unlikely(srh == NULL))
		return false;

	if (unlikely(!srh_state->valid)) {
		if ((srh_state->hdrlen & 7) != 0)
			return false;

		srh->hdrlen = (u8)(srh_state->hdrlen >> 3);
		if (!seg6_validate_srh(srh, (srh->hdrlen + 1) << 3))
			return false;

		srh_state->valid = true;
	}

	return true;
}

static int input_action_end_bpf(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct seg6_bpf_srh_state *srh_state =
		this_cpu_ptr(&seg6_bpf_srh_states);
	struct ipv6_sr_hdr *srh;
	int ret;

	srh = get_and_validate_srh(skb);
	if (!srh) {
		if (!seg6_local_deliver(skb)) {
			return 0;
		}

		kfree_skb(skb);
		return -EINVAL;
	}
	advance_nextseg(srh, &ipv6_hdr(skb)->daddr);

	/* preempt_disable is needed to protect the per-CPU buffer srh_state,
	 * which is also accessed by the bpf_lwt_seg6_* helpers
	 */
	preempt_disable();
	srh_state->srh = srh;
	srh_state->hdrlen = srh->hdrlen << 3;
	srh_state->valid = true;

	rcu_read_lock();
	bpf_compute_data_pointers(skb);
	ret = bpf_prog_run_save_cb(slwt->bpf.prog, skb);
	rcu_read_unlock();

	switch (ret) {
	case BPF_OK:
	case BPF_REDIRECT:
		break;
	case BPF_DROP:
		goto drop;
	default:
		pr_warn_once("bpf-seg6local: Illegal return value %u\n", ret);
		goto drop;
	}

	if (srh_state->srh && !seg6_bpf_has_valid_srh(skb))
		goto drop;

	preempt_enable();
	if (ret != BPF_REDIRECT)
		seg6_lookup_nexthop(skb, NULL, 0);

	return dst_input(skb);

drop:
	preempt_enable();
	kfree_skb(skb);
	return -EINVAL;
}

static inline __u8
gtpu_type_get (u16 tag)
{
  u16 val;

  val = ntohs (tag);
  if (val & SRH_TAG_ECHO_REPLY)
    return GTPU_TYPE_ECHO_REPLY;
  else if (val & SRH_TAG_ECHO_REQUEST)
    return GTPU_TYPE_ECHO_REQUEST;
  else if (val & SRH_TAG_ERROR_INDICATION)
    return GTPU_TYPE_ERROR_INDICATION;
  else if (val & SRH_TAG_END_MARKER)
    return GTPU_TYPE_END_MARKER;

  return GTPU_TYPE_GTPU;
}

static int input_action_gtp4_e(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	struct ipv6_sr_hdr *srh;
	__u8 gtpu_type = 0;
	__u32 offset;
	struct in_addr dst4;
	struct in_addr src4;
	__u8 qfi;
	__u32 teid = 0;
	__u16 seq = 0;
	__u32 hdrlen = 0;
	__u32 ie_size = 0;
	__u8 ie_buf[GTPU_IE_MAX_SIZ];
	__u8 *p;
	__u32 plen;
	__u32 len;
	__u8 type = 0;
	struct gtpu_pdu_session_t *sess;
	struct ip4_gtpu_header_t *hdr;
	int err;
	struct net *net;
	struct net_device *dev;

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	srh = get_srh(skb);
	if (srh) {
		gtpu_type = gtpu_type_get(srh->tag);
	}

	offset = slwt->src_pos / 8;
	memcpy(&src4, &ip6->saddr.s6_addr[offset], 4);

	offset = slwt->loc_len / 8;
	memcpy(&dst4, &ip6->daddr.s6_addr[offset], 4);

	qfi = ip6->daddr.s6_addr[offset + 4];

	if (gtpu_type == GTPU_TYPE_ECHO_REQUEST
	 || gtpu_type == GTPU_TYPE_ECHO_REPLY
	 || gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
		memcpy(&seq, &ip6->daddr.s6_addr[offset + 5], 2);
	} else {
		memcpy(&teid, &ip6->daddr.s6_addr[offset + 5], 4);
	}

	hdrlen = sizeof(struct gtpu_exthdr_t) + sizeof(struct gtpu_pdu_session_t);

	if (gtpu_type == GTPU_TYPE_ECHO_REPLY) {
		hdrlen += sizeof(struct gtpu_recovery_ie_t);
	}

	if (gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
		struct ip6_sr_tlv_t *tlv;
		__u8 extlen;

		extlen = srh->hdrlen * 8;

		if (extlen > sizeof(struct in6_addr) * (srh->first_segment + 1)) {
			tlv = (struct ip6_sr_tlv_t *) ((__u8 *)srh +
					sizeof(struct ipv6_sr_hdr) + sizeof(struct in6_addr) * (srh->first_segment + 1));

			if (tlv->type == SRH_TLV_USER_PLANE_CONTAINER) {
				struct user_plane_sub_tlv_t *subtlv;

				subtlv = (struct user_plane_sub_tlv_t *)tlv->value;

				ie_size =subtlv->length;
				memcpy(ie_buf, subtlv->value, ie_size);

				hdrlen += ie_size;
			}
		}
	}

	if (srh) {
		if (!pskb_pull(skb, sizeof(struct ipv6hdr) + sizeof(struct ipv6_sr_hdr)
				+ srh->hdrlen * 8)) {
			goto drop;
		}

		skb_postpull_rcsum(skb, skb_network_header(skb), sizeof(struct ipv6hdr) + sizeof(struct ipv6_sr_hdr)
                + srh->hdrlen * 8);
	} else {
		if (!pskb_pull(skb, sizeof(struct ipv6hdr))) {
			goto drop;
		}

		skb_postpull_rcsum(skb, skb_network_header(skb), sizeof(struct ipv6hdr));
	}

	p = skb->data;
	plen = skb->len;

	len = plen + hdrlen;

	hdrlen += sizeof(struct ip4_gtpu_header_t);

	skb_push(skb, hdrlen);

	hdr = (struct ip4_gtpu_header_t *)skb->data;
	memset(hdr, 0, sizeof(struct ip4_gtpu_header_t));

	hdr->ip4.version = 4;
	hdr->ip4.ihl = 5;

	hdr->ip4.protocol = IPPROTO_UDP;

	hdr->ip4.ttl = 64;

	hdr->udp.source = htons(SRV6_GTP_UDP_DST_PORT);
	hdr->udp.dest = htons(SRV6_GTP_UDP_DST_PORT);

	hdr->gtpu.ver_flags = GTPU_V1_VER | GTPU_PT_GTP;

	memcpy(&hdr->ip4.daddr, &dst4, 4);
	memcpy(&hdr->ip4.saddr, &src4, 4);

	hdr->gtpu.teid = teid;
	hdr->gtpu.length = htons(len);

	hdr->gtpu.type = gtpu_type;

	if (gtpu_type == GTPU_TYPE_ECHO_REPLY
	 || gtpu_type == GTPU_TYPE_ECHO_REQUEST
  	 || gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
		hdr->gtpu.ver_flags |= GTPU_SEQ_FLAG;
		hdr->gtpu.ext->seq = seq;
		hdr->gtpu.ext->npdu_num = 0;
		hdr->gtpu.ext->nextexthdr = 0;

		if (gtpu_type == GTPU_TYPE_ECHO_REPLY) {
			struct gtpu_recovery_ie_t *recovery;

			recovery = (struct gtpu_recovery_ie_t *)((__u8 *)hdr +
				(hdrlen - sizeof(struct gtpu_recovery_ie_t)));

			recovery->type = GTPU_RECOVERY_IE_TYPE;
			recovery->restart_counter = 0;
		} else if (gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
			if (ie_size) {
				__u8 *ie_ptr;

				ie_ptr = (__u8 *)((__u8 *)hdr + (hdrlen - ie_size));
				memcpy(ie_ptr, ie_buf, ie_size);
			}
		}
	} else {
		hdr->gtpu.ext->seq = 0;
		hdr->gtpu.ext->npdu_num = 0;
	}

	hdr->gtpu.ver_flags |= GTPU_EXTHDR_FLAG;

	hdr->gtpu.ext->nextexthdr = GTPU_EXTHDR_PDU_SESSION;

	type = qfi & SRV6_PDU_SESSION_U_BIT_MASK;

	qfi = ((qfi & SRV6_PDU_SESSION_QFI_MASK) >> 2) |
		((qfi & SRV6_PDU_SESSION_R_BIT_MASK) << 5);

	sess = (struct gtpu_pdu_session_t *)(((__u8 *)hdr) +
				sizeof(struct ip4_gtpu_header_t) +
				sizeof(struct gtpu_exthdr_t));

	sess->exthdrlen = 1;
	sess->type = type;
	sess->spare = 0;
	sess->u.val = qfi;
	sess->nextexthdr = 0;

	hdr->udp.len = htons(len + sizeof(struct udphdr) + sizeof(struct gtpu_header_t));

	hdr->ip4.tot_len = htons(len + sizeof(struct ip4_gtpu_header_t));

	hdr->ip4.check = ip_fast_csum(hdr, hdr->ip4.ihl);

	memset(IP6CB(skb), 0, sizeof(*IP6CB(skb)));

	skb_postpush_rcsum(skb, hdr, hdrlen);

	skb_reset_network_header(skb);
    skb_reset_transport_header(skb);
    skb_mac_header_rebuild(skb);

	skb_dst_drop(skb);

	err = ip_route_input_table(skb, hdr->ip4.daddr, hdr->ip4.saddr, 0, skb->dev,
							slwt->table, 0);
	if (err) {
		goto drop;
	}

	if (slwt->table) {
		net = dev_net(skb_dst(skb)->dev);

		dev = l3mdev_master_by_table(net, slwt->table);
		if (dev)
			skb->dev = dev;

		IPCB(skb)->flags |= IPSKB_L3SLAVE;
		IPCB(skb)->iif = l3mdev_master_ifindex_by_table(net, slwt->table);
	}

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

static int input_action_gtp6_e(struct sk_buff *skb,
				struct seg6_local_lwt *slwt)
{
	struct ipv6hdr *ip6;
	struct ipv6_sr_hdr *srh;
	__u8 gtpu_type = 0;
	__u32 offset;
	struct in6_addr dst6;
	struct in6_addr src6;
	__u8 qfi;
	__u32 teid = 0;
	__u16 seq = 0;
	__u32 hdrlen = 0;
	__u32 ie_size = 0;
	__u8 ie_buf[GTPU_IE_MAX_SIZ];
	__u8 *p;
	__u32 plen;
	__u32 len;
	__u8 type = 0;
	struct gtpu_pdu_session_t *sess;
	struct ip6_gtpu_header_t *hdr;
	struct net *net;
	struct net_device *dev;

	ip6 = ipv6_hdr(skb);
	if (!ip6)
		goto drop;

	srh = get_srh(skb);
	if (!srh)
		goto drop;

	if (srh->segments_left != 1)
		goto drop;

	gtpu_type = gtpu_type_get(srh->tag);

	memcpy(&src6, &ip6->saddr, 16);

	memcpy(&dst6, &srh->segments[0], 16);

	offset = slwt->loc_len / 8;

	qfi = ip6->daddr.s6_addr[offset];

	if (gtpu_type == GTPU_TYPE_ECHO_REQUEST
	 || gtpu_type == GTPU_TYPE_ECHO_REPLY
	 || gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
		memcpy(&seq, &ip6->daddr.s6_addr[offset + 1], 2);
	} else {
		memcpy(&teid, &ip6->daddr.s6_addr[offset + 1], 4);
	}

	hdrlen = sizeof(struct gtpu_exthdr_t) + sizeof(struct gtpu_pdu_session_t);

	if (gtpu_type == GTPU_TYPE_ECHO_REPLY) {
		hdrlen += sizeof(struct gtpu_recovery_ie_t);
	}

	if (gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
		struct ip6_sr_tlv_t *tlv;
		__u8 extlen;

		extlen = srh->hdrlen * 8;

		if (extlen > sizeof(struct in6_addr) * (srh->first_segment + 1)) {
			tlv = (struct ip6_sr_tlv_t *) ((__u8 *)srh +
					sizeof(struct ipv6_sr_hdr) + sizeof(struct in6_addr) * (srh->first_segment + 1));

			if (tlv->type == SRH_TLV_USER_PLANE_CONTAINER) {
				struct user_plane_sub_tlv_t *subtlv;

				subtlv = (struct user_plane_sub_tlv_t *)tlv->value;

				ie_size =subtlv->length;
				memcpy(ie_buf, subtlv->value, ie_size);

				hdrlen += ie_size;
			}
		}
	}

	if (!pskb_pull(skb, sizeof(struct ipv6hdr) + sizeof(struct ipv6_sr_hdr)
			+ srh->hdrlen * 8)) {
		goto drop;
	}

	skb_postpull_rcsum(skb, skb_network_header(skb), sizeof(struct ipv6hdr) + sizeof(struct ipv6_sr_hdr)
               + srh->hdrlen * 8);

	p = skb->data;
	plen = skb->len;

	len = plen + hdrlen;

	hdrlen += sizeof(struct ip6_gtpu_header_t);

	skb_push(skb, hdrlen);

	hdr = (struct ip6_gtpu_header_t *)skb->data;
	memset(hdr, 0, sizeof(struct ip6_gtpu_header_t));

	hdr->ip6.version = 6;

	hdr->ip6.nexthdr = IPPROTO_UDP;

	hdr->ip6.hop_limit = 64;

	hdr->ip6.payload_len = htons(len + sizeof(struct udphdr) + sizeof(struct gtpu_header_t));

	hdr->udp.source = htons(SRV6_GTP_UDP_DST_PORT);
	hdr->udp.dest = htons(SRV6_GTP_UDP_DST_PORT);

	hdr->gtpu.ver_flags = GTPU_V1_VER | GTPU_PT_GTP;

	memcpy(&hdr->ip6.daddr, &dst6, 16);
	memcpy(&hdr->ip6.saddr, &src6, 16);

	hdr->gtpu.teid = teid;
	hdr->gtpu.length = htons(len);

	hdr->gtpu.type = gtpu_type;

	if (gtpu_type == GTPU_TYPE_ECHO_REPLY
	 || gtpu_type == GTPU_TYPE_ECHO_REQUEST
  	 || gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
		hdr->gtpu.ver_flags |= GTPU_SEQ_FLAG;
		hdr->gtpu.ext->seq = seq;
		hdr->gtpu.ext->npdu_num = 0;
		hdr->gtpu.ext->nextexthdr = 0;

		if (gtpu_type == GTPU_TYPE_ECHO_REPLY) {
			struct gtpu_recovery_ie_t *recovery;

			recovery = (struct gtpu_recovery_ie_t *)((__u8 *)hdr +
				(hdrlen - sizeof(struct gtpu_recovery_ie_t)));

			recovery->type = GTPU_RECOVERY_IE_TYPE;
			recovery->restart_counter = 0;
		} else if (gtpu_type == GTPU_TYPE_ERROR_INDICATION) {
			if (ie_size) {
				__u8 *ie_ptr;

				ie_ptr = (__u8 *)((__u8 *)hdr + (hdrlen - ie_size));
				memcpy(ie_ptr, ie_buf, ie_size);
			}
		}
	} else {
		hdr->gtpu.ext->seq = 0;
		hdr->gtpu.ext->npdu_num = 0;
	}

	hdr->gtpu.ver_flags |= GTPU_EXTHDR_FLAG;

	hdr->gtpu.ext->nextexthdr = GTPU_EXTHDR_PDU_SESSION;

	type = qfi & SRV6_PDU_SESSION_U_BIT_MASK;

	qfi = ((qfi & SRV6_PDU_SESSION_QFI_MASK) >> 2) |
		((qfi & SRV6_PDU_SESSION_R_BIT_MASK) << 5);

	sess = (struct gtpu_pdu_session_t *)(((__u8 *)hdr) +
				sizeof(struct ip6_gtpu_header_t) +
				sizeof(struct gtpu_exthdr_t));

	sess->exthdrlen = 1;
	sess->type = type;
	sess->spare = 0;
	sess->u.val = qfi;
	sess->nextexthdr = 0;

	hdr->udp.len = htons(len + sizeof(struct udphdr) + sizeof(struct gtpu_header_t));

	memset(IP6CB(skb), 0, sizeof(*IP6CB(skb)));

	skb_postpush_rcsum(skb, hdr, hdrlen);

	skb_reset_network_header(skb);
    skb_reset_transport_header(skb);
    skb_mac_header_rebuild(skb);
    skb->encapsulation = 0;

	skb_dst_drop(skb);

	seg6_lookup_nexthop(skb, NULL, slwt->table);

	if (slwt->table) {
		net = dev_net(skb_dst(skb)->dev);

		dev = l3mdev_master_by_table(net, slwt->table);
		if (dev)
			skb->dev = dev;

		IP6CB(skb)->flags |= IP6SKB_L3SLAVE;
		IP6CB(skb)->iif = l3mdev_master_ifindex_by_table(net, slwt->table);
	}

	skb->skb_iif = skb_dst(skb)->dev->ifindex;

	return dst_input(skb);

drop:
	if (!seg6_local_deliver(skb)) {
		return 0;
	}

	kfree_skb(skb);
	return -EINVAL;
}

static struct seg6_action_desc seg6_action_table[] = {
	{
		.action		= SEG6_LOCAL_ACTION_END,
		.attrs		= 0,
		.input		= input_action_end,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_UN,
		.attrs		= (1 << SEG6_LOCAL_USID),
		.input		= input_action_end,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_X,
		.attrs		= (1 << SEG6_LOCAL_NH6) | (1 << SEG6_LOCAL_OIF),
		.input		= input_action_end_x,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_UA,
		.attrs		= (1 << SEG6_LOCAL_USID) | (1 << SEG6_LOCAL_NH6) | (1 << SEG6_LOCAL_OIF),
		.input		= input_action_end_x,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_T,
		.attrs		= (1 << SEG6_LOCAL_TABLE),
		.input		= input_action_end_t,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_DX2,
		.attrs		= (1 << SEG6_LOCAL_OIF),
		.input		= input_action_end_dx2,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_DX6,
		.attrs		= (1 << SEG6_LOCAL_NH6) | (1 << SEG6_LOCAL_TABLE) | (1 << SEG6_LOCAL_OIF),
		.input		= input_action_end_dx6,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_UDX6,
		.attrs		= (1 << SEG6_LOCAL_NH6) | (1 << SEG6_LOCAL_USID) | (1 << SEG6_LOCAL_TABLE) | (1 << SEG6_LOCAL_OIF),
		.input		= input_action_end_dx6,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_DX4,
		.attrs		= (1 << SEG6_LOCAL_NH4) | (1 << SEG6_LOCAL_TABLE),
		.input		= input_action_end_dx4,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_UDX4,
		.attrs		= (1 << SEG6_LOCAL_NH4) | (1 << SEG6_LOCAL_USID) | (1 << SEG6_LOCAL_TABLE),
		.input		= input_action_end_dx4,
	},
	{
		.action 	= SEG6_LOCAL_ACTION_END_DT4,
		.attrs		= (1 << SEG6_LOCAL_TABLE),
		.input		= input_action_end_dt4,
	},
	{
		.action 	= SEG6_LOCAL_ACTION_END_UDT4,
		.attrs		= (1 << SEG6_LOCAL_TABLE) | (1 << SEG6_LOCAL_USID),
		.input 		= input_action_end_dt4,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_DT6,
		.attrs		= (1 << SEG6_LOCAL_TABLE),
		.input		= input_action_end_dt6,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_UDT6,
		.attrs	 	= (1 << SEG6_LOCAL_TABLE) | (1 << SEG6_LOCAL_USID),
		.input		= input_action_end_dt6,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_B6,
		.attrs		= (1 << SEG6_LOCAL_SRH),
		.input		= input_action_end_b6,
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_B6_ENCAP,
		.attrs		= (1 << SEG6_LOCAL_SRH),
		.input		= input_action_end_b6_encap,
		.static_headroom	= sizeof(struct ipv6hdr),
	},
	{
		.action		= SEG6_LOCAL_ACTION_END_BPF,
		.attrs		= (1 << SEG6_LOCAL_BPF),
		.input		= input_action_end_bpf,
	},
	{
		.action 	= SEG6_LOCAL_ACTION_GTP4_E,
		.attrs		= (1 << SEG6_LOCAL_TABLE) | (1 << SEG6_LOCAL_LOC_LEN) | (1 << SEG6_LOCAL_SRC_POS),
		.input		= input_action_gtp4_e,
	},
	{
		.action 	= SEG6_LOCAL_ACTION_GTP6_E,
		.attrs		= (1 << SEG6_LOCAL_TABLE) | (1 << SEG6_LOCAL_LOC_LEN),
		.input		= input_action_gtp6_e,
	},
};

static struct seg6_action_desc *__get_action_desc(int action)
{
	struct seg6_action_desc *desc;
	int i, count;

	count = ARRAY_SIZE(seg6_action_table);
	for (i = 0; i < count; i++) {
		desc = &seg6_action_table[i];
		if (desc->action == action)
			return desc;
	}

	return NULL;
}

static int seg6_local_input(struct sk_buff *skb)
{
	struct dst_entry *orig_dst = skb_dst(skb);
	struct seg6_action_desc *desc;
	struct seg6_local_lwt *slwt;

	if (skb->protocol != htons(ETH_P_IPV6)) {
		kfree_skb(skb);
		return -EINVAL;
	}

	slwt = seg6_local_lwtunnel(orig_dst->lwtstate);
	desc = slwt->desc;

	return desc->input(skb, slwt);
}

static const struct nla_policy seg6_local_policy[SEG6_LOCAL_MAX + 1] = {
	[SEG6_LOCAL_ACTION]	= { .type = NLA_U32 },
	[SEG6_LOCAL_SRH]	= { .type = NLA_BINARY },
	[SEG6_LOCAL_TABLE]	= { .type = NLA_U32 },
	[SEG6_LOCAL_NH4]	= { .type = NLA_BINARY,
				    .len = sizeof(struct in_addr) },
	[SEG6_LOCAL_NH6]	= { .type = NLA_BINARY,
				    .len = sizeof(struct in6_addr) },
	[SEG6_LOCAL_IIF]	= { .type = NLA_U32 },
	[SEG6_LOCAL_OIF]	= { .type = NLA_U32 },
	[SEG6_LOCAL_USID]	= { .type = NLA_BINARY,
				    .len = sizeof(struct sr6_usid_info) },
	[SEG6_LOCAL_BPF]	= { .type = NLA_NESTED },
	[SEG6_LOCAL_LOC_LEN]	= { .type = NLA_U32 },
	[SEG6_LOCAL_SRC_POS]	= { .type = NLA_U32 },
};

static int parse_nla_usid(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	struct sr6_usid_info *usid;
	int len;

	usid = nla_data(attrs[SEG6_LOCAL_USID]);
	len = nla_len(attrs[SEG6_LOCAL_USID]);

	if (len != sizeof(struct sr6_usid_info)) {
		pr_info("Invalid attribute length (%d)\n", len);
		return -EINVAL;
	}

	if ((usid->usid_block_len % 8) != 0) {
		pr_info("Invalid usid block length (%d)\n", usid->usid_block_len);
		return -EINVAL;
	}

	if (usid->usid_len != 16 && usid->usid_len != 32 && usid->usid_len != 48) {
		pr_info("Invalid usid length (%d)\n", usid->usid_len);
		return -EINVAL;
	}

	memcpy(&slwt->usid_info, usid, sizeof(struct sr6_usid_info));

	return 0;
}

static int put_nla_usid(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct nlattr *nla;

	nla = nla_reserve(skb, SEG6_LOCAL_USID, sizeof(struct sr6_usid_info));
	if (!nla)
		return -EMSGSIZE;

	memcpy(nla_data(nla), &slwt->usid_info, sizeof(struct sr6_usid_info));
	return 0;
}

static int cmp_nla_usid(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	return memcmp(&a->usid_info, &b->usid_info, sizeof(struct sr6_usid_info));
}

static int parse_nla_srh(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	struct ipv6_sr_hdr *srh;
	int len;

	srh = nla_data(attrs[SEG6_LOCAL_SRH]);
	len = nla_len(attrs[SEG6_LOCAL_SRH]);

	/* SRH must contain at least one segment */
	if (len < sizeof(*srh) + sizeof(struct in6_addr))
		return -EINVAL;

	if (!seg6_validate_srh(srh, len))
		return -EINVAL;

	slwt->srh = kmemdup(srh, len, GFP_KERNEL);
	if (!slwt->srh)
		return -ENOMEM;

	slwt->headroom += len;

	return 0;
}

static int put_nla_srh(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct ipv6_sr_hdr *srh;
	struct nlattr *nla;
	int len;

	srh = slwt->srh;
	len = (srh->hdrlen + 1) << 3;

	nla = nla_reserve(skb, SEG6_LOCAL_SRH, len);
	if (!nla)
		return -EMSGSIZE;

	memcpy(nla_data(nla), srh, len);

	return 0;
}

static int cmp_nla_srh(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	int len = (a->srh->hdrlen + 1) << 3;

	if (len != ((b->srh->hdrlen + 1) << 3))
		return 1;

	return memcmp(a->srh, b->srh, len);
}

static int parse_nla_table(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	slwt->table = nla_get_u32(attrs[SEG6_LOCAL_TABLE]);

	return 0;
}

static int put_nla_table(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	if (nla_put_u32(skb, SEG6_LOCAL_TABLE, slwt->table))
		return -EMSGSIZE;

	return 0;
}

static int cmp_nla_table(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	if (a->table != b->table)
		return 1;

	return 0;
}

static int parse_nla_nh4(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	memcpy(&slwt->nh4, nla_data(attrs[SEG6_LOCAL_NH4]),
	       sizeof(struct in_addr));

	return 0;
}

static int put_nla_nh4(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct nlattr *nla;

	nla = nla_reserve(skb, SEG6_LOCAL_NH4, sizeof(struct in_addr));
	if (!nla)
		return -EMSGSIZE;

	memcpy(nla_data(nla), &slwt->nh4, sizeof(struct in_addr));

	return 0;
}

static int cmp_nla_nh4(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	return memcmp(&a->nh4, &b->nh4, sizeof(struct in_addr));
}

static int parse_nla_nh6(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	memcpy(&slwt->nh6, nla_data(attrs[SEG6_LOCAL_NH6]),
	       sizeof(struct in6_addr));

	return 0;
}

static int put_nla_nh6(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct nlattr *nla;

	nla = nla_reserve(skb, SEG6_LOCAL_NH6, sizeof(struct in6_addr));
	if (!nla)
		return -EMSGSIZE;

	memcpy(nla_data(nla), &slwt->nh6, sizeof(struct in6_addr));

	return 0;
}

static int cmp_nla_nh6(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	return memcmp(&a->nh6, &b->nh6, sizeof(struct in6_addr));
}

static int parse_nla_iif(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	slwt->iif = nla_get_u32(attrs[SEG6_LOCAL_IIF]);

	return 0;
}

static int put_nla_iif(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	if (nla_put_u32(skb, SEG6_LOCAL_IIF, slwt->iif))
		return -EMSGSIZE;

	return 0;
}

static int cmp_nla_iif(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	if (a->iif != b->iif)
		return 1;

	return 0;
}

static int parse_nla_oif(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	slwt->oif = nla_get_u32(attrs[SEG6_LOCAL_OIF]);

	return 0;
}

static int put_nla_oif(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	if (nla_put_u32(skb, SEG6_LOCAL_OIF, slwt->oif))
		return -EMSGSIZE;

	return 0;
}

static int cmp_nla_oif(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	if (a->oif != b->oif)
		return 1;

	return 0;
}

static int parse_nla_src_pos(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
    slwt->src_pos = nla_get_u32(attrs[SEG6_LOCAL_SRC_POS]);

    return 0;
}

static int put_nla_src_pos(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
    if (nla_put_u32(skb, SEG6_LOCAL_SRC_POS, slwt->src_pos))
        return -EMSGSIZE;

    return 0;
}

static int cmp_nla_src_pos(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
    if (a->src_pos != b->src_pos)
        return 1;

    return 0;
}

static int parse_nla_loc_len(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
    slwt->loc_len = nla_get_u32(attrs[SEG6_LOCAL_LOC_LEN]);

    return 0;
}

static int put_nla_loc_len(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
    if (nla_put_u32(skb, SEG6_LOCAL_LOC_LEN, slwt->loc_len))
        return -EMSGSIZE;

    return 0;
}

static int cmp_nla_loc_len(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
    if (a->loc_len != b->loc_len)
        return 1;

    return 0;
}

#define MAX_PROG_NAME 256
static const struct nla_policy bpf_prog_policy[SEG6_LOCAL_BPF_PROG_MAX + 1] = {
	[SEG6_LOCAL_BPF_PROG]	   = { .type = NLA_U32, },
	[SEG6_LOCAL_BPF_PROG_NAME] = { .type = NLA_NUL_STRING,
				       .len = MAX_PROG_NAME },
};

static int parse_nla_bpf(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	struct nlattr *tb[SEG6_LOCAL_BPF_PROG_MAX + 1];
	struct bpf_prog *p;
	int ret;
	u32 fd;

	ret = nla_parse_nested_deprecated(tb, SEG6_LOCAL_BPF_PROG_MAX,
					  attrs[SEG6_LOCAL_BPF],
					  bpf_prog_policy, NULL);
	if (ret < 0)
		return ret;

	if (!tb[SEG6_LOCAL_BPF_PROG] || !tb[SEG6_LOCAL_BPF_PROG_NAME])
		return -EINVAL;

	slwt->bpf.name = nla_memdup(tb[SEG6_LOCAL_BPF_PROG_NAME], GFP_KERNEL);
	if (!slwt->bpf.name)
		return -ENOMEM;

	fd = nla_get_u32(tb[SEG6_LOCAL_BPF_PROG]);
	p = bpf_prog_get_type(fd, BPF_PROG_TYPE_LWT_SEG6LOCAL);
	if (IS_ERR(p)) {
		kfree(slwt->bpf.name);
		return PTR_ERR(p);
	}

	slwt->bpf.prog = p;
	return 0;
}

static int put_nla_bpf(struct sk_buff *skb, struct seg6_local_lwt *slwt)
{
	struct nlattr *nest;

	if (!slwt->bpf.prog)
		return 0;

	nest = nla_nest_start_noflag(skb, SEG6_LOCAL_BPF);
	if (!nest)
		return -EMSGSIZE;

	if (nla_put_u32(skb, SEG6_LOCAL_BPF_PROG, slwt->bpf.prog->aux->id))
		return -EMSGSIZE;

	if (slwt->bpf.name &&
	    nla_put_string(skb, SEG6_LOCAL_BPF_PROG_NAME, slwt->bpf.name))
		return -EMSGSIZE;

	return nla_nest_end(skb, nest);
}

static int cmp_nla_bpf(struct seg6_local_lwt *a, struct seg6_local_lwt *b)
{
	if (!a->bpf.name && !b->bpf.name)
		return 0;

	if (!a->bpf.name || !b->bpf.name)
		return 1;

	return strcmp(a->bpf.name, b->bpf.name);
}

struct seg6_action_param {
	int (*parse)(struct nlattr **attrs, struct seg6_local_lwt *slwt);
	int (*put)(struct sk_buff *skb, struct seg6_local_lwt *slwt);
	int (*cmp)(struct seg6_local_lwt *a, struct seg6_local_lwt *b);
};

static struct seg6_action_param seg6_action_params[SEG6_LOCAL_MAX + 1] = {
	[SEG6_LOCAL_SRH]	= { .parse = parse_nla_srh,
				    .put = put_nla_srh,
				    .cmp = cmp_nla_srh },

	[SEG6_LOCAL_TABLE]	= { .parse = parse_nla_table,
				    .put = put_nla_table,
				    .cmp = cmp_nla_table },

	[SEG6_LOCAL_NH4]	= { .parse = parse_nla_nh4,
				    .put = put_nla_nh4,
				    .cmp = cmp_nla_nh4 },

	[SEG6_LOCAL_NH6]	= { .parse = parse_nla_nh6,
				    .put = put_nla_nh6,
				    .cmp = cmp_nla_nh6 },

	[SEG6_LOCAL_IIF]	= { .parse = parse_nla_iif,
				    .put = put_nla_iif,
				    .cmp = cmp_nla_iif },

	[SEG6_LOCAL_OIF]	= { .parse = parse_nla_oif,
				    .put = put_nla_oif,
				    .cmp = cmp_nla_oif },

	[SEG6_LOCAL_USID]	= { .parse = parse_nla_usid,
				    .put = put_nla_usid,
				    .cmp = cmp_nla_usid },

	[SEG6_LOCAL_BPF]	= { .parse = parse_nla_bpf,
				    .put = put_nla_bpf,
				    .cmp = cmp_nla_bpf },

	[SEG6_LOCAL_LOC_LEN] 	= { .parse = parse_nla_loc_len,
					.put = put_nla_loc_len,
					.cmp = cmp_nla_loc_len },

	[SEG6_LOCAL_SRC_POS] 	= { .parse = parse_nla_src_pos,
					.put = put_nla_src_pos,
					.cmp = cmp_nla_src_pos },
};


static int parse_nla_action(struct nlattr **attrs, struct seg6_local_lwt *slwt)
{
	struct seg6_action_param *param;
	struct seg6_action_desc *desc;
	int i, err;

	desc = __get_action_desc(slwt->action);
	if (!desc)
		return -EINVAL;

	if (!desc->input)
		return -EOPNOTSUPP;

	slwt->desc = desc;
	slwt->headroom += desc->static_headroom;

	for (i = 0; i < SEG6_LOCAL_MAX + 1; i++) {
		if (desc->attrs & (1 << i)) {
			if (!attrs[i])
				return -EINVAL;

			param = &seg6_action_params[i];

			err = param->parse(attrs, slwt);
			if (err < 0)
				return err;
		}
	}

	return 0;
}

static int seg6_local_build_state(struct nlattr *nla, unsigned int family,
				  const void *cfg, struct lwtunnel_state **ts,
				  struct netlink_ext_ack *extack)
{
	struct nlattr *tb[SEG6_LOCAL_MAX + 1];
	struct lwtunnel_state *newts;
	struct seg6_local_lwt *slwt;
	int err;

	if (family != AF_INET6)
		return -EINVAL;

	err = nla_parse_nested_deprecated(tb, SEG6_LOCAL_MAX, nla,
					  seg6_local_policy, extack);

	if (err < 0)
		return err;

	if (!tb[SEG6_LOCAL_ACTION])
		return -EINVAL;

	newts = lwtunnel_state_alloc(sizeof(*slwt));
	if (!newts)
		return -ENOMEM;

	slwt = seg6_local_lwtunnel(newts);
	slwt->action = nla_get_u32(tb[SEG6_LOCAL_ACTION]);

	err = parse_nla_action(tb, slwt);
	if (err < 0)
		goto out_free;

	newts->type = LWTUNNEL_ENCAP_SEG6_LOCAL;
	newts->flags = LWTUNNEL_STATE_INPUT_REDIRECT;
	newts->headroom = slwt->headroom;

	*ts = newts;

	return 0;

out_free:
	kfree(slwt->srh);
	kfree(newts);
	return err;
}

static void seg6_local_destroy_state(struct lwtunnel_state *lwt)
{
	struct seg6_local_lwt *slwt = seg6_local_lwtunnel(lwt);

	kfree(slwt->srh);

	if (slwt->desc->attrs & (1 << SEG6_LOCAL_BPF)) {
		kfree(slwt->bpf.name);
		bpf_prog_put(slwt->bpf.prog);
	}

	return;
}

static int seg6_local_fill_encap(struct sk_buff *skb,
				 struct lwtunnel_state *lwt)
{
	struct seg6_local_lwt *slwt = seg6_local_lwtunnel(lwt);
	struct seg6_action_param *param;
	int i, err;

	if (nla_put_u32(skb, SEG6_LOCAL_ACTION, slwt->action))
		return -EMSGSIZE;

	for (i = 0; i < SEG6_LOCAL_MAX + 1; i++) {
		if (slwt->desc->attrs & (1 << i)) {
			param = &seg6_action_params[i];
			err = param->put(skb, slwt);
			if (err < 0)
				return err;
		}
	}

	return 0;
}

static int seg6_local_get_encap_size(struct lwtunnel_state *lwt)
{
	struct seg6_local_lwt *slwt = seg6_local_lwtunnel(lwt);
	unsigned long attrs;
	int nlsize;

	nlsize = nla_total_size(4); /* action */

	attrs = slwt->desc->attrs;

	if (attrs & (1 << SEG6_LOCAL_SRH))
		nlsize += nla_total_size((slwt->srh->hdrlen + 1) << 3);

	if (attrs & (1 << SEG6_LOCAL_TABLE))
		nlsize += nla_total_size(4);

	if (attrs & (1 << SEG6_LOCAL_NH4))
		nlsize += nla_total_size(4);

	if (attrs & (1 << SEG6_LOCAL_NH6))
		nlsize += nla_total_size(16);

	if (attrs & (1 << SEG6_LOCAL_IIF))
		nlsize += nla_total_size(4);

	if (attrs & (1 << SEG6_LOCAL_OIF))
		nlsize += nla_total_size(4);

	if (attrs & (1 << SEG6_LOCAL_USID))
		nlsize += nla_total_size(sizeof(struct sr6_usid_info));

	if (attrs & (1 << SEG6_LOCAL_BPF))
		nlsize += nla_total_size(sizeof(struct nlattr)) +
		       nla_total_size(MAX_PROG_NAME) +
		       nla_total_size(4);

	return nlsize;
}

static int seg6_local_cmp_encap(struct lwtunnel_state *a,
				struct lwtunnel_state *b)
{
	struct seg6_local_lwt *slwt_a, *slwt_b;
	struct seg6_action_param *param;
	int i;

	slwt_a = seg6_local_lwtunnel(a);
	slwt_b = seg6_local_lwtunnel(b);

	if (slwt_a->action != slwt_b->action)
		return 1;

	if (slwt_a->desc->attrs != slwt_b->desc->attrs)
		return 1;

	for (i = 0; i < SEG6_LOCAL_MAX + 1; i++) {
		if (slwt_a->desc->attrs & (1 << i)) {
			param = &seg6_action_params[i];
			if (param->cmp(slwt_a, slwt_b))
				return 1;
		}
	}

	return 0;
}

static const struct lwtunnel_encap_ops seg6_local_ops = {
	.build_state	= seg6_local_build_state,
	.destroy_state	= seg6_local_destroy_state,
	.input		= seg6_local_input,
	.fill_encap	= seg6_local_fill_encap,
	.get_encap_size	= seg6_local_get_encap_size,
	.cmp_encap	= seg6_local_cmp_encap,
	.owner		= THIS_MODULE,
};

int __init seg6_local_init(void)
{
	return lwtunnel_encap_add_ops(&seg6_local_ops,
				      LWTUNNEL_ENCAP_SEG6_LOCAL);
}

void seg6_local_exit(void)
{
	lwtunnel_encap_del_ops(&seg6_local_ops, LWTUNNEL_ENCAP_SEG6_LOCAL);
}
