static struct sk_buff *vxlan_na_create(struct sk_buff *request,
	struct neighbour *n, bool isrouter)
{
	struct net_device *dev = request->dev;
	struct sk_buff *reply;
	struct nd_msg *ns, *na;
	struct ipv6hdr *pip6;
	u8 *daddr;
	int na_olen = 8; /* opt hdr + ETH_ALEN for target */
	int ns_olen;
	int i, len;

	if (dev == NULL)
		return NULL;

	len = LL_RESERVED_SPACE(dev) + sizeof(struct ipv6hdr) +
		sizeof(*na) + na_olen + dev->needed_tailroom;
	reply = alloc_skb(len, GFP_ATOMIC);
	if (reply == NULL)
		return NULL;

	reply->protocol = htons(ETH_P_IPV6);
	reply->dev = dev;
	skb_reserve(reply, LL_RESERVED_SPACE(request->dev));
	skb_push(reply, sizeof(struct ethhdr));
	skb_set_mac_header(reply, 0);

	ns = (struct nd_msg *)skb_transport_header(request);

	daddr = eth_hdr(request)->h_source;
	ns_olen = request->len - skb_transport_offset(request) - sizeof(*ns);
	for (i = 0; i < ns_olen-1; i += (ns->opt[i+1]<<3)) {
		if (ns->opt[i] == ND_OPT_SOURCE_LL_ADDR) {
			daddr = ns->opt + i + sizeof(struct nd_opt_hdr);
			break;
		}
	}

	/* Ethernet header */
	ether_addr_copy(eth_hdr(reply)->h_dest, daddr);
	ether_addr_copy(eth_hdr(reply)->h_source, n->ha);
	eth_hdr(reply)->h_proto = htons(ETH_P_IPV6);
	reply->protocol = htons(ETH_P_IPV6);

	skb_pull(reply, sizeof(struct ethhdr));
	skb_set_network_header(reply, 0);
	skb_put(reply, sizeof(struct ipv6hdr));

	/* IPv6 header */

	pip6 = ipv6_hdr(reply);
	memset(pip6, 0, sizeof(struct ipv6hdr));
	pip6->version = 6;
	pip6->priority = ipv6_hdr(request)->priority;
	pip6->nexthdr = IPPROTO_ICMPV6;
	pip6->hop_limit = 255;
	pip6->daddr = ipv6_hdr(request)->saddr;
	pip6->saddr = *(struct in6_addr *)n->primary_key;

	skb_pull(reply, sizeof(struct ipv6hdr));
	skb_set_transport_header(reply, 0);

	na = (struct nd_msg *)skb_put(reply, sizeof(*na) + na_olen);

	/* Neighbor Advertisement */
	memset(na, 0, sizeof(*na)+na_olen);
	na->icmph.icmp6_type = NDISC_NEIGHBOUR_ADVERTISEMENT;
	na->icmph.icmp6_router = isrouter;
	na->icmph.icmp6_override = 1;
	na->icmph.icmp6_solicited = 1;
	na->target = ns->target;
	ether_addr_copy(&na->opt[2], n->ha);
	na->opt[0] = ND_OPT_TARGET_LL_ADDR;
	na->opt[1] = na_olen >> 3;

	na->icmph.icmp6_cksum = csum_ipv6_magic(&pip6->saddr,
		&pip6->daddr, sizeof(*na)+na_olen, IPPROTO_ICMPV6,
		csum_partial(na, sizeof(*na)+na_olen, 0));

	pip6->payload_len = htons(sizeof(*na)+na_olen);

	skb_push(reply, sizeof(struct ipv6hdr));

	reply->ip_summed = CHECKSUM_UNNECESSARY;

	return reply;
}