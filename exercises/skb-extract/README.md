
# transxmit


## tc control of dev_xmit

```
/* This returns the tstamp value set by TCP in terms of the set clock. */
static ktime_t get_tcp_tstamp(struct taprio_sched *q, struct sk_buff *skb)
{
	unsigned int offset = skb_network_offset(skb);
	const struct ipv6hdr *ipv6h;
	const struct iphdr *iph;
	struct ipv6hdr _ipv6h;

	ipv6h = skb_header_pointer(skb, offset, sizeof(_ipv6h), &_ipv6h);
	if (!ipv6h)
		return 0;

	if (ipv6h->version == 4) {
		iph = (struct iphdr *)ipv6h;
		offset += iph->ihl * 4;

		/* special-case 6in4 tunnelling, as that is a common way to get
		 * v6 connectivity in the home
		 */
		if (iph->protocol == IPPROTO_IPV6) {
			ipv6h = skb_header_pointer(skb, offset,
						   sizeof(_ipv6h), &_ipv6h);

			if (!ipv6h || ipv6h->nexthdr != IPPROTO_TCP)
				return 0;
		} else if (iph->protocol != IPPROTO_TCP) {
			return 0;
		}
	} else if (ipv6h->version == 6 && ipv6h->nexthdr != IPPROTO_TCP) {
		return 0;
	}

	return taprio_mono_to_any(q, skb->skb_mstamp_ns);
}

```