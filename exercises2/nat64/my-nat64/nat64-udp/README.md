


# run (大包)

```
[root@centos7 nat64_icmp_frag]# insmod  nat64_device.ko 
[root@centos7 nat64_icmp_frag]# ip a add 2001:db8::a0a:6751/96 dev nat64
[root@centos7 nat64_icmp_frag]# ip l set nat64 up
[root@centos7 nat64_icmp_frag]# ./udp_cli 
waiting for a reply...
got 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' from 2001:db8::a0a:6752
[root@centos7 nat64_icmp_frag]# 
```

> ## fullnat checksum

```
        udphdr = udp_hdr(skb);
        //old five tuple <dest,  udphdr->dest, udp  -->  src, udphdr->source>
        // new five tuple  <src ,  udphdr->source, udp  --> dest,  udphdr->dest>
        //snat
        port = udphdr->dest;
        udphdr->dest = udphdr->source;
        udp_fast_csum_update(udphdr,(__be32 *)dest,(__be32 *)src,udphdr->dest, udphdr->source);
        //dnat
        udphdr->source = port;
        udp_fast_csum_update(udphdr,(__be32 *)src,(__be32 *)dest, udphdr->source,port);
```
> ## csum_inv_substract  csum_inv_add
```
static inline void factory_clone_udp(struct sk_buff *src, struct sk_buff *dst, __be16 sport, __be16 dport)
{
	struct udphdr	*udph;
	int		len;

	len = sizeof(struct udphdr);
	udph = (struct udphdr*)skb_push(dst, len);
	//memcpy(udph, skb_transport_header(src), len);
	memcpy(udph, skb_push(src, len), len);
	//printk("nat64: [factory] [udp] [debug] src_udph = %02x %02x %02x %02x %02x %02x %02x %02x.\n", *(src->data), *(src->data +1), *(src->data +2), *(src->data +3), *(src->data +4), *(src->data +5), *(src->data +6), *(src->data +7));
	udph->source = sport;
	udph->dest = dport;
	csum_inv_substract(&udph->check, (__be16 *)src->data, (__be16 *)(src->data + 4));
	csum_inv_add(&udph->check, (__be16 *)dst->data, (__be16 *)(dst->data + 4));
	//printk("nat64: [factory] [udp] [debug] dst_udph = %02x %02x %02x %02x %02x %02x %02x %02x.\n", *(dst->data), *(dst->data +1), *(dst->data +2), *(dst->data +3), *(dst->data +4), *(dst->data +5), *(dst->data +6), *(dst->data +7));

	skb_reset_transport_header(dst);
}
```
# run 小包

```
[root@centos7 nat64_icmp_frag]# ./udp_cli 
waiting for a reply...
got 'hi there' from 2001:db8::a0a:6752
[root@centos7 nat64_icmp_frag]# 
```

```
#if 1
  /* now send a datagram */
  if (sendto(sock, MESSAGE, sizeof(MESSAGE), 0,
             (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
      perror("sendto failed");
      exit(4);
  }
#else
  int i = 0;
  while(i< UDP_BIG_PKT_LEN){

     buffer[i] = 'a';
     ++ i;
  }
  if (sendto(sock, buffer, sizeof(buffer), 0,
             (struct sockaddr *)&server_addr,
             sizeof(server_addr)) < 0) {
      perror("sendto failed");
      exit(4);
  }
  memset(buffer,0, sizeof(buffer));
#endif
```


#  csum_tcpudp_magiccsum_tcpudp_magic

```
static void esp_output_encap_csum(struct sk_buff *skb)
{
	/* UDP encap with IPv6 requires a valid checksum */
	if (*skb_mac_header(skb) == IPPROTO_UDP) {
		struct udphdr *uh = udp_hdr(skb);
		struct ipv6hdr *ip6h = ipv6_hdr(skb);
		int len = ntohs(uh->len);
		unsigned int offset = skb_transport_offset(skb);
		__wsum csum = skb_checksum(skb, offset, skb->len - offset, 0);

		uh->check = csum_ipv6_magic(&ip6h->saddr, &ip6h->daddr,
					    len, IPPROTO_UDP, csum);
		if (uh->check == 0)
			uh->check = CSUM_MANGLED_0;
	}
}
```


```
static int
udp_snat_handler(struct sk_buff *skb,
		 struct ip_vs_protocol *pp, struct ip_vs_conn *cp)
{
	struct udphdr *udph;
	unsigned int udphoff;
	int oldlen;

#ifdef CONFIG_IP_VS_IPV6
	if (cp->af == AF_INET6)
		udphoff = sizeof(struct ipv6hdr);
	else
#endif
		udphoff = ip_hdrlen(skb);
	oldlen = skb->len - udphoff;

	/* csum_check requires unshared skb */
	if (!skb_make_writable(skb, udphoff+sizeof(*udph)))
		return 0;

	if (unlikely(cp->app != NULL)) {
		/* Some checks before mangling */
		if (pp->csum_check && !pp->csum_check(cp->af, skb, pp))
			return 0;

		/*
		 *	Call application helper if needed
		 */
		if (!ip_vs_app_pkt_out(cp, skb))
			return 0;
	}

	udph = (void *)skb_network_header(skb) + udphoff;
	udph->source = cp->vport;

	/*
	 *	Adjust UDP checksums
	 */
	if (skb->ip_summed == CHECKSUM_PARTIAL) {
		udp_partial_csum_update(cp->af, udph, &cp->daddr, &cp->vaddr,
					htons(oldlen),
					htons(skb->len - udphoff));
	} else if (!cp->app && (udph->check != 0)) {
		/* Only port and addr are changed, do fast csum update */
		udp_fast_csum_update(cp->af, udph, &cp->daddr, &cp->vaddr,
				     cp->dport, cp->vport);
		if (skb->ip_summed == CHECKSUM_COMPLETE)
			skb->ip_summed = CHECKSUM_NONE;
	} else {
		/* full checksum calculation */
		udph->check = 0;
		skb->csum = skb_checksum(skb, udphoff, skb->len - udphoff, 0);
#ifdef CONFIG_IP_VS_IPV6
		if (cp->af == AF_INET6)
			udph->check = csum_ipv6_magic(&cp->vaddr.in6,
						      &cp->caddr.in6,
						      skb->len - udphoff,
						      cp->protocol, skb->csum);
		else
#endif
			udph->check = csum_tcpudp_magic(cp->vaddr.ip,
							cp->caddr.ip,
							skb->len - udphoff,
							cp->protocol,
							skb->csum);
		if (udph->check == 0)
			udph->check = CSUM_MANGLED_0;
		IP_VS_DBG(11, "O-pkt: %s O-csum=%d (+%zd)\n",
			  pp->name, udph->check,
			  (char*)&(udph->check) - (char*)udph);
	}
	return 1;
}


static int
udp_dnat_handler(struct sk_buff *skb,
		 struct ip_vs_protocol *pp, struct ip_vs_conn *cp)
{
	struct udphdr *udph;
	unsigned int udphoff;
	int oldlen;

#ifdef CONFIG_IP_VS_IPV6
	if (cp->af == AF_INET6)
		udphoff = sizeof(struct ipv6hdr);
	else
#endif
		udphoff = ip_hdrlen(skb);
	oldlen = skb->len - udphoff;

	/* csum_check requires unshared skb */
	if (!skb_make_writable(skb, udphoff+sizeof(*udph)))
		return 0;

	if (unlikely(cp->app != NULL)) {
		/* Some checks before mangling */
		if (pp->csum_check && !pp->csum_check(cp->af, skb, pp))
			return 0;

		/*
		 *	Attempt ip_vs_app call.
		 *	It will fix ip_vs_conn
		 */
		if (!ip_vs_app_pkt_in(cp, skb))
			return 0;
	}

	udph = (void *)skb_network_header(skb) + udphoff;
	udph->dest = cp->dport;

	/*
	 *	Adjust UDP checksums
	 */
	if (skb->ip_summed == CHECKSUM_PARTIAL) {
		udp_partial_csum_update(cp->af, udph, &cp->vaddr, &cp->daddr,
					htons(oldlen),
					htons(skb->len - udphoff));
	} else if (!cp->app && (udph->check != 0)) {
		/* Only port and addr are changed, do fast csum update */
		udp_fast_csum_update(cp->af, udph, &cp->vaddr, &cp->daddr,
				     cp->vport, cp->dport);
		if (skb->ip_summed == CHECKSUM_COMPLETE)
			skb->ip_summed = CHECKSUM_NONE;
	} else {
		/* full checksum calculation */
		udph->check = 0;
		skb->csum = skb_checksum(skb, udphoff, skb->len - udphoff, 0);
#ifdef CONFIG_IP_VS_IPV6
		if (cp->af == AF_INET6)
			udph->check = csum_ipv6_magic(&cp->caddr.in6,
						      &cp->daddr.in6,
						      skb->len - udphoff,
						      cp->protocol, skb->csum);
		else
#endif
			udph->check = csum_tcpudp_magic(cp->caddr.ip,
							cp->daddr.ip,
							skb->len - udphoff,
							cp->protocol,
							skb->csum);
		if (udph->check == 0)
			udph->check = CSUM_MANGLED_0;
		skb->ip_summed = CHECKSUM_UNNECESSARY;
	}
	return 1;
}

```



# csum_replace2

```
alculate IP-checksum:
Depending on if the value you changed was a __be16 or __be32, you use csum_replace2 and csum_replace4 respectively.

Calculating transport protocol checksum:
Depending on if the value you changed was a __be16 or __be32, you use inet_proto_csum_replace2 or inet_proto_csum_replace4 respectively

```