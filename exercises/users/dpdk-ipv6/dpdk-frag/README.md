

# 分片


```
static inline int
is_ipv4_fragment(const struct rte_ipv4_hdr *hdr)
{
	uint16_t flag_offset, ip_flag, ip_ofs;

	flag_offset = rte_be_to_cpu_16(hdr->fragment_offset);
	ip_ofs = (uint16_t)(flag_offset & RTE_IPV4_HDR_OFFSET_MASK);
	ip_flag = (uint16_t)(flag_offset & RTE_IPV4_HDR_MF_FLAG);

	return ip_flag != 0 || ip_ofs  != 0;
}
```


```
gro_udp4_reassemble(struct rte_mbuf *pkt,
		struct gro_udp4_tbl *tbl,
		uint64_t start_time)
{
if (!is_ipv4_fragment(ipv4_hdr))
		return -1;
frag_offset = rte_be_to_cpu_16(ipv4_hdr->fragment_offset);
is_last_frag = ((frag_offset & RTE_IPV4_HDR_MF_FLAG) == 0) ? 1 : 0;
frag_offset = (uint16_t)(frag_offset & RTE_IPV4_HDR_OFFSET_MASK) << 3;
```


# ipv6 


```
int is_ipv4_fragment(struct iphdr *ip) {
  // A packet is a fragment if its fragment offset is nonzero or if the MF flag is set.
  return ntohs(ip->frag_off) & (IP_OFFMASK | IP_MF);
}

int is_ipv6_fragment(struct ip6_hdr *ip6, size_t len) {
  if (ip6->ip6_nxt != IPPROTO_FRAGMENT) {
    return 0;
  }
  struct ip6_frag *frag = (struct ip6_frag *)(ip6 + 1);
  return len >= sizeof(*ip6) + sizeof(*frag) &&
         (frag->ip6f_offlg & (IP6F_OFF_MASK | IP6F_MORE_FRAG));
}

int ipv4_fragment_offset(struct iphdr *ip) {
  return ntohs(ip->frag_off) & IP_OFFMASK;
}

int ipv6_fragment_offset(struct ip6_frag *frag) {
  return ntohs((frag->ip6f_offlg & IP6F_OFF_MASK) >> 3);
}
```

