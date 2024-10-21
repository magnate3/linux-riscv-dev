

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

