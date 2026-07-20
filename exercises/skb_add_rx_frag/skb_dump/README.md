

如果skb->data_len为零，则skb是线性的，并且skb的完整数据内容为skb->data。如果skb不为零，则skb->data_len不是线性的，
并且skb->data仅包含数据的第一个(线性)部分。此区域的长度为skb->len - skb->data_len。为了方便起见，
skb_headlen()帮助器函数会计算该值。skb_is_nonlinear()帮助器函数告诉我们skb是否是线性的。
其余的数据可以以分页片段和skb片段的形式存在，顺序如下。

***(1)*** skb_shinfo(skb)->nr_frags告诉分页片段的数量。每个分页片段由结构数组skb_shinfo(skb)->frags[0..skb_shinfo(skb)->nr_frags]
中的数据结构描述。skb_frag_size()和skb_frag_address()帮助函数帮助处理这些数据。
它们接受描述分页片段的结构的地址。根据您的内核版本，还有其他有用的帮助函数。

***(2)*** 如果分页片段中的数据总大小小于skb->data_len，则其余数据在skb片段中。
这是在skb_shinfo(skb)->frag_list (参见内核中的skb_walk_frags() )时附加到该skb的skb的列表。
请注意，可能线性部分中没有数据和/或分页片段中没有数据


#  jumbo and  frag_list

```
//  hinic_rx.c 
static int rxq_recv(struct hinic_rxq *rxq, int budget)
{
       if (pkt_len <= HINIC_RX_BUF_SZ) {
                        __skb_put(skb, pkt_len);
                } else {
                        __skb_put(skb, HINIC_RX_BUF_SZ);
                        num_wqes = rx_recv_jumbo_pkt(rxq, skb, pkt_len -
                                                     HINIC_RX_BUF_SZ, ci);
                }

}
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/skb_dump/jumbo.png)



# ping -c 1  -s 6500 10.10.16.251
```
[root@bogon ~]# ping -c 1  -s 6500 10.10.16.251
PING 10.10.16.251 (10.10.16.251) 6500(6528) bytes of data.
6508 bytes from 10.10.16.251: icmp_seq=1 ttl=64 time=3281 ms

--- 10.10.16.251 ping statistics ---
1 packets transmitted, 1 received, 0% packet loss, time 0ms
rtt min/avg/max/mdev = 3281.775/3281.775/3281.775/0.000 ms
[root@bogon ~]# 

```


# wallk skb frags list
```
dmesg | grep 'dump ping request skb begin' -A 400 > dump.txt
[root@centos7 skb_dump]# dmesg | grep '****************** wallk skb frags list'
[153477.995334]  ****************** wallk skb frags list 1 timers
[153478.735079]  ****************** wallk skb frags list 2 timers
[153479.474822]  ****************** wallk skb frags list 3 timers
[153480.214566]  ****************** wallk skb frags list 4 timers
[root@centos7 skb_dump]# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/skb_add_rx_frag/skb_dump/dump.png)


# linux/net/core/skbuff.c

```
/* Dump skb information and contents.
 *
 * Must only be called from net_ratelimit()-ed paths.
 *
 * Dumps whole packets if full_pkt, only headers otherwise.
 */
void skb_dump(const char *level, const struct sk_buff *skb, bool full_pkt)
{
	struct skb_shared_info *sh = skb_shinfo(skb);
	struct net_device *dev = skb->dev;
	struct sock *sk = skb->sk;
	struct sk_buff *list_skb;
	bool has_mac, has_trans;
	int headroom, tailroom;
	int i, len, seg_len;

	if (full_pkt)
		len = skb->len;
	else
		len = min_t(int, skb->len, MAX_HEADER + 128);

	headroom = skb_headroom(skb);
	tailroom = skb_tailroom(skb);

	has_mac = skb_mac_header_was_set(skb);
	has_trans = skb_transport_header_was_set(skb);

	printk("%sskb len=%u headroom=%u headlen=%u tailroom=%u\n"
	       "mac=(%d,%d) net=(%d,%d) trans=%d\n"
	       "shinfo(txflags=%u nr_frags=%u gso(size=%hu type=%u segs=%hu))\n"
	       "csum(0x%x ip_summed=%u complete_sw=%u valid=%u level=%u)\n"
	       "hash(0x%x sw=%u l4=%u) proto=0x%04x pkttype=%u iif=%d\n",
	       level, skb->len, headroom, skb_headlen(skb), tailroom,
	       has_mac ? skb->mac_header : -1,
	       has_mac ? skb_mac_header_len(skb) : -1,
	       skb->network_header,
	       has_trans ? skb_network_header_len(skb) : -1,
	       has_trans ? skb->transport_header : -1,
	       sh->tx_flags, sh->nr_frags,
	       sh->gso_size, sh->gso_type, sh->gso_segs,
	       skb->csum, skb->ip_summed, skb->csum_complete_sw,
	       skb->csum_valid, skb->csum_level,
	       skb->hash, skb->sw_hash, skb->l4_hash,
	       ntohs(skb->protocol), skb->pkt_type, skb->skb_iif);

	if (dev)
		printk("%sdev name=%s feat=%pNF\n",
		       level, dev->name, &dev->features);
	if (sk)
		printk("%ssk family=%hu type=%u proto=%u\n",
		       level, sk->sk_family, sk->sk_type, sk->sk_protocol);

	if (full_pkt && headroom)
		print_hex_dump(level, "skb headroom: ", DUMP_PREFIX_OFFSET,
			       16, 1, skb->head, headroom, false);

	seg_len = min_t(int, skb_headlen(skb), len);
	if (seg_len)
		print_hex_dump(level, "skb linear:   ", DUMP_PREFIX_OFFSET,
			       16, 1, skb->data, seg_len, false);
	len -= seg_len;

	if (full_pkt && tailroom)
		print_hex_dump(level, "skb tailroom: ", DUMP_PREFIX_OFFSET,
			       16, 1, skb_tail_pointer(skb), tailroom, false);

	for (i = 0; len && i < skb_shinfo(skb)->nr_frags; i++) {
		skb_frag_t *frag = &skb_shinfo(skb)->frags[i];
		u32 p_off, p_len, copied;
		struct page *p;
		u8 *vaddr;

		skb_frag_foreach_page(frag, skb_frag_off(frag),
				      skb_frag_size(frag), p, p_off, p_len,
				      copied) {
			seg_len = min_t(int, p_len, len);
			vaddr = kmap_atomic(p);
			print_hex_dump(level, "skb frag:     ",
				       DUMP_PREFIX_OFFSET,
				       16, 1, vaddr + p_off, seg_len, false);
			kunmap_atomic(vaddr);
			len -= seg_len;
			if (!len)
				break;
		}
	}

	if (full_pkt && skb_has_frag_list(skb)) {
		printk("skb fraglist:\n");
		skb_walk_frags(skb, list_skb)
			skb_dump(level, list_skb, true);
	}
}
EXPORT_SYMBOL(skb_dump);
```




```
 
 #ifdef CONFIG_BUG
 void skb_dump(const char *level, const struct sk_buff *skb, bool dump_header,
 	      bool dump_mac_header, bool dump_network_header)
 {
 	struct sk_buff *frag_iter;
 	int i;
 
 	if (dump_header)
 		printk("%sskb len=%u data_len=%u pkt_type=%u gso_size=%u gso_type=%u nr_frags=%u ip_summed=%u csum=%x csum_complete_sw=%d csum_valid=%d csum_level=%u\n",
 		       level, skb->len, skb->data_len, skb->pkt_type,
 		       skb_shinfo(skb)->gso_size, skb_shinfo(skb)->gso_type,
 		       skb_shinfo(skb)->nr_frags, skb->ip_summed, skb->csum,
 		       skb->csum_complete_sw, skb->csum_valid, skb->csum_level);
 
 	if (dump_mac_header && skb_mac_header_was_set(skb))
 		print_hex_dump(level, "mac header: ", DUMP_PREFIX_OFFSET, 16, 1,
 			       skb_mac_header(skb), skb_mac_header_len(skb),
 			       false);
 
 	if (dump_network_header && skb_network_header_was_set(skb))
 		print_hex_dump(level, "network header: ", DUMP_PREFIX_OFFSET,
 			       16, 1, skb_network_header(skb),
 			       skb_network_header_len(skb), false);
 
 	print_hex_dump(level, "skb data: ", DUMP_PREFIX_OFFSET, 16, 1,
 		       skb->data, skb->len, false);
 
 	for (i = 0; i < skb_shinfo(skb)->nr_frags; i++) {
 		skb_frag_t *frag = &skb_shinfo(skb)->frags[i];
 		u32 p_off, p_len, copied;
 		struct page *p;
 		u8 *vaddr;
 
 		skb_frag_foreach_page(frag, frag->page_offset, skb_frag_size(frag),
 				      p, p_off, p_len, copied) {
 			vaddr = kmap_atomic(p);
 			print_hex_dump(level, "skb frag: ", DUMP_PREFIX_OFFSET,
 				       16, 1, vaddr + p_off, p_len, false);
 			kunmap_atomic(vaddr);
 		}
 	}
 
 	if (skb_has_frag_list(skb))
 		printk("%sskb frags list:\n", level);
 	skb_walk_frags(skb, frag_iter)
 		skb_dump(level, frag_iter, false, false, false);
 }
 #endif

```