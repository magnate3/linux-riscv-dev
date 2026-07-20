
# TEST_TCP_SEG

```
//#if TEST_TCP_SEG
#if 0
	struct iphdr *iph = ip_hdr(skb);
	//pr_info("********   proto %x, %x,%x,%x and mms : %u ************* \n",iph->protocol,htons(IPPROTO_ICMP), htons(IPPROTO_UDP), htons(IPPROTO_TCP), mss);
	if(IPPROTO_ICMP == iph->protocol) {
	    pr_info("********  icmp proto and mms : %u ************* \n", mss);
	}
	else if(IPPROTO_UDP == iph->protocol) {
	    pr_info("********  udp proto and mms : %u ************* \n", mss);
	}
	else if(IPPROTO_TCP == iph->protocol) {
	    struct tcphdr *th= (struct tcphdr *)skb_transport_header(skb);
	    pr_info("tcp proto and mms : %u, src port %d, dst pot %d, skb_is_gso: %d ,skb_shinfo(skb)->nr_frags:%u, skb_is_nonlinear:%d, sk_buff: len:%u  skb->data_len:%u  \n", mss, htons(th->source), htons(th->dest), skb_is_gso(skb),skb_shinfo(skb)->nr_frags, skb_is_nonlinear(skb), skb->len,skb->data_len);
	}
#endif
```

```
static netdev_tx_t e1000_xmit_frame(struct sk_buff *skb,
				    struct net_device *netdev)
{
	struct e1000_adapter *adapter = netdev_priv(netdev);
	struct e1000_ring *tx_ring = adapter->tx_ring;
	unsigned int first;
	unsigned int tx_flags = 0;
	unsigned int len = skb_headlen(skb);
	unsigned int nr_frags;
	unsigned int mss;
	int count = 0;
	int tso;
	unsigned int f;
	__be16 protocol = vlan_get_protocol(skb);

	if (test_bit(__E1000_DOWN, &adapter->state)) {
		dev_kfree_skb_any(skb);
		return NETDEV_TX_OK;
	}

	if (skb->len <= 0) {
		dev_kfree_skb_any(skb);
		return NETDEV_TX_OK;
	}

	/* The minimum packet size with TCTL.PSP set is 17 bytes so
	 * pad skb in order to meet this minimum size requirement
	 */
	if (skb_put_padto(skb, 17))
		return NETDEV_TX_OK;

	mss = skb_shinfo(skb)->gso_size;
	//
//#if TEST_TCP_SEG
#if 0
	struct iphdr *iph = ip_hdr(skb);
	//pr_info("********   proto %x, %x,%x,%x and mms : %u ************* \n",iph->protocol,htons(IPPROTO_ICMP), htons(IPPROTO_UDP), htons(IPPROTO_TCP), mss);
	if(IPPROTO_ICMP == iph->protocol) {
	    pr_info("********  icmp proto and mms : %u ************* \n", mss);
	}
	else if(IPPROTO_UDP == iph->protocol) {
	    pr_info("********  udp proto and mms : %u ************* \n", mss);
	}
	else if(IPPROTO_TCP == iph->protocol) {
	    struct tcphdr *th= (struct tcphdr *)skb_transport_header(skb);
	    pr_info("tcp proto and mms : %u, src port %d, dst pot %d, skb_is_gso: %d ,skb_shinfo(skb)->nr_frags:%u, skb_is_nonlinear:%d, sk_buff: len:%u  skb->data_len:%u  \n", mss, htons(th->source), htons(th->dest), skb_is_gso(skb),skb_shinfo(skb)->nr_frags, skb_is_nonlinear(skb), skb->len,skb->data_len);
	}
#endif
	if (mss) {
		u8 hdr_len;

		/* TSO Workaround for 82571/2/3 Controllers -- if skb->data
		 * points to just header, pull a few bytes of payload from
		 * frags into skb->data
		 */
		hdr_len = skb_transport_offset(skb) + tcp_hdrlen(skb);
		/* we do this workaround for ES2LAN, but it is un-necessary,
		 * avoiding it could save a lot of cycles
		 */
		if (skb->data_len && (hdr_len == len)) {
			unsigned int pull_size;

			pull_size = min_t(unsigned int, 4, skb->data_len);
			if (!__pskb_pull_tail(skb, pull_size)) {
				e_err("__pskb_pull_tail failed.\n");
				dev_kfree_skb_any(skb);
				return NETDEV_TX_OK;
			}
			len = skb_headlen(skb);
		}
	}

	/* reserve a descriptor for the offload context */
	if ((mss) || (skb->ip_summed == CHECKSUM_PARTIAL))
		count++;
	count++;

	count += DIV_ROUND_UP(len, adapter->tx_fifo_limit);

	nr_frags = skb_shinfo(skb)->nr_frags;
	for (f = 0; f < nr_frags; f++)
		count += DIV_ROUND_UP(skb_frag_size(&skb_shinfo(skb)->frags[f]),
				      adapter->tx_fifo_limit);

	if (adapter->hw.mac.tx_pkt_filtering)
		e1000_transfer_dhcp_info(adapter, skb);

	/* need: count + 2 desc gap to keep tail from touching
	 * head, otherwise try next time
	 */
	if (e1000_maybe_stop_tx(tx_ring, count + 2))
		return NETDEV_TX_BUSY;

	if (skb_vlan_tag_present(skb)) {
		tx_flags |= E1000_TX_FLAGS_VLAN;
		tx_flags |= (skb_vlan_tag_get(skb) <<
			     E1000_TX_FLAGS_VLAN_SHIFT);
	}

	first = tx_ring->next_to_use;

	tso = e1000_tso(tx_ring, skb, protocol);
	if (tso < 0) {
		dev_kfree_skb_any(skb);
		return NETDEV_TX_OK;
	}

	if (tso)
		tx_flags |= E1000_TX_FLAGS_TSO;
	else if (e1000_tx_csum(tx_ring, skb, protocol))
		tx_flags |= E1000_TX_FLAGS_CSUM;

	/* Old method was to assume IPv4 packet by default if TSO was enabled.
	 * 82571 hardware supports TSO capabilities for IPv6 as well...
	 * no longer assume, we must.
	 */
	if (protocol == htons(ETH_P_IP))
		tx_flags |= E1000_TX_FLAGS_IPV4;

	if (unlikely(skb->no_fcs))
		tx_flags |= E1000_TX_FLAGS_NO_FCS;

	/* if count is 0 then mapping error has occurred */
	count = e1000_tx_map(tx_ring, skb, first, adapter->tx_fifo_limit,
			     nr_frags);
	if (count) {
		if (unlikely(skb_shinfo(skb)->tx_flags & SKBTX_HW_TSTAMP) &&
		    (adapter->flags & FLAG_HAS_HW_TIMESTAMP)) {
			if (!adapter->tx_hwtstamp_skb) {
				skb_shinfo(skb)->tx_flags |= SKBTX_IN_PROGRESS;
				tx_flags |= E1000_TX_FLAGS_HWTSTAMP;
				adapter->tx_hwtstamp_skb = skb_get(skb);
				adapter->tx_hwtstamp_start = jiffies;
				schedule_work(&adapter->tx_hwtstamp_work);
			} else {
				adapter->tx_hwtstamp_skipped++;
			}
		}

		skb_tx_timestamp(skb);

		netdev_sent_queue(netdev, skb->len);
		e1000_tx_queue(tx_ring, tx_flags, count);
		/* Make sure there is space in the ring for the next send. */
		e1000_maybe_stop_tx(tx_ring,
				    (MAX_SKB_FRAGS *
				     DIV_ROUND_UP(PAGE_SIZE,
						  adapter->tx_fifo_limit) + 2));

		if (!netdev_xmit_more() ||
		    netif_xmit_stopped(netdev_get_tx_queue(netdev, 0))) {
			if (adapter->flags2 & FLAG2_PCIM2PCI_ARBITER_WA)
				e1000e_update_tdt_wa(tx_ring,
						     tx_ring->next_to_use);
			else
				writel(tx_ring->next_to_use, tx_ring->tail);
		}
	} else {
		dev_kfree_skb_any(skb);
		tx_ring->buffer_info[first].time_stamp = 0;
		tx_ring->next_to_use = first;
	}

	return NETDEV_TX_OK;
}
```