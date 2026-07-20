 
 
 
 
 #   mlx4_en_test_loopback_xmit
 
 ```
 static int mlx4_en_test_loopback_xmit(struct mlx4_en_priv *priv)
{
	panic("Disabled");
#if 0 // AKAROS_PORT
	struct sk_buff *skb;
	struct ethhdr *ethh;
	unsigned char *packet;
	unsigned int packet_size = MLX4_LOOPBACK_TEST_PAYLOAD;
	unsigned int i;
	int err;


	/* build the pkt before xmit */
	skb = netdev_alloc_skb(priv->dev,
			       MLX4_LOOPBACK_TEST_PAYLOAD + ETHERHDRSIZE + NET_IP_ALIGN);
	if (!skb)
		return -ENOMEM;

	skb_reserve(skb, NET_IP_ALIGN);

	ethh = (struct ethhdr *)skb_put(skb, sizeof(struct ethhdr));
	packet	= (unsigned char *)skb_put(skb, packet_size);
	memcpy(ethh->h_dest, priv->dev->ea, Eaddrlen);
	eth_zero_addr(ethh->h_source);
	ethh->h_proto = cpu_to_be16(ETH_P_ARP);
	skb_set_mac_header(skb, 0);
	for (i = 0; i < packet_size; ++i)	/* fill our packet */
		packet[i] = (unsigned char)(i & 0xff);

	/* xmit the pkt */
	err = mlx4_en_xmit(skb, priv->dev);
	return err;
#endif
}
 ```
 
 ```
 /*
 * Description: This function is called to send up the packet buffer
 *              to the IP stack.
 *
 * Input:       emfi - EMF instance information
 *              skb  - Pointer to the packet buffer.
 */
void
emf_sendup(emf_info_t *emfi, struct sk_buff *skb)
{
	/* Called only when frame is received from one of the LAN ports */
	ASSERT(skb->dev->br_port);
	ASSERT(skb->protocol == __constant_htons(ETH_P_IP));

	/* Send the buffer as if the packet is being sent by bridge */
	skb->dev = emfi->br_dev;
	switch (skb->pkt_type) {
	case PACKET_MULTICAST:
	case PACKET_BROADCAST:
	case PACKET_HOST:
		break;
	case PACKET_OTHERHOST:
	default:
		skb->pkt_type = PACKET_HOST;
		break;
	}
#if LINUX_VERSION_CODE >= KERNEL_VERSION(2, 6, 22)
	skb_set_mac_header(skb, -ETH_HLEN);
#else
	skb->mac.raw = skb->data - ETH_HLEN;
#endif

	netif_rx(skb);

	return;
}
 ```
 
```
static int mlx4_en_test_loopback_xmit(struct mlx4_en_priv *priv)
{
	struct sk_buff *skb;
	struct ethhdr *ethh;
	unsigned char *packet;
	unsigned int packet_size = MLX4_LOOPBACK_TEST_PAYLOAD;
	unsigned int i;
	int err;


	/* build the pkt before xmit */
	skb = netdev_alloc_skb(priv->dev, MLX4_LOOPBACK_TEST_PAYLOAD + ETH_HLEN + NET_IP_ALIGN);
	if (!skb) {
		en_err(priv, "-LOOPBACK_TEST_XMIT- failed to create skb for xmit\n");
		return -ENOMEM;
	}
	skb_reserve(skb, NET_IP_ALIGN);

	ethh = (struct ethhdr *)skb_put(skb, sizeof(struct ethhdr));
	packet	= (unsigned char *)skb_put(skb, packet_size);
	memcpy(ethh->h_dest, priv->dev->dev_addr, ETH_ALEN);
	memset(ethh->h_source, 0, ETH_ALEN);
	ethh->h_proto = htons(ETH_P_ARP);
	skb_set_mac_header(skb, 0);
	for (i = 0; i < packet_size; ++i)	/* fill our packet */
		packet[i] = (unsigned char)(i & 0xff);

	/* xmit the pkt */
	err = mlx4_en_xmit(skb, priv->dev);
	return err;
}
```
# myri_type_trans

```static __be16 myri_type_trans(struct sk_buff *skb, struct net_device *dev)
{
	struct ethhdr *eth;
	unsigned char *rawp;

	skb_set_mac_header(skb, MYRI_PAD_LEN);
	skb_pull(skb, dev->hard_header_len);
	eth = eth_hdr(skb);

#ifdef DEBUG_HEADER
	DHDR(("myri_type_trans: "));
	dump_ehdr(eth);
#endif
	if (*eth->h_dest & 1) {
		if (memcmp(eth->h_dest, dev->broadcast, ETH_ALEN)==0)
			skb->pkt_type = PACKET_BROADCAST;
		else
			skb->pkt_type = PACKET_MULTICAST;
	} else if (dev->flags & (IFF_PROMISC|IFF_ALLMULTI)) {
		if (memcmp(eth->h_dest, dev->dev_addr, ETH_ALEN))
			skb->pkt_type = PACKET_OTHERHOST;
	}

	if (ntohs(eth->h_proto) >= 1536)
		return eth->h_proto;

	rawp = skb->data;

	
	if (*(unsigned short *)rawp == 0xFFFF)
		return htons(ETH_P_802_3);

	
	return htons(ETH_P_802_2);
}
```

#   skb_from_pkt

```
int skb_from_pkt(void *pkt, u32 pkt_len, struct sk_buff **skb)
{
	*skb = alloc_skb(LL_MAX_HEADER + pkt_len, GFP_ATOMIC);
	if (!*skb) {
		log_err("Could not allocate a skb.");
		return -ENOMEM;
	}

	skb_reserve(*skb, LL_MAX_HEADER); /* Reserve space for Link Layer data. */
	skb_put(*skb, pkt_len); /* L3 + L4 + payload. */

	skb_set_mac_header(*skb, 0);
	skb_set_network_header(*skb, 0);
	skb_set_transport_header(*skb, net_hdr_size(pkt));

	(*skb)->ip_summed = CHECKSUM_UNNECESSARY;
	switch (get_l3_proto(pkt)) {
	case 6:
		(*skb)->protocol = htons(ETH_P_IPV6);
		break;
	case 4:
		(*skb)->protocol = htons(ETH_P_IP);
		break;
	default:
		log_err("Invalid mode: %u.", get_l3_proto(pkt));
		kfree_skb(*skb);
		return -EINVAL;
	}

	/* Copy packet content to skb. */
	memcpy(skb_network_header(*skb), pkt, pkt_len);

	return 0;
}
```

# reply_to_arp_request
 
 ```
 VOID
reply_to_arp_request(struct sk_buff *skb)
{
	PMINI_ADAPTER		Adapter;
	struct ArpHeader 	*pArpHdr = NULL;
	struct ethhdr		*pethhdr = NULL;
	UCHAR 				uiIPHdr[4];
	/* Check for valid skb */
	if(skb == NULL)
	{
		BCM_DEBUG_PRINT(Adapter,DBG_TYPE_PRINTK, 0, 0, "Invalid skb: Cannot reply to ARP request\n");
		return;
	}


	Adapter = GET_BCM_ADAPTER(skb->dev);
	/* Print the ARP Request Packet */
	BCM_DEBUG_PRINT(Adapter,DBG_TYPE_TX, ARP_RESP, DBG_LVL_ALL, "ARP Packet Dump :");
	BCM_DEBUG_PRINT_BUFFER(Adapter,DBG_TYPE_TX, ARP_RESP, DBG_LVL_ALL, (PUCHAR)(skb->data), skb->len);

	/*
	 * Extract the Ethernet Header and Arp Payload including Header
     */
	pethhdr = (struct ethhdr *)skb->data;
	pArpHdr  = (struct ArpHeader *)(skb->data+ETH_HLEN);

	if(Adapter->bETHCSEnabled)
	{
		if(memcmp(pethhdr->h_source, Adapter->dev->dev_addr, ETH_ALEN))
		{
			bcm_kfree_skb(skb);
			return;
		}
	}

	// Set the Ethernet Header First.
	memcpy(pethhdr->h_dest, pethhdr->h_source, ETH_ALEN);
	if(!memcmp(pethhdr->h_source, Adapter->dev->dev_addr, ETH_ALEN))
	{
		pethhdr->h_source[5]++;
	}

	/* Set the reply to ARP Reply */
	pArpHdr->arp.ar_op = ntohs(ARPOP_REPLY);

	/* Set the HW Address properly */
	memcpy(pArpHdr->ar_sha, pethhdr->h_source, ETH_ALEN);
	memcpy(pArpHdr->ar_tha, pethhdr->h_dest, ETH_ALEN);

	// Swapping the IP Adddress
	memcpy(uiIPHdr,pArpHdr->ar_sip,4);
	memcpy(pArpHdr->ar_sip,pArpHdr->ar_tip,4);
	memcpy(pArpHdr->ar_tip,uiIPHdr,4);

	/* Print the ARP Reply Packet */

	BCM_DEBUG_PRINT(Adapter,DBG_TYPE_TX, ARP_RESP, DBG_LVL_ALL, "ARP REPLY PACKET: ");

	/* Send the Packet to upper layer */
	BCM_DEBUG_PRINT_BUFFER(Adapter,DBG_TYPE_TX, ARP_RESP, DBG_LVL_ALL, (PUCHAR)(skb->data), skb->len);

	skb->protocol = eth_type_trans(skb,skb->dev);
	skb->pkt_type = PACKET_HOST;

//	skb->mac.raw=skb->data+LEADER_SIZE;
	skb_set_mac_header (skb, LEADER_SIZE);
	netif_rx(skb);
	BCM_DEBUG_PRINT(Adapter,DBG_TYPE_TX, ARP_RESP, DBG_LVL_ALL, "<=============\n");
	return;
}
 ```
 
 
 
 
 
 
 
 
 
 
 
 # refercences
 
  