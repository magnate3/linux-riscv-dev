
# NF_INET_LOCAL_OUT

```
static struct nf_hook_ops test_hookops = {
    .pf = NFPROTO_IPV4,
    .priority = NF_IP_PRI_MANGLE,
    //.hooknum = NF_INET_PRE_ROUTING,
    .hooknum = NF_INET_LOCAL_OUT,
    .hook = test_hookfn,
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,4,0)
    .owner = THIS_MODULE,
#endif
};
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/jubo_ping_send/ping.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/jubo_ping_send/dump.png)

# morefrag: 0 

```
[161341.320596] ************** remain: 6494,  offset 0, morefrag: 0 
```

# switch_hook_forward

```
static unsigned int switch_hook_forward(
    unsigned int hook,
    struct sk_buff *skb,
    const struct net_device *dev_in,
    const struct net_device *dev_out,
    int (*okfn)(struct sk_buff *)
) {
    unsigned int result = NF_ACCEPT;
    struct iphdr *ip_header = ip_hdr(skb);

    if (ip_header->protocol == IPPROTO_TCP) {
        unsigned int ip_header_length = ip_hdrlen(skb);
        unsigned int ip_packet_length = ntohs(ip_header->tot_len);
        
        if (ip_packet_length > MTU) {
            int unused, remain, length;
            int offset, pstart;
            __be16 morefrag;
            struct sk_buff *skb_frag;
            struct iphdr *ip_header_frag;
            skb_push(skb, ETH_HLEN);

            unused = LL_RESERVED_SPACE(skb->dev);
            remain = skb->len - ETH_HLEN - ip_header_length;
            offset = (ntohs(ip_header->frag_off) & IP_OFFSET) << 3;
            pstart = ETH_HLEN + ip_header_length;
            morefrag = ip_header->frag_off & htons(IP_MF);

            while (remain > 0) {
                length = remain > MTU ? MTU : remain;
                if ((skb_frag = alloc_skb(unused + ETH_HLEN + ip_header_length + length, GFP_ATOMIC)) == NULL) {
                    break;
                }
                skb_frag->dev = skb->dev;
                skb_reserve(skb_frag, unused);
                skb_put(skb_frag, ETH_HLEN + ip_header_length + length);
                skb_frag->mac_header = skb_frag->data;
                skb_frag->network_header = skb_frag->data + ETH_HLEN;
                skb_frag->transport_header = skb_frag->data + ETH_HLEN + ip_header_length;
                skb_copy_from_linear_data(skb, skb_mac_header(skb_frag), ETH_HLEN + ip_header_length);
                skb_copy_bits(skb, pstart, skb_transport_header(skb_frag), length);
                remain = remain - length;

                skb_pull(skb_frag, ETH_HLEN);

                skb_reset_network_header(skb_frag);
                skb_pull(skb_frag, ip_header_length);

                skb_reset_transport_header(skb_frag);
                skb_push(skb_frag, ip_header_length);

                ip_header_frag = ip_hdr(skb_frag);
                ip_header_frag->frag_off = htons(offset >> 3);
                if (remain > 0 || morefrag) {
                    ip_header_frag->frag_off = ip_header_frag->frag_off | htons(IP_MF);
                }

                ip_header_frag->frag_off = ip_header_frag->frag_off | htons(IP_DF);
                ip_header_frag->tot_len  = htons(ip_header_length + length);
                ip_header_frag->protocol = IPPROTO_TCP;
                ip_send_check(ip_header_frag);
                skb_push(skb_frag, ETH_HLEN);

                dev_queue_xmit(skb_frag);
                pstart = pstart + length;
                offset = offset + length;
            }
            skb_pull(skb, ETH_HLEN);
            result = NF_DROP;
        }

    }
    return result;
}

```


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/jubo_ping_send/frag.png)

[NetDevKernels/firewall_packsplit/main.c](https://github.com/ahmedskhalil/NetDevKernels/blob/c580cd7da5d597c338eb4e5e85619221f3f37535/firewall_packsplit/main.c)