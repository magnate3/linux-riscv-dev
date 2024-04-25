

# 丢包


[ipv6: defrag: drop non-last frags smaller than min mtu](https://lists.linaro.org/archives/list/linux-stable-mirror@lists.linaro.org/message/MWJB32QVPVELA73EWNAKSJM5CQQEMJKQ/)     


```
---
 net/ipv6/netfilter/nf_conntrack_reasm.c |    4 ++++
 net/ipv6/reassembly.c                   |    4 ++++
 2 files changed, 8 insertions(+)


--- a/net/ipv6/netfilter/nf_conntrack_reasm.c
+++ b/net/ipv6/netfilter/nf_conntrack_reasm.c
@@ -565,6 +565,10 @@ int nf_ct_frag6_gather(struct net *net,
    hdr = ipv6_hdr(skb);
    fhdr = (struct frag_hdr *)skb_transport_header(skb);


+	if (skb->len - skb_network_offset(skb) < IPV6_MIN_MTU &&
+	    fhdr->frag_off & htons(IP6_MF))
+		return -EINVAL;
+
    skb_orphan(skb);
    fq = fq_find(net, fhdr->identification, user, hdr,
    	     skb->dev ? skb->dev->ifindex : 0);
--- a/net/ipv6/reassembly.c
+++ b/net/ipv6/reassembly.c
@@ -522,6 +522,10 @@ static int ipv6_frag_rcv(struct sk_buff
    	return 1;
    }


+	if (skb->len - skb_network_offset(skb) < IPV6_MIN_MTU &&
+	    fhdr->frag_off & htons(IP6_MF))
+		goto fail_hdr;
+
    iif = skb->dev ? skb->dev->ifindex : 0;
    fq = fq_find(net, fhdr->identification, hdr, iif);
    if (fq) {


```


