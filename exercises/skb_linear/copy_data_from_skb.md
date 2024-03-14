#  skb_linearize

```
static int
ip_vs_sip_fill_param(struct ip_vs_conn_param *p, struct sk_buff *skb)
{
	struct ip_vs_iphdr iph;
	unsigned int dataoff, datalen, matchoff, matchlen;
	const char *dptr;
	int retc;

	ip_vs_fill_iphdr(p->af, skb_network_header(skb), &iph);

	
	if (iph.protocol != IPPROTO_UDP)
		return -EINVAL;

	
	dataoff = iph.len + sizeof(struct udphdr);
	if (dataoff >= skb->len)
		return -EINVAL;

	if ((retc=skb_linearize(skb)) < 0)
		return retc;
	dptr = skb->data + dataoff;
	datalen = skb->len - dataoff;

	if (get_callid(dptr, dataoff, datalen, &matchoff, &matchlen))
		return -EINVAL;

	p->pe_data = kmemdup(dptr + matchoff, matchlen, GFP_ATOMIC);
	if (!p->pe_data)
		return -ENOMEM;

	p->pe_data_len = matchlen;

	return 0;
}
```

#  skb_copy_datagram_iovec
```
static int rawsock_recvmsg(struct kiocb *iocb, struct socket *sock,
				struct msghdr *msg, size_t len, int flags)
{
	int noblock = flags & MSG_DONTWAIT;
	struct sock *sk = sock->sk;
	struct sk_buff *skb;
	int copied;
	int rc;

	nfc_dbg("sock=%p sk=%p len=%zu flags=%d", sock, sk, len, flags);

	skb = skb_recv_datagram(sk, flags, noblock, &rc);
	if (!skb)
		return rc;

	msg->msg_namelen = 0;

	copied = skb->len;
	if (len < copied) {
		msg->msg_flags |= MSG_TRUNC;
		copied = len;
	}

	rc = skb_copy_datagram_iovec(skb, 0, msg->msg_iov, copied);

	skb_free_datagram(sk, skb);

	return rc ? : copied;
}
```