

# 第二层recv


```
static struct packet_type vrr_packet_type __read_mostly = {
	.type = cpu_to_be16(ETH_P_VRR),
	.func = vrr_rcv,
};
```



```err = proto_register(&vrr_proto, 1);
	if (err) {
		goto out;
	}
	/* --- */

	/* Register our sockets protocol handler */
	err = sock_register(&vrr_family_ops);
	if (err) {
		goto out;
	}

	dev_add_pack(&vrr_packet_type);
```


```
int vrr_rcv(struct sk_buff *skb, struct net_device *dev, struct packet_type *pt,
	    struct net_device *orig_dev)
{
	const struct vrr_header *vh = vrr_hdr(skb);
	int err;

        WARN_ATOMIC;

	printk(KERN_ALERT "Received a VRR packet!");

	/* VRR_INFO("vrr_version: %x", vh->vrr_version); */
	/* VRR_INFO("pkt_type: %x", vh->pkt_type); */
	/* VRR_INFO("protocol: %x", ntohs(vh->protocol)); */
	/* VRR_INFO("data_len: %x", ntohs(vh->data_len)); */
	/* VRR_INFO("free: %x", vh->free); */
	/* VRR_INFO("h_csum: %x", vh->h_csum); */
	/* VRR_INFO("src_id: %x", ntohl(vh->src_id)); */
	/* VRR_INFO("dest_id: %x", ntohl(vh->dest_id)); */

	if (vh->pkt_type < 0 || vh->pkt_type >= VRR_NPTYPES) {
		VRR_ERR("Unknown pkt_type: %x", vh->pkt_type);
		goto drop;
	}

	reset_active_timeout();
        pset_reset_fail_count(ntohl(vh->src_id));

	err = (*vrr_rcvfunc[vh->pkt_type])(skb, vh);

	if (err) {
		VRR_ERR("Error in rcv func.");
		goto drop;
	}

	return NET_RX_SUCCESS;
drop:
        kfree_skb(skb);
	return NET_RX_DROP;
}
```

# 第三层


##  # 第三层 rcv    
```
static const struct net_protocol tunnel4_protocol = {
        .handler        =       tunnel4_rcv,
        ... ...
};
```

当IP层处理完成（此处指outer的IP header处理），进行上层协议分用，如果上层协议类型是IPPROTO_IPIP，则交由tunnel4_rcv处理。     
```
ip_rcv
   +
   |- ip_rcv_finish
       +
       |- 理由查询
       |- rt->dst.input 即 ip_local_deliver # 路由查询结果Local IN
            +
            |- ip_local_deliver_finish
                  +
                  |- inet_protos[]表查询（L4分用）
                  |- ipprot->handler 即 tunnel4_rcv
```

### quic   
```
static const struct net_protocol quic_protocol = {
	.handler =	quic_rcv,
	.err_handler =	quic_err,
	.no_policy =	1,
#if LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
	.netns_ok =	1,
#endif
};


	if ((rc = inet_del_protocol(inet_protos[IPPROTO_UDP],
			IPPROTO_UDP)) < 0) {
		pr_crit("%s: cannot remove UDP protocol\n", __func__);
		return rc;
	}

	if ((rc = inet_add_protocol(&quic_protocol, IPPROTO_UDP)) < 0) {
		pr_crit("%s: cannot add UDP protocol shim\n", __func__);
		return rc;
	}

```

#  第四层

##  第四层send   
```
static int ax25_sendmsg(struct socket *sock, struct msghdr *msg, size_t len)
{
   ax25_queue_xmit(skb, ax25->ax25_dev->dev);
}
```



```
static int tls_sendmsg(struct socket *sock, struct msghdr *msg, size_t size)
{
	struct sock *sk = sock->sk;
	struct tls_sock *tsk = tls_sk(sk);
	int ret = 0;
	long timeo = sock_sndtimeo(sk, msg->msg_flags & MSG_DONTWAIT);
	bool eor = !(msg->msg_flags & MSG_MORE) || IS_DTLS(tsk);
	struct sk_buff *skb = NULL;
	size_t copy, copied = 0;

	lock_sock(sock->sk);

	if (msg->msg_flags & MSG_OOB) {
		ret = -ENOTSUPP;
		goto send_end;
	}
	sk_clear_bit(SOCKWQ_ASYNC_NOSPACE, sk);

	if (!KTLS_SEND_READY(tsk)) {
		ret = -EBADMSG;
		goto send_end;
	}

	if (size > KTLS_MAX_PAYLOAD_SIZE && IS_DTLS(tsk)) {
		ret = -E2BIG;
		goto send_end;
	}

	while (msg_data_left(msg)) {
		bool merge = true;
		int i;
		struct page_frag *pfrag;

		if (sk->sk_err)
			goto send_end;

		if (!sk_stream_memory_free(sk))
			goto wait_for_memory;

		skb = tcp_write_queue_tail(sk);

		while (!skb) {
			skb = alloc_skb(0, sk->sk_allocation);
			if (skb)
				__skb_queue_tail(&sk->sk_write_queue, skb);
		}

		i = skb_shinfo(skb)->nr_frags;
		pfrag = sk_page_frag(sk);

		if (!sk_page_frag_refill(sk, pfrag))
			goto wait_for_memory;

		if (!skb_can_coalesce(skb, i, pfrag->page,
				      pfrag->offset)) {
			if (i == ALG_MAX_PAGES) {
				struct sk_buff *tskb;

				tskb = alloc_skb(0, sk->sk_allocation);
				if (!tskb)
					goto wait_for_memory;

				if (skb)
					skb->next = tskb;
				else
					__skb_queue_tail(&sk->sk_write_queue,
							 tskb);

				skb = tskb;
				skb->ip_summed = CHECKSUM_UNNECESSARY;
				continue;
			}
			merge = false;
		}

		copy = min_t(int, msg_data_left(msg),
			     pfrag->size - pfrag->offset);
		copy = min_t(int, copy, KTLS_MAX_PAYLOAD_SIZE - tsk->unsent);

		if (!sk_wmem_schedule(sk, copy))
			goto wait_for_memory;

		ret = skb_copy_to_page_nocache(sk, &msg->msg_iter, skb,
					       pfrag->page,
					       pfrag->offset,
					       copy);
		if (ret)
			goto send_end;

		/* Update the skb. */
		if (merge) {
			skb_frag_size_add(&skb_shinfo(skb)->frags[i - 1], copy);
		} else {
			skb_fill_page_desc(skb, i, pfrag->page,
					   pfrag->offset, copy);
			get_page(pfrag->page);
		}

		pfrag->offset += copy;
		copied += copy;
		skb->len += copy;
		skb->data_len += copy;
		tsk->unsent += copy;

		if (tsk->unsent >= KTLS_MAX_PAYLOAD_SIZE)
			tls_push(tsk);

		continue;

wait_for_memory:
		tls_push(tsk);
		set_bit(SOCK_NOSPACE, &sk->sk_socket->flags);
		ret = sk_stream_wait_memory(sk, &timeo);
		if (ret)
			goto send_end;
	}

	if (eor)
		tls_push(tsk);

send_end:
	ret = sk_stream_error(sk, msg->msg_flags, ret);

	/* make sure we wake any epoll edge trigger waiter */
	if (unlikely(skb_queue_len(&sk->sk_write_queue) == 0 && ret == -EAGAIN))
		sk->sk_write_space(sk);

	release_sock(sk);

	return ret < 0 ? ret : size;
}

```

## 第四层recv

```
static const struct proto_ops ax25_proto_ops = {
	.family		= PF_AX25,
	.owner		= THIS_MODULE,
	.release	= ax25_release,
	.bind		= ax25_bind,
	.connect	= ax25_connect,
	.socketpair	= sock_no_socketpair,
	.accept		= ax25_accept,
	.getname	= ax25_getname,
	.poll		= datagram_poll,
	.ioctl		= ax25_ioctl,
	.gettstamp	= sock_gettstamp,
	.listen		= ax25_listen,
	.shutdown	= ax25_shutdown,
	.setsockopt	= ax25_setsockopt,
	.getsockopt	= ax25_getsockopt,
	.sendmsg	= ax25_sendmsg,
	.recvmsg	= ax25_recvmsg,
	.mmap		= sock_no_mmap,
	.sendpage	= sock_no_sendpage,
};
```