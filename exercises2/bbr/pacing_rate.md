 
#  Pacing处理
 

bbr的pacing处理有两种方式：
> ## 1）、依赖于tc-fq的pacing

当使用tc-fq时，qdisc默认会使能rate_enable限速，这个流程也会利用bbr算法计算得到的sk_pacing_rate完成pacing功能；
```
static struct sk_buff *fq_dequeue(struct Qdisc *sch)
{
	//rate_enable模式使能
	if (!q->rate_enable)
		goto out;
 
	/* Do not pace locally generated ack packets */
	if (skb_is_tcp_pure_ack(skb))
		goto out;
 
	rate = q->flow_max_rate;
	if (skb->sk)
		rate = min(skb->sk->sk_pacing_rate, rate);
 
	if (rate <= q->low_rate_threshold) {
		f->credit = 0;
		plen = qdisc_pkt_len(skb);
	} else {
		plen = max(qdisc_pkt_len(skb), q->quantum);
		if (f->credit > 0)
			goto out;
	}
	if (rate != ~0U) {
		u64 len = (u64)plen * NSEC_PER_SEC;
 
		if (likely(rate))
			do_div(len, rate);
		/* Since socket rate can change later,
		 * clamp the delay to 1 second.
		 * Really, providers of too big packets should be fixed !
		 */
		if (unlikely(len > NSEC_PER_SEC)) {
			len = NSEC_PER_SEC;
			q->stat_pkts_too_long++;
		}
		/* Account for schedule/timers drifts.
		 * f->time_next_packet was set when prior packet was sent,
		 * and current time (@now) can be too late by tens of us.
		 */
		if (f->time_next_packet)
			len -= min(len/2, now - f->time_next_packet);
		f->time_next_packet = now + len;
	}
out:
	qdisc_bstats_update(sch, skb);
	return skb;
}
```

> ## 2）、tcp主动pacing

bbr_init时，默认使能SK_PACING_NEEDED；

```
static void bbr_init(struct sock *sk)
{
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
}
```

tcp_write_xmit的时候，通过tcp_pacing_check判断当前是否已经启动pacing高精度定时器，如果已经启动，则退出xmit流程；
```
static bool tcp_write_xmit(struct sock *sk, unsigned int mss_now, int nonagle,
			   int push_one, gfp_t gfp)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct sk_buff *skb;
	unsigned int tso_segs, sent_pkts;
	int cwnd_quota;
	int result;
	bool is_cwnd_limited = false, is_rwnd_limited = false;
	u32 max_segs;
 
	sent_pkts = 0;
 
	tcp_mstamp_refresh(tp);
	if (!push_one) {
		/* Do MTU probing. */
		result = tcp_mtu_probe(sk);
		if (!result) {
			return false;
		} else if (result > 0) {
			sent_pkts = 1;
		}
	}
 
	max_segs = tcp_tso_segs(sk, mss_now);
	while ((skb = tcp_send_head(sk))) {
		unsigned int limit;
 
		if (tcp_pacing_check(sk))
			break;
    ...
}
```

__tcp_transmit_skb判断是否需要tcp层做pacing，需要的话就启动高精度定时器；

```
static int __tcp_transmit_skb(struct sock *sk, struct sk_buff *skb,
			      int clone_it, gfp_t gfp_mask, u32 rcv_nxt)
{
	...
	if (skb->len != tcp_header_size) {
		tcp_event_data_sent(tp, sk);
		tp->data_segs_out += tcp_skb_pcount(skb);
		tp->bytes_sent += skb->len - tcp_header_size;
		tcp_internal_pacing(sk, skb);
	}
	...
}
static void tcp_internal_pacing(struct sock *sk, const struct sk_buff *skb)
{
	u64 len_ns;
	u32 rate;
 
	if (!tcp_needs_internal_pacing(sk))
		return;
	rate = sk->sk_pacing_rate;
	if (!rate || rate == ~0U)
		return;
 
	/* Should account for header sizes as sch_fq does,
	 * but lets make things simple.
	 */
	//sk_pacing_rate表示1分钟能发送的字节数
	//skb->len / rate表示发送skb->len字节数需要的时间长度(长度是分钟)
	//len_ns = skb->len / rate * NSEC_PER_SEC即将时间换算成纳秒，然后启动pacing高精度定时器
	len_ns = (u64)skb->len * NSEC_PER_SEC;
	do_div(len_ns, rate);
	hrtimer_start(&tcp_sk(sk)->pacing_timer,
		      ktime_add_ns(ktime_get(), len_ns),
		      HRTIMER_MODE_ABS_PINNED_SOFT);
	sock_hold(sk);
}
```
 