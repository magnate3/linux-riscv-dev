


```
static void bbr_reset_startup_mode(struct sock *sk)
{
        struct bbr *bbr = inet_csk_ca(sk);

        bbr->mode = BBR_STARTUP;
        if(((struct inet_sock*)sk)->inet_dport == 3000)
        {
        pr_err("mode 2 mode=%d,min_rtt_us=%d,full_bw=%d,cycle_idx=%d,pacing_gain=%d,cwnd_gain=%d,rtt_cnt=%d\r\n",bbr->mode,bbr->min_rtt_us,bbr->full_bw,bbr->cycle_idx,bbr->pacing_gain,bbr->cwnd_gain,bbr->rtt_cnt);
        }
       
。。。。
}



static void bbr_reset_probe_bw_mode(struct sock *sk)
{
        struct bbr *bbr = inet_csk_ca(sk);

        bbr->mode = BBR_PROBE_BW;
        if(((struct inet_sock*)sk)->inet_dport == 3000)
        {
        pr_err("mode 2 mode=%d,min_rtt_us=%d,full_bw=%d,cycle_idx=%d,pacing_gain=%d,cwnd_gain=%d,rtt_cnt=%d\r\n",bbr->mode,bbr->min_rtt_us,bbr->full_bw,bbr->cycle_idx,bbr->pacing_gain,bbr->cwnd_gain,bbr->rtt_cnt);
        }
。。。。
}

static void bbr_check_drain(struct sock *sk, const struct rate_sample *rs)
{
        struct bbr *bbr = inet_csk_ca(sk);

        if (bbr->mode == BBR_STARTUP && bbr_full_bw_reached(sk)) {
                bbr->mode = BBR_DRAIN;  /* drain queue we created */
        if(((struct inet_sock*)sk)->inet_dport == 3000)
        {
        pr_err("mode 2 mode=%d,min_rtt_us=%d,full_bw=%d,cycle_idx=%d,pacing_gain=%d,cwnd_gain=%d,rtt_cnt=%d\r\n",bbr->mode,bbr->min_rtt_us,bbr->full_bw,bbr->cycle_idx,bbr->pacing_gain,bbr->cwnd_gain,bbr->rtt_cnt);
        }

。。。
}


static void bbr_update_min_rtt(struct sock *sk, const struct rate_sample *rs)
{
        struct tcp_sock *tp = tcp_sk(sk);
        struct bbr *bbr = inet_csk_ca(sk);
        bool filter_expired;
        static int count_expire=0;

  。。。。。。if (rs->rtt_us >= 0 &&
            (rs->rtt_us <= bbr->min_rtt_us || filter_expired)) {
                bbr->min_rtt_us = rs->rtt_us;
                bbr->min_rtt_stamp = tcp_time_stamp;
        }

        if (bbr_probe_rtt_mode_ms > 0 && filter_expired &&
            !bbr->idle_restart && bbr->mode != BBR_PROBE_RTT) {
                bbr->mode = BBR_PROBE_RTT;  /* dip, drain queue */
        if(((struct inet_sock*)sk)->inet_dport == 3000)
        {
        pr_err("mode 2 mode=%d,min_rtt_us=%d,full_bw=%d,cycle_idx=%d,pacing_gain=%d,cwnd_gain=%d,rtt_cnt=%d\r\n",bbr->mode,bbr->min_rtt_us,bbr->full_bw,bbr->cycle_idx,bbr->pacing_gain,bbr->cwnd_gain,bbr->rtt_cnt);
        }

                bbr->pacing_gain = BBR_UNIT;
.....
}

```


```
static void bbr_main(struct sock *sk, const struct rate_sample *rs)
{
struct bbr *bbr = inet_csk_ca(sk);
u32 bw;

bbr_update_model(sk, rs);

bw = bbr_bw(sk);
bbr_set_pacing_rate(sk, bw, bbr->pacing_gain);
bbr_set_tso_segs_goal(sk);
bbr_set_cwnd(sk, rs, rs->acked_sacked, bw, bbr->cwnd_gain);
if(((struct inet_sock*)sk)->inet_dport == 3000)
{
printk("main mode=%d,min_rtt_us=%d,cur_bw=%d,cycle_idx=%d,pacing_gain=%d,cwnd_gain=%d,rtt_cnt=%d,snd_cwnd=%d, snd_nxt=%u,snd_una=%u\r\n",bbr->mode,bbr->min_rtt_us,bbr_bw(sk),bbr->cycle_idx,bbr->pacing_gain,bbr->cwnd_gain,bbr->rtt_cnt,tcp_sk(sk)->snd_cwnd, tcp_sk(sk)->snd_nxt, tcp_sk(sk)->snd_una );
}

}
```


> ##   bbr_high_gain   bbr_drain_gain 

```
static const int bbr_high_gain  = BBR_UNIT * 2885 / 1000 + 1;
/* The pacing gain of 1/high_gain in BBR_DRAIN is calculated to typically drain
 * the queue created in BBR_STARTUP in a single round:
 */
static const int bbr_drain_gain = BBR_UNIT * 1000 / 2885;
```

```
>>> print(256* 2885 / 1000 + 1)
739.56
>>> print(256 * 1000 / 2885)
88.73483535528597
>>> 
```

```
>>> print(2885 / 1000 + 1)
3.885
>>> print(1000 / 2885)
0.3466204506065858
>>> print(1/2.89)
0.3460207612456747
>>> 
```

```
BBR 的 startup
链路上带宽跨度很大，从几个 bps 到上百 Gbps，所以 BBR 一开始也会以指数形式增大 BtlBw，每一个 RTT 下发送速度都增大 2/ln(2) 约 2.89 倍，从而在 O(log(BDP)) 个 RTT 内找到链路的 BtlBw，log 的底是 2。BBR 在发现提高发送速度但 deliveryRate 提高很小的时候标记 full_pipe，开始进入 Drain 阶段，将排队的数据包都消费完。BBR 能保证排队的数据最多为实际 BDP 的 1.89 倍。BBR 下并没有 ssthresh，即 CUBIC 那样增加到某个配置值后开始进入线性增加 CWND 阶段。

Drain 阶段就是把发送速率调整为 Startup 阶段的倒数。比如 Startup 阶段发送速度是 2.89，那 Drain 阶段发送速度是 1/ 2.89。BBR 会计算 inflights 数据包量，当与估计的 BDP 差不多的时候，BBR 进入 ProbeBW 状态，后续就在 ProbeBW 和 ProbeRTT 之间切换。

```

>> # log

```
[2657098.523112] mode 2 mode=2,min_rtt_us=70,full_bw=0,cycle_idx=0,pacing_gain=320,cwnd_gain=512,rtt_cnt=4822
[2657191.312006] mode 2 mode=0,min_rtt_us=136,full_bw=0,cycle_idx=0,pacing_gain=739,cwnd_gain=739,rtt_cnt=0
[2657191.363813] mode 2 mode=1,min_rtt_us=136,full_bw=1918801,cycle_idx=0,pacing_gain=88,cwnd_gain=739,rtt_cnt=9
[2657191.373793] mode 2 mode=2,min_rtt_us=136,full_bw=1918801,cycle_idx=7,pacing_gain=256,cwnd_gain=512,rtt_cnt=9
[2657201.319377] mode 2 mode=3,min_rtt_us=3149,full_bw=1918801,cycle_idx=0,pacing_gain=256,cwnd_gain=256,rtt_cnt=6487
[2657201.537785] mode 2 mode=2,min_rtt_us=38,full_bw=1918801,cycle_idx=7,pacing_gain=256,cwnd_gain=512,rtt_cnt=8895
[2657211.555625] mode 2 mode=3,min_rtt_us=17316,full_bw=1918801,cycle_idx=0,pacing_gain=256,cwnd_gain=256,rtt_cnt=15880
[2657211.768141] mode 2 mode=2,min_rtt_us=1184,full_bw=1918801,cycle_idx=7,pacing_gain=256,cwnd_gain=512,rtt_cnt=15897
[2657225.403476] mode 2 mode=3,min_rtt_us=13171,full_bw=1918801,cycle_idx=0,pacing_gain=256,cwnd_gain=256,rtt_cnt=18808
[2657225.625526] mode 2 mode=2,min_rtt_us=663,full_bw=1918801,cycle_idx=0,pacing_gain=320,cwnd_gain=512,rtt_cnt=18835
[2657238.768105] mode 2 mode=3,min_rtt_us=3129,full_bw=1918801,cycle_idx=0,pacing_gain=256,cwnd_gain=256,rtt_cnt=25872
[2657239.021140] mode 2 mode=2,min_rtt_us=143,full_bw=1918801,cycle_idx=5,pacing_gain=256,cwnd_gain=512,rtt_cnt=25927
[2657249.027536] mode 2 mode=3,min_rtt_us=852,full_bw=1918801,cycle_idx=0,pacing_gain=256,cwnd_gain=256,rtt_cnt=32222
[2657249.247267] mode 2 mode=2,min_rtt_us=90,full_bw=1918801,cycle_idx=5,pacing_gain=256,cwnd_gain=512,rtt_cnt=32360
```