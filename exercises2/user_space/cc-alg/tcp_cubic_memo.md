- [Paper](#paper)
- [Linux implementation](#linux-implementation)
- [Hystart](#hystart)
- [Tuning parameters](#tuning-parameters)

# Paper

use a cubic function to replace the linear window growth to improve performance in high BDP network, also achieved RTT fairness.

tcp-cubic is succssor of tcp-bic, tcp-bic use binary search to find the avialble bandwidth when congestion happens, and MAY enter an lieaner increasing state (Additive Increase)if it dropped too low, if it passed the maximum window, it goes into max probing state and will do the reverse and grow slowly initially and switch to additive increase if it can't find new maximum window nearby.

The problem with tcp-bic is that the growth function is too aggressive for short RTT or low speed network, and different phases (additive increase, binary search, max probing) add complexity in implememting and analyzing.

tcp-cubic is replacing the phase functions with a unified one function, which is a cubic function of the elapsed time from the last congestion event.

for the tcp-cubic function, we can start with `w=t^^3`, which gives the initial shape center at (0, 0), then we can move the center towards the right to K, which gives `w = (t-K)^^3`, and then move the center upwards to Wmax, then we have `w = (t-K)^^3 + Wmax`, and we can tune the shape via parameter C, the larger C, more aggressive on the steps, now we have **equation (1)** `w = C*(t-K)^^3 + Wmax`.

Say if now we have congestion and the congestion window drops from `Wmax` to `Wmax*(1 - beta)`, beta is the window decrease factor. And we assume this point is `(0, Wmax*(1 - beta)`, then K is actually the time it takes to ramp up back to Wmax. and we can get the **equation (2)**:

```
cwnd = Wmax - Wmax * beta = C * (0 - K)^^3 + Wmax
=>  Wmax * beta = C * K^^3 
=>  K = (Wmax * beta / C)^^(1/3)
```

That is how we get the eqaution (1) and (2) in the paper.

In each RTT period, we calcuate curent Window size using equation (1), `W_curr = W(t + RTT)`.

The current window might be to small compared with traditional TCP, and might cause friendly issue(actually no wrose than traditional TCP in anyway), so it will give an estimation of the traditional TCP congestion window after time t, and if the window calcualated using cubic function is smaller than that, it will use the traditional TCP congestion window.

To make it fairness with traditional TCP, we should get similar average window size. 

The average window size of AIMD additive increase (alpha) and multplicative decrease (beta) is give as `sqrt[alpha/2 * (2-beta)/beta / p] / RTT`, for tcp alpha = 1 and beta = 0.5, so we have average window size for traditional TCP `sqrt(3/2/p) / RTT`. Since now in use different beta (0.2) for cubic, to get similiar average window size, the alpha should be `3 * beta / (2 - beta)`, and we can caculate the window size if using AIMD with alpha and beta, given the elapsed time t since last decrease.

```
Wtcp = Wmax * (1-beta) + alpha * t / RTT = Wmax * (1-beta) + 3 * beta / (2-beta) * t / RTT
```

fast convergence is used to let new flow to get its fair share more quickly, the idea is that when loss event occurrs, if the Wmax keeps dropping (less than before), it will reduce more to apply a factor `1 - beta/2` (that is * 0.9), so that the flow have the plateau earlier to give other flows more chance to catch up.

# Linux implementation

At a first glance, the Linux implementation looks more complicate than the paper, the reason is that for kernel peformance, for example it convert all the float number operation into integer by scaling, and also the congestion avoidance implementation is actually increase the congestion window by 1 every N ACKs, it won't be able to increase by a float number calculated using the equation (1), we have to calculate a N to approximate that to achieve similar effect, and need to consider the effect of delay-ack. Another confusing thing is that the beta in linux implementation is actually 1-beta in the paper. The code also include the hystart slow start algorithm which is not in the cubic paper.

First, let's see how the code use the equations above, the first thing is when there is a packet drop, it will update `ssthresh` and `last_max_cwnd`, beta is 717 and BICTCP_BETA_SCALE is 1024, beta/BICTCP_BETA_SCALE = 0.7, so the `snd_ssthresh` will drop to `0.7 * snd_cwnd`. And `snd_cwnd` will be updated based on the state of TCP (todo add more details, see tcp_input.c)

If no fast convergence, `last_max_cwnd` will set to the current `snd_cwnd`, otherwise it will apply another factor: `(BICTCP_BETA_SCALE + beta) / (2 * BICTCP_BETA_SCALE) = (1+beta_before_scaled)/2 = (1+1-beta_in_paper)/2 = (2 - beta_in_paper)/2 = (1 - beta_in_paper/2`

```
static u32 bictcp_recalc_ssthresh(struct sock *sk)
{
	const struct tcp_sock *tp = tcp_sk(sk);
	struct bictcp *ca = inet_csk_ca(sk);

	ca->epoch_start = 0;	/* end of epoch */

	/* Wmax and fast convergence */
	if (tp->snd_cwnd < ca->last_max_cwnd && fast_convergence)
		ca->last_max_cwnd = (tp->snd_cwnd * (BICTCP_BETA_SCALE + beta))
			/ (2 * BICTCP_BETA_SCALE);
	else
		ca->last_max_cwnd = tp->snd_cwnd;

	ca->loss_cwnd = tp->snd_cwnd;

	return max((tp->snd_cwnd * beta) / BICTCP_BETA_SCALE, 2U);
}
```

When get ack, it will calculate a moving average of delayed acked packet count `delayed_ack`, `ACK_RATIO_SHIFT`(default to 4) is the moving average factor and `cnt` is the delayed acked packet count in current ACK. `delayed_ack` will be used later to update congestion window, in the paper it assumes that we update congestion window on each ACK per RTT, and delayed ack might gives longer interval per ACK, and we need compensate for that.

The following code is actuall doing the math `ratio = (15*ratio + sample) / 16`, `delayed_ack`(ratio) is initilized to be 16 times larger `ca->delayed_ack = 2 << ACK_RATIO_SHIFT;`, if the last value is X without scaling by 16 times (`ratio = 16 * X`), then `15 * ratio / 16 + cnt = 15 * 16 * X / 16 + cnt = 15 * X + cnt`, `15 * ratio / 16 + cnt` is what used in the code below, and if we check the above equation in reverse direction: moving average of the raw value X without scaling is `(15 * X + cnt) / 16 = (15 * 16 * X / 16 + cnt) / 16 = (15 * ratio / 16 + cnt) / 16`, that is we can calculate moving average X using the scaled value ratio, and we need `* 16` to convert the raw value to `ratio` for next iteration, and store that in `delayed_ack`, so the `delayed_ack` is actually the moving avarege of scaled value.

```
static void bictcp_acked(struct sock *sk, u32 cnt, s32 rtt_us)
{
    ......
                u32 ratio = ca->delayed_ack;
		ratio -= ca->delayed_ack >> ACK_RATIO_SHIFT;
		ratio += cnt;
		ca->delayed_ack = min(ratio, ACK_RATIO_LIMIT);
    ......
}
```

If the current `snd_cwnd` is no larger than `snd_ssthresh` it will enter into slow start, can either use the traditional slow start or hystart based on the settings, otherwise, it will enter into congestion avoidance, this is what cubic does most of its job:

If current `cwnd` is larger than `last_max_cwnd`, will reset the origin point to `cwnd`, and reset `bic_K` to zero, this is the max probing state. Else it's in steady state, and will compute new K, the estimate time needed to reach `last_max_cwnd`, we can rewrite equation 2 `cwnd = Wmax - Wmax * beta = C * (0 - K)^^3 + Wmax => K = [(Wmax - cwnd) / C]^^(1/3)`, this is the equation used in the code.

```
    if (ca->last_max_cwnd <= cwnd) {
			ca->bic_K = 0;
			ca->bic_origin_point = cwnd;
		} else {
			/* Compute new K based on
			 * (wmax-cwnd) * (srtt>>3 / HZ) / c * 2^(3*bictcp_HZ)
			 */
			ca->bic_K = cubic_root(cube_factor
					       * (ca->last_max_cwnd - cwnd));
			ca->bic_origin_point = ca->last_max_cwnd;
		}
```

should be notice that the unit of bic_K (same for the elapsed time since epoch) is not ms nor jiffies(HZ), it's using `BHZ = HZ*(2^^BICTCP_HZ) = HZ*(2^^10) = HZ*1024`, to avoid overflow when doing the math.

The C used is actually `bic_scale * 10 / 1024 =  41 * 10 / 1024 =  0.4`, and in order to convert the time to units of BHZ, `bic_K (HZ) = cubic_root(1/C * (last_max_cwnd - cwnd)) => bic_K (BHz) * 2^^BICTCP_HZ = 2^^BICTCP_HZ * cubic_root(1/C * (last_max_cwnd - cwnd)) = cubic_root((2^^BICTCP_HZ)^^3 * 1/C * (last_max_cwnd - cwnd)) = cubic_root(2^^(3*BICTCP_HZ) * 1/C * (last_max_cwnd - cwnd)) = cubic_root(2^^(3*BICTCP_HZ) * 2^^10/(bic_scale*10) * (last_max_cwnd - cwnd))`

`2^^(3*BICTCP_HZ) * 2^^10/(bic_scale*10) = 2^^(3*BICTCP_HZ+10)/(bic_scale*10) = 1ull << (10+3*BICTCP_HZ)/(bic_scale*10)`, this is how `cube_factor` is caculated.

```
cube_factor = 1ull << (10+3*BICTCP_HZ);
do_div(cube_factor, bic_scale * 10);
```

Then we will caculate the elapsed time plus one RTT in unit BHZ, and then caculate the target congestion window, for calculating delta of congestion window, `delta = (cube_rtt_scale * offs * offs * offs) >> (10+3*BICTCP_HZ);`, since the `offset` is in BHZ, to convert it back to HZ, we need `/ (2^^BICTCP_HZ)`, that is `>>BICTCP_HZ`, and we have multiplied 3 offset, so it's converting 'BHZ * BHZ * BHZ' back to 'HZ * HZ * HZ', that is `/(2^^BICTCP_HZ)^^3 = /(2^^(3*BICTCP_HZ) = >>(3*BICTCP_HZ)`, and `C = cube_rtt_scale/1024 = cube_rtt_scale >> 10`.

so `delta = C * offs * offs * offs >> (3+BICTCP_HZ) = cube_rtt_scale * offs * offs * offs >> (10+3*BICTCP_HZ)`

```
	t = ((tcp_time_stamp + msecs_to_jiffies(ca->delay_min>>3)
	      - ca->epoch_start) << BICTCP_HZ) / HZ;

	if (t < ca->bic_K)		/* t - K */
		offs = ca->bic_K - t;
	else
		offs = t - ca->bic_K;

	/* c/rtt * (t-K)^3 */
	delta = (cube_rtt_scale * offs * offs * offs) >> (10+3*BICTCP_HZ);
	if (t < ca->bic_K)                                	/* below origin*/
		bic_target = ca->bic_origin_point - delta;
	else                                                	/* above origin*/
		bic_target = ca->bic_origin_point + delta
```

Once we have `bic_target`, we can calculate the congestion window increment for each RTT, `tcp_cong_avoid_ai` is called for each ACK, and it will increase `snd_cwnd` by 1 on every `w` ACKs (that is increase by `1/w` on each ACK), for TCP reno `w` is the congestion window, that is increment by 1 for each RTT (receive `w` ACKs when `w` packets send out in one RTT).

```
/* In theory this is tp->snd_cwnd += 1 / tp->snd_cwnd (or alternative w) */
void tcp_cong_avoid_ai(struct tcp_sock *tp, u32 w)
{
	if (tp->snd_cwnd_cnt >= w) {
		if (tp->snd_cwnd < tp->snd_cwnd_clamp)
			tp->snd_cwnd++;
		tp->snd_cwnd_cnt = 0;
	} else {
		tp->snd_cwnd_cnt++;
	}
}
```

For cubic, we are expecting to increase the congestion window by `bic_target - cwnd` in the next round trip time, since we will send out `cwnd` packets in one RTT, if we don't consider delayed ack and assume ack per packet, then it's for each ACK the congestion window increases by `(bic_target - cwnd)/cwnd`, then `(bic_target - cwnd)/cwnd = 1/w => w = cwnd/ (bic_target - cwnd)`, which is exactly what we have for `ca->cnt` when `bic_target > cwnd`, if it's smaller, then the increment will be very small, `1/(100 * cwnd)` on each ACK, that is `1/100` on each RTT if no delay ACK, this is just a safety guard.

```
	/* cubic function - calc bictcp_cnt*/
	if (bic_target > cwnd) {
		ca->cnt = cwnd / (bic_target - cwnd);
	} else {
		ca->cnt = 100 * cwnd;              /* very small increment*/
	}
	
	...
	tcp_cong_avoid_ai(tp, ca->cnt);
```

And if we consider the delay ack, we should compensate for that b/c we may get less Ack when using delay ack, which means for each ACK we should increase more, that said if no delay ACK, we increase `1/ca->cnt` per ACK (w = ca->cnt), then with delay ACK, assume we have pkts_per_ack, we should increase `pkts_per_ack/ca->cnt` per ACK (w = ca->cnt/pktsPerAck . `delayed_ack` is a scaled value, then `pkts_per_ack = delayed_ack/16`

```
ca->cnt = (ca->cnt << ACK_RATIO_SHIFT) / ca->delayed_ack;
```

To avoid the case cubic is slower than TCP, it will estimated the congestion window of traditional TCP. In the paper we have the increament factor `alpha = 3 * beta_in_paper / (2-beta_in_paper)`, since in the code the beta is actual `1-beta_in_paper`, then `alpha = 3 * (1 - beta) / (1 + beta)`.

in the code, `beta_scale = 8 / alpha`, 8 is just another scaling factor which ensure beta_scale is integer, and for later use it will apply `>>3` which give `1 / alpha`.

```
beta_scale = 8*(BICTCP_BETA_SCALE+beta)/ 3 / (BICTCP_BETA_SCALE - beta);
```

When `tcp_friendliness` enabled, to simulate traditional TCP, the congestion window increase per RTT is `alpha`, so for each ACK, the increment `1/w` is `alpha/cwnd = 8/(beta_scale * cwnd)`, so we have `w = beta_scale * cwnd / 8 = beta_scale * cwnd >> 3`. That is every `w` ACK packets the congestion window increment by 1, give current has `ca->ack_cnt` acks in total, we can estimate `tcp_cwnd`.

once we have `tcp_cwnd`, if it's larger than current `cwnd`, we should increase cwnd by `delta = tcp_cwnd - cwnd`, and we use the max between `bic_target` and `tcp_cwnd`, the code is using smaller cnt but it's actually the same.

```
	/* TCP Friendly */
	if (tcp_friendliness) {
		u32 scale = beta_scale;
		delta = (cwnd * scale) >> 3;
		while (ca->ack_cnt > delta) {		/* update tcp cwnd */
			ca->ack_cnt -= delta;
			ca->tcp_cwnd++;
		}

		if (ca->tcp_cwnd > cwnd){	/* if bic is slower than tcp */
			delta = ca->tcp_cwnd - cwnd;
			max_cnt = cwnd / delta;
			if (ca->cnt > max_cnt)
				ca->cnt = max_cnt;
		}
	}
```

# Hystart

Linux implementation also use hystart slow start by default, hystart slow start is to exit early to avoid too many packet loss caused by slow start phase, and it utilize the packet train and delay to give hint when to stop slow start. Since for most TCP it's window based and data are send out in a burst in one congestion window, which means we can use those packets as packet train, suppose we send out N packets in the train, the time gap between the 1st and Nst packet is delta(N), then the bandwidth estimation will be `bw = (N-1) * packet_length / delta(N)`,  the network pipe capcaity (without buffer) is `K = bw * one_way_delay_min`, the data sent in a cwnd should be no larger than the pipe capacity.

```
(N-1) * packet_length <= bw * one_way_delay_min = (N-1) * packet_length / delta(N) * one_way_delay_min
=> delta(N) <= one_way_delay_min
```

That means we can measure the time gap of the train to know whether we are exceed the available bandwidth and enter into congestion avoidance. Since it's not easy to measure one way delay, half of RTT is used, and to avoid modification on both sides, ACK gap between Nst and 1st packet is used. 

Using RTT/2 as one way delay estimation won't make things worse,  exit too early will cause under untilization, `beta * (bw * one_way_delay_min)` can be used as the lowerbound for safety exit, for standard TCP, beta is 0.5, other variances has beta larger than 0.5;

suppose we have forward and backward delay `a` and `b`, if we use `RTT/2 = (a+b)/2` as one way delay, the BDP estimation `K' = bw * (a+b)/2` while `K =  bw * a`, `K'/K = (a + b) / 2 / a = 1/2 + b/2/a >= 1/2`, so we have `K' >= 0.5 * K`, that is to say if we use RTT/2 as estimation, the BDP estimations is no less than half of the real BDP, so we won't exit slow start before we reach 0.5 * BDP (threshold of the standard TCP slow start).  

And if it congestin window goes beyond `K + S`(S is network buffer size), which is the upper bound of safety exit, packet will be dropped, so we have `K'/K = 1/2 + b/2/a <= (K + S)/K => 1/2 + b/2/a <= 1 + S/K`, if `S=K`, then `1/2 + b/2/a <= 2 => b/a <= 3 => b <= 3a`, and there are less than 5% cases that has reverse path delay larger than forward path, which means in this case, the slow start will fallback to traditional slow start which overshoots and cause packet loss.

Using Ack may give larger time gap, which gives lower bandwidth estimation, and cause conservative behavior to exit slow start earlier. Delay Ack also affect the accuracy, so if there is significantly delay in the last ACK, that sample is filter out and mixed with next train.

And there may be cases that minimum RTT is not available, for example when multiple flows are competing, the idea is use delay increasing as an exit indicator, it measure the first a few packets of the train, and calculate the average RTT for that train, and compare the trian K and train K-1, if RTT(K) > RTT(K-1) + delta, then exit slow start.

The Linux implementation is slightly different and simpler than that in the paper.

Hystart will be triggered when the cwnd is larger than `hystart_low_window` (default to 16)

```
	/* hystart triggers when cwnd is larger than some threshold */
	if (hystart && tp->snd_cwnd <= tp->snd_ssthresh &&
	    tp->snd_cwnd >= hystart_low_window)
		hystart_update(sk, delay);
```

It will keep track to minimum delay(`delay_min`) in the tcp session so far. On each ACK, it will check if the time since last ACK is less than the threshold `hystart_ack_delta` to filter the invalid sample, if it's invalid, then in this round, it will not do ack train detection, and if the sample is valid, check if it goes beyond minRTT/2 (`delay_min` is scaled by 8, so `>>4` is actually minRTT/2), if so it will detect as `HYSTART_ACK_TRAIN` happens.

It also track minimum delay(`curr_delay`) among the first `HYSTART_MIN_SAMPLES` samples in each round (send cwnd packets in the burst train). It `curr_delay` is larger than `delay_min` more than a threshold value, which is `delay_min/2`, clamp to `[4, 16] ms`, then it will detect as `HYSTART_DELAY`.

Either one detected will cause the slow start exit by setting the `ssthresh` to current congeston window `snd_cwnd`.

```
static void hystart_update(struct sock *sk, u32 delay)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct bictcp *ca = inet_csk_ca(sk);

	if (!(ca->found & hystart_detect)) {
		u32 now = bictcp_clock();

		/* first detection parameter - ack-train detection */
		if ((s32)(now - ca->last_ack) <= hystart_ack_delta) {
			ca->last_ack = now;
			if ((s32)(now - ca->round_start) > ca->delay_min >> 4)
				ca->found |= HYSTART_ACK_TRAIN;
		}

		/* obtain the minimum delay of more than sampling packets */
		if (ca->sample_cnt < HYSTART_MIN_SAMPLES) {
			if (ca->curr_rtt == 0 || ca->curr_rtt > delay)
				ca->curr_rtt = delay;

			ca->sample_cnt++;
		} else {
			if (ca->curr_rtt > ca->delay_min +
			    HYSTART_DELAY_THRESH(ca->delay_min>>4))
				ca->found |= HYSTART_DELAY;
		}
		/*
		 * Either one of two conditions are met,
		 * we exit from slow start immediately.
		 */
		if (ca->found & hystart_detect)
			tp->snd_ssthresh = tp->snd_cwnd;
	}
}
```

# Tuning parameters

Linux cubic implementation has some parameters which can be used to tune the algorithm, by set different values for files (filename is the same with parameter name) in `/sys/module/tcp_cubic/parameters/`

- fast_convergence: default is on. enable fast convergence will degrade more when consecutive downgrade, this is used for let new comers to get fair share faster. 

- beta: beta is used for multiplicative decrease, larger value will decrease less, default is 0.7, that is 0.7 * cwnd when loss happens. beta will also affect the fairness with standard TCP. beta is scaled by 1024.

- initial_ssthresh: use to set the initial ssthresh value, only used at the beginning whe hystart not enabled.

- bic_scale: bi_scale is used to tune the cubic funtion curve (C in the equation 1), large beta will increase more aggressively, default is 0.4. bic_scale also affect teh fairness with standard TCP. bic_scale is scaled by 1024.

- tcp_friendliness: used to ensure in any case cubic is no worse than standard TCP. default is on.

- hystart: enable hystart, which exit slow start earlier based on delay to avoid too much packet loss.

- hystart_detect: enable which methods are used for hystart detection, default is enable both packet train and delay increase.

- hystart_low_window: when cwnd is larger than hystart_low_window, will start hystart slow start. default is 16.

- hystart_ack_delta: threshold value to filter delay ACK, default is 2.

```
module_param(fast_convergence, int, 0644);
MODULE_PARM_DESC(fast_convergence, "turn on/off fast convergence");
module_param(beta, int, 0644);
MODULE_PARM_DESC(beta, "beta for multiplicative increase");
module_param(initial_ssthresh, int, 0644);
MODULE_PARM_DESC(initial_ssthresh, "initial value of slow start threshold");
module_param(bic_scale, int, 0444);
MODULE_PARM_DESC(bic_scale, "scale (scaled by 1024) value for bic function (bic_scale/1024)");
module_param(tcp_friendliness, int, 0644);
MODULE_PARM_DESC(tcp_friendliness, "turn on/off tcp friendliness");
module_param(hystart, int, 0644);
MODULE_PARM_DESC(hystart, "turn on/off hybrid slow start algorithm");
module_param(hystart_detect, int, 0644);
MODULE_PARM_DESC(hystart_detect, "hyrbrid slow start detection mechanisms"
		 " 1: packet-train 2: delay 3: both packet-train and delay");
module_param(hystart_low_window, int, 0644);
MODULE_PARM_DESC(hystart_low_window, "lower bound cwnd for hybrid slow start");
module_param(hystart_ack_delta, int, 0644);
MODULE_PARM_DESC(hystart_ack_delta, "spacing between ack's indicating train (msecs)");
```