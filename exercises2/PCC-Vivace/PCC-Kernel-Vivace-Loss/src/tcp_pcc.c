/* PCC Vivace-Loss */
#include <linux/module.h>
#include <net/tcp.h>
#include <linux/random.h>

/* Probing changes rate by 5% up and down of current rate. */
static const u32 pcc_probing_eps = 5;
static const u32 pcc_probing_eps_part = 100;

static const u64 pcc_factor = 1000; /* scale for fractions, utilities, gradients, ... */

static const u64 pcc_min_rate = 1024u;
static const u32 pcc_min_rate_packets_per_rtt = 2;
static const u32 pcc_interval_per_packet = 50;
static const u32 pcc_alpha = 100;

static const u32 pcc_grad_step_size = 25; /* defaults step size for gradient ascent */
static const u32 pcc_max_swing_buffer = 2; /* number of RTTs to dampen gradient ascent */

static const u32 pcc_lat_infl_filter = 30; /* latency inflation below 3% is ignored */

/* Rates must differ by at least 2% or gradients are very noisy. */
static const u32 pcc_min_rate_diff_ratio_for_grad = 20;

static const u32 pcc_min_change_bound = 100; /* first rate change is at most 10% of rate */
static const u32 pcc_change_bound_step = 70; /* consecutive rate changes can go up by 7% */
static const u32 pcc_min_amp = 2; /* starting amplifier for gradient ascent step size */

/* The number of past monitor intervals used for decision making.*/
static const u32 pcc_intervals = 4;

#define USE_PROBING

enum PCC_DECISION {
	PCC_RATE_UP,
	PCC_RATE_DOWN,
	PCC_RATE_STAY,
};

struct pcc_interval {
	u64 rate;		/* sending rate of this interval, bytes/sec */
	
	s64 recv_start; /* timestamps for when interval was waiting for acks */
	s64 recv_end;

	s64 send_start; /* timestamps for when interval data was being sent */
	s64 send_end;

	s64 start_rtt; /* smoothed RTT at start and end of this interval */
	s64 end_rtt;

	u32 packets_sent_base; /* packets sent before this interval started */
	u32 packets_ended;

	s64 utility; /* observed utility of this interval */
	u32 lost; /* packets sent during this interval that were lost */
	u32 delivered; /* packets sent during this interval that were delivered */
};

static int id = 0;
struct pcc_data {
	struct pcc_interval *intervals; /* containts stats for 1 RTT */
	int send_index; /* index of interval currently being sent */
	int recive_index; /* index of interval currently receiving acks */

	s64 rate; /* current sending rate */
	s64 last_rate; /* previous sending rate */

	/* utility function pointer (can be loss- or latency-based) */
	void (*util_func)(struct pcc_data *, struct pcc_interval *, struct sock *);

	bool start_mode;
	bool moving; /* using gradient ascent to move to a new rate? */
	bool loss_state;

	bool wait;
	enum PCC_DECISION last_decision; /* most recent rate change direction */
	u32 lost_base; /* previously lost packets */
	u32 delivered_base; /* previously delivered packets */

	// debug helpers
	int id;
	int decisions_count;

	u32 packets_sent;
	u32 packets_counted;
	u32 spare;

	s32 amplifier; /* multiplier on the current step size */
	s32 swing_buffer; /* number of RTTs left before step size can grow */
	s32 change_bound; /* maximum change as a proportion of the current rate */
};

/*********************
 * Getters / Setters *
 * ******************/
static u32 pcc_get_rtt(struct tcp_sock *tp)
{
	/* Get initial RTT - as measured by SYN -> SYN-ACK.
	 * If information does not exist - use 1ms as a "LAN RTT".
	 */
	if (tp->srtt_us) {
		return max(tp->srtt_us >> 3, 1U);
	} else {
		return USEC_PER_MSEC;
	}
}

/* Initialize cwnd to support current pacing rate (but not less then 4 packets)
 */
static void pcc_set_cwnd(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);

		u64 cwnd = sk->sk_pacing_rate;
		cwnd *= pcc_get_rtt(tcp_sk(sk));
	cwnd /= tp->mss_cache;
	
		cwnd /= USEC_PER_SEC;
		cwnd *= 2;

	cwnd = max(4ULL, cwnd);
		cwnd = min((u32)cwnd, tp->snd_cwnd_clamp); /* apply cap */
		tp->snd_cwnd = cwnd;
}


/* was the pcc struct fully inited */
bool pcc_valid(struct pcc_data *pcc)
{
	return (pcc && pcc->intervals && pcc->intervals[0].rate);
}

/******************
 * Intervals init *
 * ****************/

 /* Set the target rates of all intervals and reset statistics. */
static void pcc_setup_intervals_probing(struct pcc_data *pcc)
{
	u64 rate_low, rate_high;
	char rand;
	int i;

	get_random_bytes(&rand, 1);
	rate_high = pcc->rate * (pcc_probing_eps_part + pcc_probing_eps);
	rate_low = pcc->rate * (pcc_probing_eps_part - pcc_probing_eps);

	rate_high /= pcc_probing_eps_part;
	rate_low /= pcc_probing_eps_part;

	for (i = 0; i < pcc_intervals; i += 2) {
		if ((rand >> (i / 2)) & 1) {
			pcc->intervals[i].rate = rate_low;
			pcc->intervals[i + 1].rate = rate_high;
		} else {
			pcc->intervals[i].rate = rate_high;
			pcc->intervals[i + 1].rate = rate_low;
		}

		pcc->intervals[i].packets_sent_base = 0;
		pcc->intervals[i + 1].packets_sent_base = 0;
	}

	pcc->send_index = 0;
	pcc->recive_index = 0;
	pcc->wait = false;
}

/* Reset statistics and set the target rate for just one monitor interval */
static void pcc_setup_intervals_moving(struct pcc_data *pcc)
{
	pcc->intervals[0].packets_sent_base = 0;
	pcc->intervals[0].rate = pcc->rate;
	pcc->send_index = 0;
	pcc->recive_index = 0;
	pcc->wait = false;
}

/* Set the pacing rate and cwnd base on the currently-sending interval */
static void start_interval(struct sock *sk, struct pcc_data *pcc)
{
	u64 rate = pcc->rate;
	struct pcc_interval *interval;

	if (!pcc->wait) {
		interval = &pcc->intervals[pcc->send_index];
		interval->packets_ended = 0;
		interval->lost = 0;
		interval->delivered = 0;
		interval->packets_sent_base = tcp_sk(sk)->data_segs_out;
		interval->packets_sent_base = max(interval->packets_sent_base, 1U);
		interval->send_start = tcp_sk(sk)->tcp_mstamp;
		rate = interval->rate;
	}

	rate = max(rate, pcc_min_rate);
	rate = min(rate, sk->sk_max_pacing_rate);
	sk->sk_pacing_rate = rate;
	pcc_set_cwnd(sk);
}

/************************
 * Utility and decisions *
 * **********************/
#define PCC_LOSS_MARGIN 5
#define PCC_MAX_LOSS 10
/* get x = number * pcc_factor, return (e^number)*pcc_factor */
static u32 pcc_exp(s32 x)
{
	s64 temp = pcc_factor;
	s64 e = pcc_factor;
	int i;

	for (i = 1; temp != 0; i++) {
		temp *= x;
		temp /= i;
		temp /= pcc_factor;
		e += temp;
	}
	return e;
}

/* Calculate the graident of utility w.r.t. sending rate, but only if the rates
 * are far enough apart for the measurment to have low noise.
 */
static s64 pcc_calc_util_grad(s64 rate_1, s64 util_1, s64 rate_2, s64 util_2) {
	s64 rate_diff_ratio = (pcc_factor * (rate_2 - rate_1)) / rate_1;
	if (rate_diff_ratio < pcc_min_rate_diff_ratio_for_grad && 
		rate_diff_ratio > -1 * pcc_min_rate_diff_ratio_for_grad)
		return 0;

	return (pcc_factor * pcc_factor * (util_2 - util_1)) / (rate_2 - rate_1);
}

static void pcc_calc_utility_vivace_latency(struct pcc_data *pcc, struct pcc_interval *interval, struct sock *sk) {
	s64 loss_ratio, delivered, lost, mss, rate, throughput, util;
	s64 lat_infl = 0;
    s64 rtt_diff;
    s64 rtt_diff_thresh = 0;
	s64 send_dur = interval->send_end - interval->send_start;
	s64 recv_dur = interval->recv_end - interval->recv_start;

	lost = interval->lost;
	delivered = interval->delivered;
	mss = tcp_sk(sk)->mss_cache;
	rate = interval->rate;
	throughput = 0;
	if (recv_dur > 0)
		throughput = (USEC_PER_SEC * delivered * mss) / recv_dur;
	if (delivered == 0) {
        printk(KERN_INFO "No packets delivered\n");
		//interval->utility = S64_MIN;
		interval->utility = 0;
		return;
	}

	rtt_diff = interval->end_rtt - interval->start_rtt;
    if (throughput > 0)
	    rtt_diff_thresh = (2 * USEC_PER_SEC * mss) / throughput;
	if (send_dur > 0)
		lat_infl = (pcc_factor * rtt_diff) / send_dur;
	
	printk(KERN_INFO
		"%d ucalc: lat (%lld->%lld) lat_infl %lld\n",
		 pcc->id, interval->start_rtt / USEC_PER_MSEC, interval->end_rtt / USEC_PER_MSEC,
		 lat_infl);

	if (rtt_diff < rtt_diff_thresh && rtt_diff > -1 * rtt_diff_thresh)
		lat_infl = 0;

	if (lat_infl < pcc_lat_infl_filter && lat_infl > -1 * pcc_lat_infl_filter)
		lat_infl = 0;
	
	if (lat_infl < 0 && pcc->start_mode)
		lat_infl = 0;

	/* loss rate = lost packets / all packets counted*/
	loss_ratio = (lost * pcc_factor) / (lost + delivered);

    if (pcc->start_mode && loss_ratio < 100)
        loss_ratio = 0;

	util = /* int_sqrt((u64)rate)*/ rate - (rate * (900 * lat_infl + 11 * loss_ratio)) / pcc_factor;

	printk(KERN_INFO
		"%d ucalc: rate %lld sent %u delv %lld lost %lld lat (%lld->%lld) util %lld rate %lld thpt %lld\n",
		 pcc->id, rate, interval->packets_ended - interval->packets_sent_base,
		 delivered, lost, interval->start_rtt / USEC_PER_MSEC, interval->end_rtt / USEC_PER_MSEC, util, rate, throughput);
	interval->utility = util;

}

static void pcc_calc_utility_vivace_loss(struct pcc_data *pcc, struct pcc_interval *interval, struct sock *sk) {
	s64 loss_ratio, delivered, lost, mss, rate, throughput, util;
	s64 lat_infl = 0;
    s64 rtt_diff;
    s64 rtt_diff_thresh = 0;
	s64 send_dur = interval->send_end - interval->send_start;
	s64 recv_dur = interval->recv_end - interval->recv_start;

	lost = interval->lost;
	delivered = interval->delivered;
	mss = tcp_sk(sk)->mss_cache;
	rate = interval->rate;
	throughput = 0;
	if (recv_dur > 0)
		throughput = (USEC_PER_SEC * delivered * mss) / recv_dur;
	if (delivered == 0) {
        printk(KERN_INFO "No packets delivered\n");
		//interval->utility = S64_MIN;
		interval->utility = 0;
		return;
	}

	rtt_diff = interval->end_rtt - interval->start_rtt;
    if (throughput > 0)
	    rtt_diff_thresh = (2 * USEC_PER_SEC * mss) / throughput;
	if (send_dur > 0)
		lat_infl = (pcc_factor * rtt_diff) / send_dur;
	
	printk(KERN_INFO
		"%d ucalc: lat (%lld->%lld) lat_infl %lld\n",
		 pcc->id, interval->start_rtt / USEC_PER_MSEC, interval->end_rtt / USEC_PER_MSEC,
		 lat_infl);

	if (rtt_diff < rtt_diff_thresh && rtt_diff > -1 * rtt_diff_thresh)
		lat_infl = 0;

	if (lat_infl < pcc_lat_infl_filter && lat_infl > -1 * pcc_lat_infl_filter)
		lat_infl = 0;
	
	if (lat_infl < 0 && pcc->start_mode)
		lat_infl = 0;

	/* loss rate = lost packets / all packets counted*/
	loss_ratio = (lost * pcc_factor) / (lost + delivered);

    if (pcc->start_mode && loss_ratio < 100)
        loss_ratio = 0;

	util = /* int_sqrt((u64)rate)*/ rate - (rate * (11 * loss_ratio)) / pcc_factor;

	printk(KERN_INFO
		"%d ucalc: rate %lld sent %u delv %lld lost %lld lat (%lld->%lld) util %lld rate %lld thpt %lld\n",
		 pcc->id, rate, interval->packets_ended - interval->packets_sent_base,
		 delivered, lost, interval->start_rtt / USEC_PER_MSEC, interval->end_rtt / USEC_PER_MSEC, util, rate, throughput);
	interval->utility = util;

}
static void pcc_calc_utility_allegro(struct pcc_data *pcc, struct pcc_interval *interval, struct sock *sk)
{
	s64 loss_ratio, delivered, lost, mss, rate, throughput, util;

	lost = interval->lost;
	delivered = interval->delivered;
	mss = tcp_sk(sk)->mss_cache;
	rate = interval->rate;
	throughput = 0;
	if (interval->recv_start < interval->recv_end)
		throughput = (USEC_PER_SEC * delivered * mss) / (interval->recv_end - interval->recv_start);
	if (!(lost+ delivered)) {
		interval->utility = S64_MIN;
		return;
	}

	/* loss rate = lost packets / all packets counted *100 * FACTOR_1 */
	loss_ratio = (lost * pcc_factor * pcc_alpha) / (lost+ delivered);
	
	/* util = delivered rate / (1 + e^(100*loss_rate)) - lost_ratio * rate
	 */
	util = loss_ratio- (PCC_LOSS_MARGIN * pcc_factor);
	if (util < PCC_MAX_LOSS*pcc_factor)
		util = (throughput /* rate */ * pcc_factor) / (pcc_exp(util) + pcc_factor);
	else
		util = 0;

	/* util *= goodput */
	util *= (pcc_factor * pcc_alpha) - loss_ratio;
	util /= pcc_factor * pcc_alpha;
	/* util -= "wasted rate" */
	util -= (rate * loss_ratio) / (pcc_alpha * pcc_factor);

	printk(KERN_INFO
		"rate %lld sent %u delv %lld lost %lld util %lld\n",
		 rate, interval->packets_ended - interval->packets_sent_base,
		 delivered, lost, util);
	interval->utility = util;
}

static enum PCC_DECISION
pcc_get_decision(struct pcc_data *pcc, u32 new_rate)
{
	if (pcc->rate == new_rate)
		return PCC_RATE_STAY;

	return pcc->rate < new_rate ? PCC_RATE_UP : PCC_RATE_DOWN;
}

static u32 pcc_decide_rate(struct pcc_data *pcc)
{
	bool run_1_res, run_2_res, did_agree;

	run_1_res = pcc->intervals[0].utility > pcc->intervals[1].utility;
	run_2_res = pcc->intervals[2].utility > pcc->intervals[3].utility;

	/* did_agree: was the 2 sets of intervals with the same result */
	did_agree = !((run_1_res == run_2_res) ^ 
			(pcc->intervals[0].rate == pcc->intervals[2].rate));

	if (did_agree) {
		if (run_2_res) {
			pcc->last_rate = pcc->intervals[2].rate;
			pcc->intervals[0].utility = pcc->intervals[2].utility;
		} else {
			pcc->last_rate = pcc->intervals[3].rate;
			pcc->intervals[0].utility = pcc->intervals[3].utility;
		}
		return run_2_res ? pcc->intervals[2].rate :
				   pcc->intervals[3].rate;
	} else {
		return pcc->rate;
	}
}

static void pcc_decide(struct pcc_data *pcc, struct sock *sk)
{
	struct pcc_interval *interval;
	u32 new_rate;
	int i;

	for (i = 0; i < pcc_intervals; i++ ) {
		interval = &pcc->intervals[i];
		(*pcc->util_func)(pcc, interval, sk);
	}

	new_rate = pcc_decide_rate(pcc);

	if (new_rate != pcc->rate) {
		printk(KERN_INFO "%d decide: on new rate %d %d (%d)\n",
			   pcc->id, pcc->rate < new_rate, new_rate,
			   pcc->decisions_count);
		pcc->moving = true;
	    pcc_setup_intervals_moving(pcc);
	} else {
		printk(KERN_INFO "%d decide: stay %d (%d)\n", pcc->id,
			pcc->rate, pcc->decisions_count);
	    pcc_setup_intervals_probing(pcc);
	}

	pcc->rate = new_rate;
	start_interval(sk, pcc);
	pcc->decisions_count++;
}

/* Take larger steps if we keep changing rate in the same direction, otherwise
 * reset to take smaller steps again.
 */
static void pcc_update_step_params(struct pcc_data *pcc, s64 step) {
	if ((step > 0) == (pcc->rate > pcc->last_rate)) {
		if (pcc->swing_buffer > 0)
			pcc->swing_buffer--;
		else
			pcc->amplifier++;
	} else {
		pcc->swing_buffer = min(pcc->swing_buffer + 1, pcc_max_swing_buffer);
		pcc->amplifier = pcc_min_amp;
		pcc->change_bound = pcc_min_change_bound;
	}
}

/* Bound any rate change as a proportion of the current rate, so large gradients
 * don't drasitcally change sending rate.
 */
static s64 pcc_apply_change_bound(struct pcc_data *pcc, s64 step) {
	s32 step_sign;
	s64 change_ratio;
	if (pcc->rate == 0)
		return step;

	step_sign = step > 0 ? 1 : -1;
	step *= step_sign;
	change_ratio = (pcc_factor * step) / pcc->rate;

	if (change_ratio > pcc->change_bound) {
		step = (pcc->rate * pcc->change_bound) / pcc_factor;
		printk("bound %u rate %u step %lld\n", pcc->change_bound, pcc->rate, step);
		pcc->change_bound += pcc_change_bound_step;
	} else {
		pcc->change_bound = pcc_min_change_bound;
	}
	return step_sign * step;
}

/* Choose up/down rate changes based on utility gradient */
static u32 pcc_decide_rate_moving(struct sock *sk, struct pcc_data *pcc)
{
	struct pcc_interval *interval = &pcc->intervals[0];
	s64 utility, prev_utility;
	s64 grad, step, min_step;

	prev_utility = interval->utility;
	(*pcc->util_func)(pcc, interval, sk);
	utility = interval->utility;
	
	printk(KERN_INFO "%d mv: pr %u pu %lld nr %u nu %lld\n",
		   pcc->id, pcc->last_rate, prev_utility, pcc->rate, utility);

	grad = pcc_calc_util_grad(pcc->rate, utility, pcc->last_rate, prev_utility);

	step = grad * pcc_grad_step_size; /* gradient ascent */
	pcc_update_step_params(pcc, step); /* may accelerate/decellerate changes */
	step *= pcc->amplifier; 
    step /= pcc_factor;
	step = pcc_apply_change_bound(pcc, step);

	/* We need our step size to be large enough that we can compute the gradient
	 * with low noise
	 */
	min_step = (pcc->rate * pcc_min_rate_diff_ratio_for_grad) / pcc_factor;
	min_step *= 11; /* step slightly larger than the minimum */
	min_step /= 10;
	if (step >= 0 && step < min_step)
		step = min_step;
	else if (step < 0 && step > -1 * min_step)
		step = -1 * min_step;

	printk(KERN_INFO "%d mv: grad %lld step %lld amp %d min_step %lld\n",
		   pcc->id, grad, step, pcc->amplifier, min_step);

	return pcc->rate + step;
}

/* Choose new direction and update state from the moving state.*/
static void pcc_decide_moving(struct sock *sk, struct pcc_data *pcc)
{
	s64 new_rate = pcc_decide_rate_moving(sk, pcc);
	enum PCC_DECISION decision = pcc_get_decision(pcc, new_rate);
	enum PCC_DECISION last_decision = pcc->last_decision;
    s64 packet_min_rate = (USEC_PER_SEC * pcc_min_rate_packets_per_rtt *
        tcp_sk(sk)->mss_cache) / pcc_get_rtt(tcp_sk(sk));
    new_rate = max(new_rate, packet_min_rate);
	pcc->last_rate = pcc->rate;
	printk(KERN_INFO "%d moving: new rate %lld (%d) old rate %d\n",
		   pcc->id, new_rate,
		   pcc->decisions_count, pcc->last_rate);
	pcc->rate = new_rate;
	if (decision != last_decision) {

#ifdef USE_PROBING
		pcc->moving = false;
		pcc_setup_intervals_probing(pcc);
#else
		pcc_setup_intervals_moving(pcc);
#endif
	} else {
		pcc_setup_intervals_moving(pcc);
	}

	start_interval(sk, pcc);
}

/* Double target rate until the link utility doesn't increase accordingly. Then,
 * cut the rate in half and change to the gradient ascent moving stage.
 */
static void pcc_decide_slow_start(struct sock *sk, struct pcc_data *pcc)
{
	struct pcc_interval *interval = &pcc->intervals[0];
	s64 utility, prev_utility, adjust_utility, prev_adjust_utility, tmp_rate;
	u32 extra_rate;

	prev_utility = interval->utility;
	(*pcc->util_func)(pcc, interval, sk);
	utility = interval->utility;

	/* The new utiltiy should be at least 75% of the expected utility given
	 * a significant increase. If the utility isn't as high as expected, then
	 * we end slow start.
	 */
	adjust_utility = utility * (utility > 0 ? 1000 : 750) / pcc->rate;
	prev_adjust_utility = prev_utility * (prev_utility > 0 ? 750 : 1000) /
				pcc->last_rate;

	printk(KERN_INFO "%d: start mode: r %lld u %lld pr %lld pu %lld\n",
		pcc->id, pcc->rate, utility, pcc->last_rate, prev_utility);
	//if (utility > prev_utility) {
	if (adjust_utility > prev_adjust_utility) {
		pcc->last_rate = pcc->rate;
		extra_rate = pcc->intervals[0].delivered *
				 tcp_sk(sk)->mss_cache;
		extra_rate = min(extra_rate, pcc->rate / 2);

		pcc->rate += extra_rate; //extra_rate;
		interval->utility = utility;
		interval->rate = pcc->rate;
		pcc->send_index = 0;
		pcc->recive_index = 0;
		pcc->wait = false;
	} else {
		tmp_rate = pcc->last_rate;
		pcc->last_rate = pcc->rate;
		pcc->rate = tmp_rate;
		pcc->start_mode = false;
		printk(KERN_INFO "%d: start mode ended\n", pcc->id);

#ifdef USE_PROBING
        pcc_setup_intervals_probing(pcc);
#else
		pcc->moving = true;
		pcc_setup_intervals_moving(pcc);
#endif
    }
	start_interval(sk, pcc);
}

/**************************
 * intervals & sample:
 * was started, was ended,
 * find interval per sample
 * ************************/
bool send_interval_ended(struct pcc_interval *interval, struct tcp_sock *tsk,
			 struct pcc_data *pcc)
{
	int packets_sent = tsk->data_segs_out - interval->packets_sent_base;

	if (packets_sent < pcc_interval_per_packet)
		return false;

	if (pcc->packets_counted > interval->packets_sent_base ) {
		interval->packets_ended = tsk->data_segs_out;
		return true;
	}
	return false;
}

bool recive_interval_ended(struct pcc_interval *interval,
			   struct tcp_sock *tsk, struct pcc_data *pcc)
{

	return interval->packets_ended && interval->packets_ended - 10 < pcc->packets_counted;
}

/* Start the next interval's sending stage. If there is no interval scheduled
 * to send (we have enough for probing, or we are in slow start or moving),
 * then we will be maintaining our rate while we wait for acks.
 */
static void start_next_send_interval(struct sock *sk, struct pcc_data *pcc)
{
	pcc->send_index++;
	if (pcc->send_index == pcc_intervals || pcc->start_mode || pcc->moving) {
		pcc->wait = true;
	}

	start_interval(sk, pcc);
}

/* Update the receiving time window and the number of packets lost/delivered
 * based on socket statistics.
 */
static void
pcc_update_interval(struct pcc_interval *interval,	struct pcc_data *pcc,
		struct sock *sk)
{
	interval->recv_end = tcp_sk(sk)->tcp_mstamp;
	interval->end_rtt = tcp_sk(sk)->srtt_us >> 3;
	if (interval->lost + interval->delivered == 0) {
		interval->recv_start = tcp_sk(sk)->tcp_mstamp;
		interval->start_rtt = tcp_sk(sk)->srtt_us >> 3;
	}

	interval->lost += tcp_sk(sk)->lost - pcc->lost_base;
	interval->delivered += tcp_sk(sk)->delivered - pcc->delivered_base;
}

/* Updates the PCC model */
static void pcc_process(struct sock *sk)
{
	struct pcc_data *pcc = inet_csk_ca(sk);
	struct tcp_sock *tsk = tcp_sk(sk);
	struct pcc_interval *interval;
	int index;
	u32 before;

	if (!pcc_valid(pcc))
		return;

	pcc_set_cwnd(sk);
	if (pcc->loss_state)
		goto end;
	if (!pcc->wait) {
		interval = &pcc->intervals[pcc->send_index];
		if (send_interval_ended(interval, tsk, pcc)) {
			interval->send_end = tcp_sk(sk)->tcp_mstamp;
			start_next_send_interval(sk, pcc);
		}
	}

	index = pcc->recive_index;
	interval = &pcc->intervals[index];
	before = pcc->packets_counted;
	pcc->packets_counted =	tsk->delivered + tsk->lost - pcc->spare;

	if (!interval->packets_sent_base)
		goto end;

	if (before > 10 + interval->packets_sent_base) {
		pcc_update_interval(interval, pcc, sk);
	}
	if (recive_interval_ended(interval, tsk, pcc)) {
		pcc->recive_index++;
		if (pcc->start_mode)
			pcc_decide_slow_start(sk, pcc);
		else if (pcc->moving)
			pcc_decide_moving(sk, pcc);
		else if (pcc->recive_index == pcc_intervals)
			pcc_decide(pcc, sk);
	}

end:
	pcc->lost_base = tsk->lost;
	pcc->delivered_base = tsk->delivered;
}

static void pcc_process_sample(struct sock *sk, const struct rate_sample *rs)
{
	pcc_process(sk);
}

static void pcc_init(struct sock *sk)
{
	struct pcc_data *pcc = inet_csk_ca(sk);

	pcc->intervals = kzalloc(sizeof(struct pcc_interval) *pcc_intervals*2,
				 GFP_KERNEL);
	if (!pcc->intervals) {
		printk(KERN_INFO "init fails\n");
		return;
	}

	id++;
	pcc->id = id;
	pcc->amplifier = pcc_min_amp;
	pcc->swing_buffer = 0;
	pcc->change_bound = pcc_min_change_bound;
	pcc->rate = pcc_min_rate*512;
	pcc->last_rate = pcc_min_rate*512;
	tcp_sk(sk)->snd_ssthresh = TCP_INFINITE_SSTHRESH;
	pcc->start_mode = true;
	pcc->moving = false;
	pcc->intervals[0].utility = S64_MIN;

	//pcc->util_func = &pcc_calc_utility_vivace_latency;
	pcc->util_func = &pcc_calc_utility_vivace_loss;
	//pcc->util_func = &pcc_calc_utility_allegro;

	pcc_setup_intervals_probing(pcc);
	start_interval(sk,pcc);
	cmpxchg(&sk->sk_pacing_status, SK_PACING_NONE, SK_PACING_NEEDED);
}

static void pcc_release(struct sock *sk)
{
	struct pcc_data *pcc = inet_csk_ca(sk);

	kfree(pcc->intervals);
}

/* PCC does not need to undo the cwnd since it does not
 * always reduce cwnd on losses (see pcc_main()). Keep it for now.
 */
static u32 pcc_undo_cwnd(struct sock *sk)
{
	return tcp_sk(sk)->snd_cwnd;
}

static u32 pcc_ssthresh(struct sock *sk)
{
	return TCP_INFINITE_SSTHRESH; /* PCC does not use ssthresh */
}

static void pcc_set_state(struct sock *sk, u8 new_state)
{
	struct pcc_data *pcc = inet_csk_ca(sk);
	s32 spare;

	if (!pcc_valid(pcc))
		return;

	if (pcc->loss_state && new_state != 4) {
		spare = tcp_sk(sk)->delivered + tcp_sk(sk)->lost+
			tcp_packets_in_flight(tcp_sk(sk));
		spare -= tcp_sk(sk)->data_segs_out;
		spare -= pcc->spare;
		pcc->spare+= spare;
		printk(KERN_INFO "%d loss ended: spare %d\n", pcc->id, spare);

		pcc->loss_state = false;
		pcc_setup_intervals_probing(pcc);
		start_interval(sk, pcc);
	}
	else if (!pcc->loss_state && new_state	== 4) {
		printk(KERN_INFO "%d loss: started\n", pcc->id);
		pcc->loss_state = true;
		pcc->wait = true;
		start_interval(sk, pcc);
	}
}

static void pcc_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
}

static void pcc_pkts_acked(struct sock *sk, const struct ack_sample *acks)
{
}

static void pcc_ack_event(struct sock *sk, u32 flags)
{
}

static void pcc_cwnd_event(struct sock *sk, enum tcp_ca_event event)
{
}

static struct tcp_congestion_ops tcp_pcc_cong_ops __read_mostly = {
	.flags = TCP_CONG_NON_RESTRICTED,
	.name = "pcc",
	.owner = THIS_MODULE,
	.init = pcc_init,
	.release = pcc_release,
	.cong_control = pcc_process_sample, // congestion control hook
	.undo_cwnd = pcc_undo_cwnd, // static windows
	.ssthresh = pcc_ssthresh,
	.set_state	= pcc_set_state,
	.cong_avoid = pcc_cong_avoid,
	.pkts_acked = pcc_pkts_acked,
	.in_ack_event = pcc_ack_event,
	.cwnd_event	= pcc_cwnd_event,
};

/* Kernel module section */

static int __init pcc_register(void)
{
	BUILD_BUG_ON(sizeof(struct pcc_data) > ICSK_CA_PRIV_SIZE);
	printk(KERN_INFO "pcc init reg\n");
	return tcp_register_congestion_control(&tcp_pcc_cong_ops);
}

static void __exit pcc_unregister(void)
{
	tcp_unregister_congestion_control(&tcp_pcc_cong_ops);
}

module_init(pcc_register);
module_exit(pcc_unregister);

MODULE_AUTHOR("Han Xue <xiaoxiaoxh@sjtu.edu.cn>");
MODULE_AUTHOR("Yifan Zhang <zhangyf_sjtu@sjtu.edu.cn>");
MODULE_AUTHOR("Kaiwen Zha <Kevin_zha@sjtu.edu.cn>");
MODULE_AUTHOR("Qibing Ren <renqibing@sjtu.edu.cn>");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("TCP PCC (Performance-oriented Congestion Control)");
