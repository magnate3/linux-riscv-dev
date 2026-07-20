/* TCP Inigo congestion control.
 *
 * This is an implementation of TCP Inigo, which takes the measure of
 * the extent of congestion introduced in DCTCP and applies it to
 * networks outside the data center.
 *
 * https://www.soe.ucsc.edu/research/technical-reports/UCSC-SOE-14-14
 *
 * The motivation behind the RTT fairness functionality comes from
 * the 2nd DCTCP paper listed below.
 *
 * Authors:
 *
 *	Andrew Shewmaker <agshew@gmail.com>
 *
 * Forked from DataCenter TCP (DCTCP) congestion control.
 *
 * http://simula.stanford.edu/~alizade/Site/DCTCP.html
 *
 * This is an implementation of DCTCP, an enhancement to the
 * TCP congestion control algorithm designed for data centers. DCTCP
 * leverages Explicit Congestion Notification (ECN) in the network to
 * provide multi-bit feedback to the end hosts. DCTCP's goal is to meet
 * the following three data center transport requirements:
 *
 *  - High burst tolerance (incast due to partition/aggregate)
 *  - Low latency (short flows, queries)
 *  - High throughput (continuous data updates, large file transfers)
 *    with commodity shallow buffered switches
 *
 * The algorithm is described in detail in the following two papers:
 *
 * 1) Mohammad Alizadeh, Albert Greenberg, David A. Maltz, Jitendra Padhye,
 *    Parveen Patel, Balaji Prabhakar, Sudipta Sengupta, and Murari Sridharan:
 *      "Data Center TCP (DCTCP)", Data Center Networks session
 *      Proc. ACM SIGCOMM, New Delhi, 2010.
 *   http://simula.stanford.edu/~alizade/Site/DCTCP_files/inigo-final.pdf
 *
 * 2) Mohammad Alizadeh, Adel Javanmard, and Balaji Prabhakar:
 *      "Analysis of DCTCP: Stability, Convergence, and Fairness"
 *      Proc. ACM SIGMETRICS, San Jose, 2011.
 *   http://simula.stanford.edu/~alizade/Site/DCTCP_files/inigo_analysis-full.pdf
 *
 * Initial prototype from Abdul Kabbani, Masato Yasuda and Mohammad Alizadeh.
 *
 * Authors:
 *
 *	Daniel Borkmann <dborkman@redhat.com>
 *	Florian Westphal <fw@strlen.de>
 *	Glenn Judd <glenn.judd@morganstanley.com>
 *
 * RTT Fallback:
 *
 *	Andrew Shewmaker <shewa@lanl.gov>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 */

#include <linux/module.h>
#include <linux/mm.h>
#include <net/tcp.h>
#include <linux/inet_diag.h>

#define DCTCP_MAX_ALPHA	1024U
#define INIGO_MIN_FAIRNESS 3U   // alpha sensitivity of 684 / 1024
#define INIGO_MAX_FAIRNESS 512U // alpha sensitivity of 4 / 1024
#define INIGO_MAX_MARK 1024U

struct inigo {
	u32 acked_bytes_ecn;
	u32 acked_bytes_total;
	u32 prior_snd_una;
	u32 prior_rcv_nxt;
	u32 inigo_alpha;
	u32 next_seq;
	u32 delayed_ack_reserved;
	u32 rtt_min;
	u32 rtts_late;
	u32 rtts_observed;
	u8 ce_state;
};

static unsigned int inigo_shift_g __read_mostly = 4; /* g = 1/2^4 */
module_param(inigo_shift_g, uint, 0644);
MODULE_PARM_DESC(inigo_shift_g, "parameter g for updating inigo_alpha");

static unsigned int inigo_alpha_on_init __read_mostly = DCTCP_MAX_ALPHA;
module_param(inigo_alpha_on_init, uint, 0644);
MODULE_PARM_DESC(inigo_alpha_on_init, "parameter for initial alpha value");

static unsigned int inigo_clamp_alpha_on_loss __read_mostly;
module_param(inigo_clamp_alpha_on_loss, uint, 0644);
MODULE_PARM_DESC(inigo_clamp_alpha_on_loss,
		 "parameter for clamping alpha on loss");

static unsigned int markthresh __read_mostly = 174;
module_param(markthresh, uint, 0644);
MODULE_PARM_DESC(markthresh, "rtts >  rtt_min + rtt_min * markthresh / 1024"
		" are considered marks of congestion, defaults to 174 out of 1024");

static unsigned int slowstart_rtt_observations_needed __read_mostly = 8;
module_param(slowstart_rtt_observations_needed, uint, 0644);
MODULE_PARM_DESC(slowstart_rtt_observations_needed, "minimum number of RTT observations needed"
		 " to exit slowstart, defaults to 8");

static unsigned int rtt_fairness  __read_mostly = 10;
module_param(rtt_fairness, uint, 0644);
MODULE_PARM_DESC(rtt_fairness, "if non-zero, react to congestion every x acks during cong avoid,"
		 " 0 indicates once per window, otherwise 3 < x < 512, defaults to 10");

static unsigned int inigo_force_ecn __read_mostly = 0;
module_param(inigo_force_ecn, uint, 0644);
MODULE_PARM_DESC(inigo_force_ecn, "force use of ecn (needed for Mininet testing of fallback)");

static struct tcp_congestion_ops inigo;
static struct tcp_congestion_ops inigo_rtt;

static void inigo_reset(const struct tcp_sock *tp, struct inigo *ca)
{
	ca->next_seq = tp->snd_nxt;

	ca->acked_bytes_ecn = 0;
	ca->acked_bytes_total = 0;
}

static void inigo_init(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct inigo *ca = inet_csk_ca(sk);
	const struct inet_sock *inet = inet_sk(sk);

	if (rtt_fairness != 0)
		rtt_fairness = clamp(rtt_fairness, INIGO_MIN_FAIRNESS, INIGO_MAX_FAIRNESS);

	ca->inigo_alpha = 0;
	ca->rtt_min = USEC_PER_SEC;
	ca->rtts_late = 0;
	ca->rtts_observed = 0;

	if (inigo_force_ecn || (tp->ecn_flags & TCP_ECN_OK) ||
	    (sk->sk_state == TCP_LISTEN ||
	     sk->sk_state == TCP_CLOSE)) {
		bool use_ecn = sock_net(sk)->ipv4.sysctl_tcp_ecn == 1 || tcp_ca_needs_ecn(sk);
		const struct dst_entry *dst = __sk_dst_get(sk);
		pr_info("inigo: ecn enabled %pI4 force=%u ecn_ok=%u state=%u use_ecn=%u dst_feature=%u\n",
			&inet->inet_saddr, inigo_force_ecn, tp->ecn_flags & TCP_ECN_OK, sk->sk_state, use_ecn, dst && dst_feature(dst, RTAX_FEATURE_ECN));

		inigo.flags |= TCP_CONG_NEEDS_ECN;
		//TCP_SKB_CB(skb)->tcp_flags |= TCPHDR_ECE | TCPHDR_CWR;
		tp->ecn_flags = TCP_ECN_OK;
		INET_ECN_xmit(sk);

		ca->inigo_alpha = min(inigo_alpha_on_init, DCTCP_MAX_ALPHA);

		ca->prior_snd_una = tp->snd_una;
		ca->prior_rcv_nxt = tp->rcv_nxt;

		ca->delayed_ack_reserved = 0;
		ca->ce_state = 0;

		inigo_reset(tp, ca);
		return;
	}

	/* No ECN support? Fall back to RTT. Also need to clear
	 * ECT from sk since it is set during 3WHS for DCTCP.
	 */
	pr_info("inigo: ecn disabled %pI4 force=%u ecn_ok=%u state=%u\n", &inet->inet_saddr, inigo_force_ecn, tp->ecn_flags & TCP_ECN_OK, sk->sk_state);
	inet_csk(sk)->icsk_ca_ops = &inigo_rtt;
	INET_ECN_dontxmit(sk);
}

static u32 inigo_ssthresh(struct sock *sk)
{
	const struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	return max(tp->snd_cwnd - ((tp->snd_cwnd * ca->inigo_alpha) >> 11U), 2U);
}

static u32 inigo_rtt_ssthresh(struct sock *sk)
{
	const struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);
	u32 nsubwnd = 1;

	if (rtt_fairness) {
		nsubwnd = tp->snd_cwnd;
		if (do_div(nsubwnd, rtt_fairness))
			nsubwnd++;
	}

	return max(tp->snd_cwnd - ((tp->snd_cwnd * ca->inigo_alpha) >> 11U) / nsubwnd, 2U);
}

/* Minimal DCTP CE state machine:
 *
 * S:	0 <- last pkt was non-CE
 *	1 <- last pkt was CE
 */

static void inigo_ce_state_0_to_1(struct sock *sk)
{
	struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	/* State has changed from CE=0 to CE=1 and delayed
	 * ACK has not sent yet.
	 */
	if (!ca->ce_state && ca->delayed_ack_reserved) {
		u32 tmp_rcv_nxt;

		/* Save current rcv_nxt. */
		tmp_rcv_nxt = tp->rcv_nxt;

		/* Generate previous ack with CE=0. */
		tp->ecn_flags &= ~TCP_ECN_DEMAND_CWR;
		tp->rcv_nxt = ca->prior_rcv_nxt;

		tcp_send_ack(sk);

		/* Recover current rcv_nxt. */
		tp->rcv_nxt = tmp_rcv_nxt;
	}

	ca->prior_rcv_nxt = tp->rcv_nxt;
	ca->ce_state = 1;

	tp->ecn_flags |= TCP_ECN_DEMAND_CWR;
}

static void inigo_ce_state_1_to_0(struct sock *sk)
{
	struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	/* State has changed from CE=1 to CE=0 and delayed
	 * ACK has not sent yet.
	 */
	if (ca->ce_state && ca->delayed_ack_reserved) {
		u32 tmp_rcv_nxt;

		/* Save current rcv_nxt. */
		tmp_rcv_nxt = tp->rcv_nxt;

		/* Generate previous ack with CE=1. */
		tp->ecn_flags |= TCP_ECN_DEMAND_CWR;
		tp->rcv_nxt = ca->prior_rcv_nxt;

		tcp_send_ack(sk);

		/* Recover current rcv_nxt. */
		tp->rcv_nxt = tmp_rcv_nxt;
	}

	ca->prior_rcv_nxt = tp->rcv_nxt;
	ca->ce_state = 0;

	tp->ecn_flags &= ~TCP_ECN_DEMAND_CWR;
}

static void inigo_update_rtt_alpha(struct inigo *ca)
{
	u32 alpha = ca->inigo_alpha;
	u32 marks = ca->rtts_late;
	u32 total = ca->rtts_observed;

	/* alpha = (1 - g) * alpha + g * F */
        alpha -= min_not_zero(alpha, alpha >> inigo_shift_g);

	if (marks) {
		/* If shift_g == 1, a 32bit value would overflow
		 * after 8 M.
		 */
		marks <<= (10 - inigo_shift_g);
		do_div(marks, max(1U, total));

		alpha = min(alpha + (u32)marks, DCTCP_MAX_ALPHA);
        }

	ca->inigo_alpha = alpha;
}

/* The cwnd reduction in CWR and Recovery use the PRR algorithm
 * https://datatracker.ietf.org/doc/draft-ietf-tcpm-proportional-rate-reduction/
 * It computes the number of packets to send (sndcnt) based on packets newly
 * delivered:
 *   1) If the packets in flight is larger than ssthresh, PRR spreads the
 *      cwnd reductions across a full RTT.
 *   2) If packets in flight is lower than ssthresh (such as due to excess
 *      losses and/or application stalls), do not perform any further cwnd
 *      reductions, but instead slow start up to ssthresh.
 */
static void inigo_init_cwnd_reduction(struct sock *sk)
{
	struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	tp->high_seq = tp->snd_nxt;
	tp->tlp_high_seq = 0;
	// tp->snd_cwnd_cnt = 0; commented out because of rtt-fairness support
	tp->prior_cwnd = tp->snd_cwnd;
	tp->prr_delivered = 0;
	tp->prr_out = 0;
	tp->snd_ssthresh = inigo_rtt_ssthresh(sk);
	ca->rtts_late = 0;
	ca->rtts_observed = 0;
}

/* Enter CWR state. Disable cwnd undo since congestion is proven with ECN or Delay */
void inigo_enter_cwr(struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);

	tp->prior_ssthresh = 0;
	if (inet_csk(sk)->icsk_ca_state < TCP_CA_CWR) {
		tp->undo_marker = 0;
		inigo_init_cwnd_reduction(sk);
		tcp_set_ca_state(sk, TCP_CA_CWR);
	}
}

static void inigo_update_alpha(struct sock *sk, u32 flags)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct inigo *ca = inet_csk_ca(sk);
	u32 acked_bytes = tp->snd_una - ca->prior_snd_una;

	/* If ack did not advance snd_una, count dupack as MSS size.
	 * If ack did update window, do not count it at all.
	 */
	if (acked_bytes == 0 && !(flags & CA_ACK_WIN_UPDATE))
		acked_bytes = inet_csk(sk)->icsk_ack.rcv_mss;
	if (acked_bytes) {
		if (flags & CA_ACK_ECE) {
			if (ca->acked_bytes_ecn == 0 && tcp_in_slow_start(tp)) {
				tp->snd_ssthresh = tp->snd_cwnd;
			}

			ca->acked_bytes_ecn += acked_bytes;
		}
		ca->acked_bytes_total += acked_bytes;
		ca->prior_snd_una = tp->snd_una;
	}

	/* Expired RTT */
	if (!before(tp->snd_una, ca->next_seq)) {
		u64 bytes_ecn = ca->acked_bytes_ecn;
		u32 alpha = ca->inigo_alpha;

		/* alpha = (1 - g) * alpha + g * F */

		alpha -= min_not_zero(alpha, alpha >> inigo_shift_g);
		if (bytes_ecn) {
			/* If inigo_shift_g == 1, a 32bit value would overflow
			 * after 8 Mbytes.
			 */
			bytes_ecn <<= (10 - inigo_shift_g);
			do_div(bytes_ecn, max(1U, ca->acked_bytes_total));

			alpha = min(alpha + (u32)bytes_ecn, DCTCP_MAX_ALPHA);
		}
		/* inigo_alpha can be read from inigo_get_info() without
		 * synchro, so we ask compiler to not use inigo_alpha
		 * as a temporary variable in prior operations.
		 */
		WRITE_ONCE(ca->inigo_alpha, alpha);
		inigo_reset(tp, ca);

		/* Fall back to RTT-based congestion control if alpha stays low
		 * even though RTTs are increasing during the window.
		 */
		if (ca->acked_bytes_ecn == 0 && ca->rtts_late > 1 && ca->rtts_observed >= slowstart_rtt_observations_needed) {
			pr_info("inigo: ecn unconfigured, falling back to RTT-based congestion control\n");
			inet_csk(sk)->icsk_ca_ops = &inigo_rtt;
			INET_ECN_dontxmit(sk);

/*
			inigo_update_rtt_alpha(ca);

			if (ca->inigo_alpha) {
				inigo_enter_cwr(sk);
				return;
			}
 */
		}
		ca->rtts_late = 0;
	}
}

static void inigo_state(struct sock *sk, u8 new_state)
{
	if (inigo_clamp_alpha_on_loss && new_state == TCP_CA_Loss) {
		struct inigo *ca = inet_csk_ca(sk);

		/* If this extension is enabled, we clamp inigo_alpha to
		 * max on packet loss; the motivation is that inigo_alpha
		 * is an indicator to the extend of congestion and packet
		 * loss is an indicator of extreme congestion; setting
		 * this in practice turned out to be beneficial, and
		 * effectively assumes total congestion which reduces the
		 * window by half.
		 */
		ca->inigo_alpha = DCTCP_MAX_ALPHA;
	}
}

static void inigo_update_ack_reserved(struct sock *sk, enum tcp_ca_event ev)
{
	struct inigo *ca = inet_csk_ca(sk);

	switch (ev) {
	case CA_EVENT_DELAYED_ACK:
		if (!ca->delayed_ack_reserved)
			ca->delayed_ack_reserved = 1;
		break;
	case CA_EVENT_NON_DELAYED_ACK:
		if (ca->delayed_ack_reserved)
			ca->delayed_ack_reserved = 0;
		break;
	default:
		/* Don't care for the rest. */
		break;
	}
}

static void inigo_cwnd_event(struct sock *sk, enum tcp_ca_event ev)
{
	switch (ev) {
	case CA_EVENT_ECN_IS_CE:
		inigo_ce_state_0_to_1(sk);
		break;
	case CA_EVENT_ECN_NO_CE:
		inigo_ce_state_1_to_0(sk);
		break;
	case CA_EVENT_DELAYED_ACK:
	case CA_EVENT_NON_DELAYED_ACK:
		inigo_update_ack_reserved(sk, ev);
		break;
	default:
		/* Don't care for the rest. */
		break;
	}
}

/* This is the same as newer tcp_slow_start(). It is only here for while inigo
 * is being built out of tree against older kernels that don't do it this way.
 */
u32 inigo_slow_start(struct tcp_sock *tp, u32 acked)
{
	u32 cwnd = tp->snd_cwnd + acked;

	if (cwnd > tp->snd_ssthresh)
		cwnd = tp->snd_ssthresh + 1;
	acked -= cwnd - tp->snd_cwnd;
	tp->snd_cwnd = min(cwnd, tp->snd_cwnd_clamp);

	return acked;
}

void inigo_cong_avoid_ai(struct sock *sk, u32 w, u32 acked)
{
	struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);
	u32 interval = tp->snd_cwnd;

	if (tp->snd_cwnd_cnt >= w) {
		if (tp->snd_cwnd < tp->snd_cwnd_clamp) {
			tp->snd_cwnd++;
			if (rtt_fairness)
				tp->snd_cwnd++;
		}

		tp->snd_cwnd_cnt = 0;
	}

	if (rtt_fairness)
		interval = min(interval, rtt_fairness);

	if (tp->snd_cwnd_cnt >= interval) {
		if (tp->snd_cwnd_cnt % interval == 0 || tp->snd_cwnd_cnt >= w) {
			inigo_update_rtt_alpha(ca);

			if (ca->inigo_alpha)
				inigo_enter_cwr(sk);
		}
	}

	if (tp->snd_cwnd_cnt < w) {
		tp->snd_cwnd_cnt += acked;
	}
}

void inigo_cong_avoid(struct sock *sk, u32 ack, u32 acked)
{
	struct inigo *ca = inet_csk_ca(sk);
	struct tcp_sock *tp = tcp_sk(sk);

	if (!tcp_is_cwnd_limited(sk)) {
		return;
	}

	if (tp->snd_cwnd <= tp->snd_ssthresh) {
		if (ca->rtts_observed >= slowstart_rtt_observations_needed) {
			inigo_update_rtt_alpha(ca);

			if (ca->inigo_alpha) {
				inigo_enter_cwr(sk);
				return;
			}
		}

		/* In "safe" area, increase. */
		acked = inigo_slow_start(tp, acked);
		if (!acked)
			return;
	}
	/* In dangerous area, increase slowly. */
	inigo_cong_avoid_ai(sk, tp->snd_cwnd, acked);
}

static void inigo_pkts_acked(struct sock *sk, u32 num_acked, s32 rtt)
{
	struct inigo *ca = inet_csk_ca(sk);

	/* Some calls are for duplicates without timetamps */
	if (rtt <= 0)
		return;

	ca->rtts_observed++;

	ca->rtt_min = min((u32) rtt, ca->rtt_min);

	/* Mimic DCTCP's ECN marking threshhold of approximately 0.17*BDP */
	if ((u32) rtt > (ca->rtt_min + (ca->rtt_min * markthresh / INIGO_MAX_MARK)))
		ca->rtts_late++;
}

static struct tcp_congestion_ops inigo __read_mostly = {
	.init		= inigo_init,
	.in_ack_event   = inigo_update_alpha,
	.cwnd_event	= inigo_cwnd_event,
	.ssthresh	= inigo_ssthresh,
	.cong_avoid	= tcp_reno_cong_avoid,
	.pkts_acked 	= inigo_pkts_acked,
	.set_state	= inigo_state,
	.owner		= THIS_MODULE,
	.name		= "inigo",
};

static struct tcp_congestion_ops inigo_rtt __read_mostly = {
	.ssthresh	= inigo_rtt_ssthresh,
	.cong_avoid	= inigo_cong_avoid,
	.pkts_acked 	= inigo_pkts_acked,
	.owner		= THIS_MODULE,
	.name		= "inigo-rtt",
};

static int __init inigo_register(void)
{
	BUILD_BUG_ON(sizeof(struct inigo) > ICSK_CA_PRIV_SIZE);
	return tcp_register_congestion_control(&inigo);
}

static void __exit inigo_unregister(void)
{
	tcp_unregister_congestion_control(&inigo);
}

module_init(inigo_register);
module_exit(inigo_unregister);

MODULE_AUTHOR("Daniel Borkmann <dborkman@redhat.com>");
MODULE_AUTHOR("Florian Westphal <fw@strlen.de>");
MODULE_AUTHOR("Glenn Judd <glenn.judd@morganstanley.com>");

MODULE_LICENSE("GPL v2");
MODULE_DESCRIPTION("TCP Inigo");
