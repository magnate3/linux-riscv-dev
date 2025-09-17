#ifndef _MINMAX_H_
#define _MINMAX_H_

struct minmax_sample {
  uint32_t t;   /* time measurement was taken */
  uint32_t v;   /* value measured */
};

struct minmax {
  struct minmax_sample s[3];
};

static inline uint32_t minmax_get(const struct minmax *m)
{
  return m->s[0].v;
}

static inline uint32_t minmax_reset(struct minmax *m, uint32_t t, uint32_t meas)
{
  struct minmax_sample val = { .t = t, .v = meas };

  m->s[2] = m->s[1] = m->s[0] = val;
  return m->s[0].v;
}

#define min_t(t, a, b) min((t)(a), (t)(b))
#define min(a, b) ((a) < (b)? (a): (b))

#define max_t(t, a, b) max((t)(a), (t)(b))
#define max(a, b) ((a) < (b)? (b): (a))

struct tcp_sock {
    uint32_t srtt_us;
    uint32_t snd_cwnd;
    uint32_t app_limited;
    uint32_t mss_cache;
    uint64_t tcp_mstamp;
    uint32_t tcp_clock_cache;
    uint32_t tcp_wstamp_ns;
    uint32_t snd_ssthresh;
    uint64_t delivered_mstamp;
    uint32_t delivered;
    uint32_t lost;
    uint32_t snd_cwnd_clamp;
    uint32_t packets_out;
};

struct sock {
    struct tcp_sock sk_tcp_sock;
    unsigned long sk_pacing_rate;
    unsigned long sk_max_pacing_rate;
    unsigned char sk_pacing_shift;
    unsigned char sk_pacing_status;
    unsigned char sk_csk_ca[512];
};

typedef unsigned char bool;

static inline void * inet_csk_ca(const void *d) {
    struct sock *psock = (struct sock *)d;
    return psock->sk_csk_ca;
}

static inline struct tcp_sock * tcp_sk(const void *p) {
    struct sock *psock = (struct sock *)p;
    return &psock->sk_tcp_sock;
}

#define USEC_PER_SEC	1000000L
#define USEC_PER_MSEC   1000L
#define NSEC_PER_USEC   1000L
#define unlikely(exp)   exp
#define GSO_MAX_SIZE	65536
#define MAX_HEADER      64
#define MAX_TCP_HEADER	(128 + MAX_HEADER)
#define READ_ONCE(x)     (x)

enum {
	TCP_CA_Open,
	TCP_CA_Loss,
	TCP_CA_Recovery
};

enum tcp_ca_event {
	CA_EVENT_TX_START
};
#define TCP_INIT_CWND     10
struct rate_sample {
	uint64_t  prior_mstamp; /* starting timestamp for interval */
	uint32_t  prior_delivered;	/* tp->delivered at "prior_mstamp" */
#if 0
	uint32_t  prior_delivered_ce;/* tp->delivered_ce at "prior_mstamp" */
#endif
	int32_t  delivered;		/* number of packets delivered over interval */
#if 0
	int32_t  delivered_ce;	/* number of packets delivered w/ CE marks*/
#endif
	long interval_us;	/* time for tp->delivered to incr "delivered" */
#if 0
	uint32_t snd_interval_us;	/* snd interval for delivered packets */
	uint32_t rcv_interval_us;	/* rcv interval for delivered packets */
#endif
	long rtt_us;		/* RTT of last (S)ACKed packet (or -1) */
	int  losses;		/* number of packets marked lost upon ACK */
	uint32_t  acked_sacked;	/* number of packets newly (S)ACKed upon ACK */
	uint32_t  prior_in_flight;	/* in flight before this ACK */
	bool is_app_limited;	/* is sample from packet with bubble in pipe? */
#if 0
	bool is_retrans;	/* is sample from retransmission? */
#endif
	bool is_ack_delayed;	/* is this (likely) a delayed ACK? */
	int32_t last_end_seq;
};

struct tcp_bbr_info {
	/* u64 bw: max-filtered BW (app throughput) estimate in Byte per sec: */
	uint32_t	bbr_bw_lo;		/* lower 32 bits of bw */
	uint32_t	bbr_bw_hi;		/* upper 32 bits of bw */
	uint32_t	bbr_min_rtt;		/* min-filtered RTT in uSec */
	uint32_t	bbr_pacing_gain;	/* pacing gain shifted left 8 bits */
	uint32_t	bbr_cwnd_gain;		/* cwnd gain shifted left 8 bits */
};

union tcp_cc_info {
	struct tcp_bbr_info	bbr;
};

enum {
	INET_DIAG_NONE,
	INET_DIAG_BBRINFO,
	INET_DIAG_VEGASINFO
};

enum {
	SK_PACING_NONE,
	SK_PACING_NEEDED
};

extern uint32_t tcp_jiffies32;

#define HZ 100
#define TCP_INFINITE_SSTHRESH 65535
#define cmpxchg(p, x, y) {if (*p == x) *p = y;}
enum {false, true};

static uint32_t minmax_subwin_update(struct minmax *m, uint32_t win,
    const struct minmax_sample *val)
{
  uint32_t dt = val->t - m->s[0].t;

  if (dt > win) {
    m->s[1] = m->s[2];
    m->s[2] = *val;
    if (val->t - m->s[0].t > win) {
      m->s[0] = m->s[1];
      m->s[1] = m->s[2];
      m->s[2] = *val;
    }
  } else if (m->s[1].t == m->s[0].t && dt > win/4) {
    m->s[2] = m->s[1] = *val;
  } else if (m->s[2].t == m->s[1].t && dt > win/2) {
    m->s[2] = *val;
  }
  return m->s[0].v;
}

static uint32_t minmax_running_max(struct minmax *m, uint32_t win, uint32_t t, uint32_t meas)
{
  struct minmax_sample val = { .t = t, .v = meas };

  if ((val.v >= m->s[0].v) ||
      (val.t - m->s[2].t > win)) {
    m->s[2] = m->s[1] = m->s[0] = val;
    return m->s[0].v;
  }

  if (val.v >= m->s[1].v)
    m->s[2] = m->s[1] = val;
  else if (val.v >= m->s[2].v)
    m->s[2] = val;

  return minmax_subwin_update(m, win, &val);
}

#define after(a, b)  ((int32_t)((a) - (b)) > 0)
#define before(a, b) ((int32_t)((a) - (b)) < 0)

#define div64_long(a, b) (a/b)
#define div_u64(a, b)  (a/b)
#define do_div(a,b)      a/=(b)
#define msecs_to_jiffies(t) (t/10)
#define tcp_min_rtt(a) 1234567
#define WARN_ONCE
#define tcp_packets_in_flight(tp) tp->packets_out
#define tcp_stamp_us_delta(p, c) (p - c)
#define prandom_u32_max(a) 90

#endif
