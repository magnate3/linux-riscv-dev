#ifndef _FORGE_H
#define _FORGE_H

#include <linux/types.h>

#define SOCK_FORGE      9   /* new protocol */
#define TCP_STATE       18  /* new TCP sockopt */

struct tcp_state {
	__be32      src_ip;
	__be32      dst_ip;

	__be16      sport;
	__be16      dport;
	__u32       seq;
	__u32       ack;
	__u32	    snd_una;	/* First byte we want an ack for */
        __u32       snd_nxt;        /* Next sequence we send                */
        __u32       flight;        /* Next sequence we send                */
        __u32       snd_cwnd;       /* Sending congestion window            */
        __u32       mss_cache;      /* Cached effective mss, not including SACKS */
        __u32       current_mss;
        __u32       snd_ssthresh;   /* Slow start size threshold            */
        __u32       snd_cwnd_cnt;   /* Linear increase counter              */
        __u32       snd_cwnd_clamp; /* Do not allow snd_cwnd to grow above this */
        __u32       snd_cwnd_used;
        __u32       snd_cwnd_stamp;
	__u8        tstamp_ok;
	__u8        sack_ok;
	__u8        wscale_ok;
	__u8        ecn_ok;
	__u8        snd_wscale;
	__u8        rcv_wscale;

	__u32       snd_wnd;
	__u32       rcv_wnd;
        __u32       rcv_nxt;        /* What we want to receive next         */
        __u32       rcv_wup;        /* rcv_nxt on last window update sent   */

	__u32       ts_recent;  /* Timestamp to echo next. */
	__u32       ts_val;     /* Timestamp to use next. */

	__u32       mss_clamp;

	/*
	 * Fields that are below the TCP layer, but that we
	 * might want to mess with anyway.
	 */
	__s16       inet_ttl;   /* unicast IP ttl (use -1 for the default) */
};


#ifdef __KERNEL__
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 9, 0)
int forge_setsockopt(struct sock *sk, int level, int optname,
		char __user *optval, unsigned int optlen);
#else
int forge_setsockopt(struct sock *sk, int level, int optname,
		sockptr_t optval, unsigned int optlen);
#endif
int forge_getsockopt(struct sock *sk, int level, int optname,
		char __user *optval, int __user *optlen);
int forge_getsockopt_socket(struct socket *sock, int level, int optname,
		char __user *optval, int __user *optlen)
{
	return forge_getsockopt(sock->sk, level, optname, optval, optlen);
}
struct sock *forge_csk_accept(struct sock *sk, int flags, int *err);
#endif



#endif
