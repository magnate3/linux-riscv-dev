#include "tcp.h" 

/*
  * TCP Vegas congestion control interface
  */
 #ifndef __TCP_VEGAS_H
 #define __TCP_VEGAS_H 1

# define do_div(n,base) ({                                      \
		uint32_t __base = (base);                               \
		uint32_t __rem;                                         \
		__rem = ((uint64_t)(n)) % __base;                       \
		(n) = ((uint64_t)(n)) / __base;                         \
		__rem;                                                  \
		})


 
 /* Vegas variables */
 struct vegas {
         uint32_t     beg_snd_nxt;    /* right edge during last RTT */
         uint32_t     beg_snd_una;    /* left edge  during last RTT */
         uint32_t     beg_snd_cwnd;   /* saves the size of the cwnd */
         uint8_t      doing_vegas_now;/* if true, do vegas for this RTT */
         uint16_t     cntRTT;         /* # of RTTs measured within last RTT */
         uint32_t     minRTT;         /* min of RTTs measured within last RTT (in usec) */
         uint32_t     baseRTT;        /* the min of all Vegas RTT measurements seen (in usec) */
	 uint32_t     baseRTT_update_pacing;
 };
 
 void tcp_vegas_init(rudp_srv_state_t *sk);
/* void tcp_vegas_state(rudp_srv_state_t *sk, uint8_t ca_state);*/
 void tcp_vegas_pkts_acked(rudp_srv_state_t *sk, uint32_t ts);
/*void tcp_vegas_cwnd_event(rudp_srv_state_t *sk, enum tcp_ca_event event);*/
void tcp_vegas_cong_avoid(rudp_srv_state_t *tp, uint32_t ack, uint32_t acked);

/* size_t tcp_vegas_get_info(rudp_srv_state_t *sk, uint32_t ext, int *attr,
                           union tcp_cc_info *info);
 */
 #endif  /* __TCP_VEGAS_H */
