#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

#include "timely_tcp.h"



/* Slow start is used when congestion window is no greater than the slow start
  * threshold. We base on RFC2581 and also handle stretch ACKs properly.
  * We do not implement RFC3465 Appropriate Byte Counting (ABC) per se but
  * something better;) a packet is only considered (s)acked in its entirety to
  * defend the ACK attacks described in the RFC. Slow start processes a stretch
  * ACK of degree N as if N acks of degree 1 are received back to back except
  * ABC caps N to 2. Slow start exits when cwnd grows over ssthresh and
  * returns the leftover acks to adjust cwnd in congestion avoidance mode.
  */
 uint32_t tcp_slow_start(rudp_srv_state_t *tp, uint32_t acked)
 {
         uint32_t cwnd = GET_MIN(tp->cwnd_bytes + acked, tp->ss_thresh);
         acked -= cwnd - tp->cwnd_bytes;
        // tp->cwnd_size = GET_MIN(cwnd, tp->max_cwnd_size);
  	 tp->cwnd_bytes = cwnd; 
//	 printf("@tcp_slow_start  %"PRIu64", %"PRIu32"\n", tp->cwnd_bytes,acked);
         return acked;
 }
 
 /* In theory this is tp->cwnd_size += 1 / tp->cwnd_size (or alternative w),
  * for every packet that was ACKed.
  */
 void tcp_cong_avoid_ai(rudp_srv_state_t *tp, uint32_t w, uint32_t acked)
 {
         /* If credits accumulated at a higher w, apply them gently now. */
         if (tp->cwnd_size_cnt >= w) {
                 tp->cwnd_size_cnt = 0;
                 tp->cwnd_bytes += PACKET_SIZE;
         }
 
         tp->cwnd_size_cnt += acked;
         if (tp->cwnd_size_cnt >= w) {
		 //printf("%d, %d", tp->cwnd_size_cnt, w);
                 uint32_t delta = tp->cwnd_size_cnt / w;
 
                 tp->cwnd_size_cnt -= delta * w;
                 tp->cwnd_bytes += delta*PACKET_SIZE;
         }
        // tp->cwnd_size = GET_MIN(tp->cwnd_size, tp->max_cwnd_size);
 }

 /*
  * TCP Reno congestion control
  * This is special case used for fallback as well.
  */
 /* This is Jacobson's slow start and congestion avoidance.
  * SIGCOMM '88, p. 328.
  */
 void tcp_reno_cong_avoid(rudp_srv_state_t  *tp, uint32_t ack, uint32_t acked)
 {
 
/*         if (!tcp_is_cwnd_limited(sk))
                 return;
*/ 
         /* In "safe" area, increase. */
         if (tcp_in_slow_start(tp)) {
                 acked = tcp_slow_start(tp, acked);
                 if (!acked)
                         return;
         }
         /* In dangerous area, increase slowly. */
         tcp_cong_avoid_ai(tp, tp->cwnd_bytes, acked);
 }
 
 /* Slow start threshold is half the congestion window (min 2) */
 uint32_t tcp_reno_ssthresh(rudp_srv_state_t *tp )
 {
         return GET_MAX(tp->cwnd_bytes >> 1U, 2U);
 }
