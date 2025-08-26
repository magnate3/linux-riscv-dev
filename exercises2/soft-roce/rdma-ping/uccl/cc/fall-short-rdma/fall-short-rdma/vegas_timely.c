/*
  * TCP Vegas congestion control
  *
  * This is based on the congestion detection/avoidance scheme described in
  *    Lawrence S. Brakmo and Larry L. Peterson.
  *    "TCP Vegas: End to end congestion avoidance on a global internet."
  *    IEEE Journal on Selected Areas in Communication, 13(8):1465--1480,
  *    October 1995. Available from:
  *      ftp://ftp.cs.arizona.edu/xkernel/Papers/jsac.ps
  *
  * See http://www.cs.arizona.edu/xkernel/ for their implementation.
  * The main aspects that distinguish this implementation from the
  * Arizona Vegas implementation are:
  *   o We do not change the loss detection or recovery mechanisms of
  *     Linux in any way. Linux already recovers from losses quite well,
  *     using fine-grained timers, NewReno, and FACK.
  *   o To avoid the performance penalty imposed by increasing cwnd
  *     only every-other RTT during slow start, we increase during
  *     every RTT during slow start, just like Reno.
  *   o Largely to allow continuous cwnd growth during slow start,
  *     we use the rate at which ACKs come back as the "actual"
  *     rate, rather than the rate at which data is sent.
  *   o To speed convergence to the right rate, we set the cwnd
  *     to achieve the right ("actual") rate when we exit slow start.
  *   o To filter out the noise caused by delayed ACKs, we use the
  *     minimum RTT sample observed during the last RTT to calculate
  *     the actual rate.
  *   o When the sender re-starts from idle, it waits until it has
  *     received ACKs for an entire flight of new data before making
  *     a cwnd adjustment decision. The original Vegas implementation
  *     assumed senders never went idle.
  */
 
// #include <linux/mm.h>
/* #include <linux/module.h>
 #include <linux/skbuff.h>
 #include <linux/inet_diag.h>
 
 #include <net/tcp.h>
 */

 #include "timely.h"

#define BASERTT_PACING_STEP 100000
 static int alpha = 2;
 static int beta  = 4;
 static int gamma_timely = 1;

static double TIMELY_BETA = 0.8;
static double TIMELY_ALPHA = 0.25; 
/* module_param(alpha, int, 0644);
 MODULE_PARM_DESC(alpha, "lower bound of packets in network");
 module_param(beta, int, 0644);
 MODULE_PARM_DESC(beta, "upper bound of packets in network");
 module_param(gamma_timely, int, 0644);
 MODULE_PARM_DESC(gamma_timely, "limit on increase (scale by 2)");
 */
 /* There are several situations when we must "re-start" Vegas:
  *
  *  o when a connection is established
  *  o after an RTO
  *  o after fast recovery
  *  o when we send a packet and there is no outstanding
  *    unacknowledged data (restarting an idle connection)
  *
  * In these circumstances we cannot do a Vegas calculation at the
  * end of the first RTT, because any calculation we do is using
  * stale info -- both the saved cwnd and congestion feedback are
  * stale.
  *
  * Instead we must wait until the completion of an RTT during
  * which we actually receive ACKs.
  */
 static void timely_enable(rudp_srv_state_t *tp)
 {
         struct Timely *timely = (struct Timely *)tp->cong; 
 
		 
        /* Begin taking Vegas samples next time we send something. */
         timely->doing_timely_now = 1;
 
         /* Set the beginning of the next send window. */
         timely->beg_snd_nxt = tp->send_bytes;
 
         timely->cntRTT = 0;
	 timely->rtt_diff = 0;
         timely->minRTT = 0x7fffffff;
	 timely->prev_minRTT = 0x7fffffff;
	 timely->add_count = 0;
 }
 
 /* Stop taking Vegas samples for now. */
/* static inline void timely_disable(rudp_srv_state_t *tp)
 {
         struct Timely *timely = inet_csk_ca(sk);
 
         timely->doing_timely_now = 0;
 }
 */
 void tcp_timely_init(rudp_srv_state_t *tp)
 {
         struct Timely *timely = (struct Timely *)malloc(sizeof(struct Timely));

	 tp->cong = timely; 
         timely->baseRTT = 0x7fffffff;
	 timely->baseRTT_update_pacing = BASERTT_PACING_STEP;
         timely_enable(tp);
 }
 
 /* Do RTT sampling needed for Vegas.
  * Basically we:
  *   o min-filter RTT samples from within an RTT to get the current
  *     propagation delay + queuing delay (we are min-filtering to try to
  *     avoid the effects of delayed ACKs)
  *   o min-filter RTT samples from a much longer window (forever for now)
  *     to find the propagation delay (baseRTT)
  */
 void tcp_timely_pkts_acked(rudp_srv_state_t *tp, uint32_t rtt_us)
 {
         struct Timely *timely = (struct Timely *)tp->cong;
         uint32_t vrtt;
 
        /* if (sample->rtt_us < 0)
                 return;
		*/ 
         /* Never allow zero rtt or baseRTT */
         vrtt = rtt_us + 1;
 
         /* Filter to find propagation delay: */
         if (vrtt < timely->baseRTT)
//         if(timely->baseRTT == 0x7fffffff)   
	      timely->baseRTT = vrtt;
 
         /* Find the min RTT during the last RTT to find
          * the current prop. delay + queuing delay:
          */
         timely->minRTT = GET_MIN(timely->minRTT, vrtt);
	 if(timely->prev_minRTT == 0x7fffffff)
		timely->prev_minRTT = timely->minRTT;
 
         timely->cntRTT++;
 }
 
/* void tcp_timely_state(rudp_srv_state_t *tp, u8 ca_state)
 {
         if (ca_state == TCP_CA_Open)
                 timely_enable(sk);
         else
                 timely_disable(sk);
 }
 */
 /*
  * If the connection is idle and we are restarting,
  * then we don't want to do any Vegas calculations
  * until we get fresh RTT samples.  So when we
  * restart, we reset our Vegas state to a clean
  * slate. After we get acks for this flight of
  * packets, _then_ we can make Vegas calculations
  * again.
  */
/* void tcp_timely_cwnd_event(rudp_srv_state_t *tp, enum tcp_ca_event event)
 {
         if (event == CA_EVENT_CWND_RESTART ||
             event == CA_EVENT_TX_START)
                 tcp_timely_init(sk);
 }
 */

 static inline uint32_t tcp_timely_ssthresh(rudp_srv_state_t *tp)
 {
         return  GET_MIN(tp->ss_thresh, tp->cwnd_bytes);
 }
 
void tcp_timely_cong_avoid(rudp_srv_state_t *tp, uint32_t ack, uint32_t acked)
 {
         struct Timely *timely = (struct Timely *)tp->cong;
 
         if (!timely->doing_timely_now) {
                 tcp_reno_cong_avoid(tp, ack, acked);
                 return;
         }

         if (after(ack, timely->beg_snd_nxt)) {
                 /* Do the Vegas once-per-RTT cwnd adjustment. */
 
                 /* Save the extent of the current window so we can use this
                  * at the end of the next RTT.
                  */
				//FIX ME TODO
		
                 timely->beg_snd_nxt  = tp->send_bytes;// tp->snd_nxt;
 
                 /* We do the Vegas calculations only if we got enough RTT
                  * samples that we can be reasonably sure that we got
                  * at least one RTT sample that wasn't from a delayed ACK.
                  * If we only had 2 samples total,
                  * then that means we're getting only 1 ACK per RTT, which
                  * means they're almost certainly delayed ACKs.
                  * If  we have 3 samples, we should be OK.
                  */
 
                 if (timely->cntRTT <= 1) {
                         /* We don't have enough RTT samples to do the Vegas
                          * calculation, so we'll behave like Reno.
                          */
                         tcp_reno_cong_avoid(tp, ack, acked);
                 } else {
			//printf("in tcp_timely\n");
                         uint32_t rtt, diff;
			uint64_t target_cwnd;
                         /* We have enough RTT samples, so, using the Vegas
                          * algorithm, we determine if we should increase or
                          * decrease cwnd, and by how much.
                          */
 
                         /* Pluck out the RTT we are using for the Vegas
                          * calculations. This is the min RTT seen during the
                          * last RTT. Taking the min filters out the effects
                          * of delayed ACKs, at the cost of noticing congestion
                          * a bit later.
                          */
                         rtt = timely->minRTT;
			 // printf("prev %"PRIu32"\n", timely->prev_minRTT); 
		          /* Calculate the cwnd we should have, if we weren't
                          * going too fast.
                          *
                          * This is:
                          *     (actual rate in segments) * baseRTT
                          */
			
			 //printf("sample rtt %"PRIu32", new_rtt_diff %ld, rtt_diff %ld\n",rtt, new_rtt_diff, timely->rtt_diff); 
			 //printf("rtt_diff %ld, baseRTT %"PRIu32"\n", timely->rtt_diff, timely->baseRTT);
                         //printf("normalized_gradient %f\n", normalized_gradient); 
			 target_cwnd = (uint64_t)tp->cwnd_bytes * timely->baseRTT;
                         do_div(target_cwnd, rtt);
                         
			/* Calculate the difference between the window we had,
                          * and the window we would like to have. This quantity
                          * is the "Diff" from the Arizona Vegas papers.
                          */
                         diff = tp->cwnd_bytes/PACKET_SIZE * (rtt-timely->baseRTT) / timely->baseRTT;
                         if (diff> gamma_timely && tcp_in_slow_start(tp)) {
                                 /* Going too fast. Time to slow down
                                  * and switch to congestion avoidance.
                                  */
 
                                 /* Set cwnd to match the actual rate
                                  * exactly:
                                  *   cwnd = (actual rate) * baseRTT
                                  * Then we add 1 because the integer
                                  * truncation robs us of full link
                                  * utilization.
                                  */
                                 tp->cwnd_bytes = GET_MIN(tp->cwnd_bytes, (uint64_t)(target_cwnd+PACKET_SIZE));
				 
                                 tp->ss_thresh = tcp_timely_ssthresh(tp);
 				
                         } else if (tcp_in_slow_start(tp)) {
                                 /* Slow start.  */
                                 tcp_slow_start(tp, acked);
				 //fprintf(stderr, "in tcp_slow_start %"PRIu64"\n", tp->cwnd_bytes);
                         } else {
                                 /* Congestion avoidance. */
 
                                 /* Figure out where we would like cwnd
                                  * to be.
                                  */
                                 if (diff > beta) {
                                         /* The old window was too fast, so
                                          * we slow down.
                                          */
					 tp->cwnd_bytes -= PACKET_SIZE;
                                         tp->ss_thresh
                                                 = tcp_timely_ssthresh(tp);
					
					//fprintf(stderr,"timely > beta%"PRIu64",a %ld\n", tp->cwnd_bytes,a);
                                 } else if (diff < alpha) {
                                         /* We don't have enough extra packets
                                          * in the network, so speed up.
                                          */
                                         tp->cwnd_bytes += PACKET_SIZE;
				 	//fprintf(stderr, "diff < alpha %"PRIu64"\n", tp->cwnd_bytes);
                                 } else {
                                         /* Sending just as fast as we
                                          * should be.
                                          */
                                 }

				}
 
                         if (tp->cwnd_bytes < 2*PACKET_SIZE)
                                 tp->cwnd_bytes = 2*PACKET_SIZE;
                        /* else if (tp->cwnd_size > tp->max_cwnd_size)
                                 tp->cwnd_size = tp->max_cwnd_size;
 			*/
                         tp->ss_thresh = tcp_current_ssthresh(tp);
			 //fprintf(stderr, "after ss_thresh %"PRIu64"\n", tp->cwnd_bytes);
                 }
         	 
	/*	if(timely->beg_snd_nxt/(PACKET_SIZE*timely->baseRTT_update_pacing) > 0){
			timely->baseRTT_update_pacing +=BASERTT_PACING_STEP;
			timely->baseRTT = timely->minRTT;	
		 }*/ 
                 /* Wipe the slate clean for the next RTT. */
                 timely->cntRTT = 0;
                 timely->minRTT = 0x7fffffff;

	}
         /* Use normal slow start */
         else if (tcp_in_slow_start(tp)){
	//	printf("acked %d\n", acked);
	        tcp_slow_start(tp, acked);
	}
	tp->rate_limiter = GET_MIN((double)(tp->cwnd_bytes*8.0/timely->baseRTT)/1000, 10.0);
	fprintf(stdout, "cwnd %"PRIu64", rate %f, baseRTT %"PRIu32"\n", tp->cwnd_bytes, tp->rate_limiter, timely->baseRTT);
	//printf("rate_limiter %f\n", tp->rate_limiter);
 }
 
 /* Extract info for Tcp socket info provided via netlink. */
/* size_t tcp_timely_get_info(rudp_srv_state_t *tp, uint32_t ext, int *attr,
                           union tcp_cc_info *info)
 {
         const struct Timely *ca = inet_csk_ca(sk);
 
         if (ext & (1 << (INET_DIAG_VEGASINFO - 1))) {
                 info->timely.tcpv_enabled = ca->doing_timely_now,
                 info->timely.tcpv_rttcnt = ca->cntRTT,
                 info->timely.tcpv_rtt = ca->baseRTT,
                 info->timely.tcpv_minrtt = ca->minRTT,
 
                 *attr = INET_DIAG_VEGASINFO;
                 return sizeof(struct tcptimely_info);
         }
         return 0;
 }
 */
/* static struct tcp_congestion_ops tcp_timely __read_mostly = {
         .init           = tcp_timely_init,
         .ssthresh       = tcp_reno_ssthresh,
         .cong_avoid     = tcp_timely_cong_avoid,
         .pkts_acked     = tcp_timely_pkts_acked,
         .set_state      = tcp_timely_state,
         .cwnd_event     = tcp_timely_cwnd_event,
         .get_info       = tcp_timely_get_info,
 
         .owner          = THIS_MODULE,
         .name           = "timely",
 };
 
*/
