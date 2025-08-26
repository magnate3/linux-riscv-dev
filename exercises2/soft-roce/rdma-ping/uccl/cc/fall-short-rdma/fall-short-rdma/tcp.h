#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>

#include "share.h"
#include "queue.h"
#define INIT_CWND 	1	
#define MAX_CWND_SIZE 	1024
#define MAX_SLOTS_NUM 	1024
#define SAMPLE_RTT_INTERVAL 64
 /* Events passed to congestion control interface */
 enum tcp_ca_event {
         CA_EVENT_TX_START,      /* first transmit when no packets in flight */
         CA_EVENT_CWND_RESTART,  /* congestion window restart */
         CA_EVENT_COMPLETE_CWR,  /* end of congestion recovery */
         CA_EVENT_LOSS,          /* loss timeout */
         CA_EVENT_ECN_NO_CE,     /* ECT set, but not CE marked */
         CA_EVENT_ECN_IS_CE,     /* received CE marked IP packet */
         CA_EVENT_DELAYED_ACK,   /* Delayed ack is sent */
         CA_EVENT_NON_DELAYED_ACK,
 };

 struct ack_sample {
         uint32_t  pkts_acked;
         uint32_t rtt_us;
         uint32_t in_flight;
 };

/* Datastructures */

/* Header prepended to each UDP datagram - For reliability */
struct hdr {
	uint32_t msg_type;
	uint32_t app_seq;
	uint32_t seq;
//	uint32_t ts;
	uint32_t nic_ts;
	uint32_t window_size;
	int      retransmit;
	int 	end; 
	struct ibv_send_wr *wr;	
} ;

typedef struct rudp_payload_s {
    uint8_t     *data;
    uint8_t     valid;
    uint32_t    data_size;
    struct ibv_send_wr wr; 
} rudp_payload_t;


typedef struct rudp_srv_state_s {
	int 	send_transport;
	struct qp_attr *local_qp_attribute;
	struct qp_attr *remote_qp_attribute;
	struct ibv_context *ib_ctx;
	struct ibv_comp_channel *send_channel;
	struct ibv_cq *recv_cq;
	struct ibv_cq *send_cq;
	struct ibv_qp *qp;

	struct Queue *usr_send_cq;
	struct Queue *usr_recv_cq;

	char * hostname;
	int inter_port_no;
	int sockfd;

	uint32_t 	hw_timeout;
	int index;
	uint32_t 	message_size;
	uint64_t 	cp_bytes;
	int  event;
	int duration;
	uint32_t 		pending_acks;
	int 			current_psn;
	int 		vegas_previous_ack;

	uint32_t 		current_seq;
	uint32_t        cwnd_size;
	uint32_t 		cwnd_size_cnt; /* Linear increase counter              */

	uint32_t        max_cwnd_size;
	uint32_t        advw_size;
	uint64_t        cwnd_free;
	//uint64_t        free_bytes;
	//uint64_t 	free_slots;
	uint32_t        ss_thresh;
	uint32_t        cwnd_start; /*index to point to the first message in congestion window*/
	uint32_t        cwnd_end; 	/* index to point to the first free slot*/
	uint32_t        expected_ack;
	uint32_t        num_acks;
	uint32_t        rudp_state;
	uint32_t        num_dup_acks;
	uint32_t        last_dup_ack;
	rudp_payload_t  *cwnd;
	void *			cong;

} rudp_srv_state_t;

void tcp_reno_cong_avoid(rudp_srv_state_t  *tp, uint32_t ack, uint32_t acked);
 
static inline int tcp_in_slow_start(const rudp_srv_state_t *tp)
 {
         if(tp->cwnd_size < tp->ss_thresh)
				return 1;
		else
				return 0;
 }


 uint32_t tcp_slow_start(rudp_srv_state_t *tp, uint32_t acked);

 /* The next routines deal with comparing 32 bit unsigned ints
  * and worry about wraparound (automatic with unsigned arithmetic).
  */
 
 static inline int before(uint32_t seq1, uint32_t seq2)
 {
         return ((int)(seq1-seq2)<0)? 1:0;
 }
 #define after(seq2, seq1)       before(seq1, seq2)
 

 /* If cwnd > ssthresh, we may raise ssthresh to be half-way to cwnd.
  * The exception is cwnd reduction phase, when cwnd is decreasing towards
  * ssthresh.
  */
 static inline uint32_t tcp_current_ssthresh(const rudp_srv_state_t *tp)
 {
 
         //if (tcp_in_cwnd_reduction(sk))
         //        return tp->snd_ssthresh;
         //else
                 return GET_MAX(tp->ss_thresh,
                            ((tp->cwnd_size >> 1) +
                             (tp->cwnd_size >> 2)));
 }
