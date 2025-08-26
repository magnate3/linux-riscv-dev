#ifndef RXE_TIMELY
#define RXE_TIMELY

static const uint64_t timely_rdtsc_mul = 2200; // 1ns = 2.2 cpu cycles

// TODO THIS NEED MORE TEST FOR PARAMS!
#define TIMELY_MINRTT (30 * timely_rdtsc_mul) 
#define TIMELY_T_LOW (50 * timely_rdtsc_mul) 
#define TIMELY_T_HIGH (750 * timely_rdtsc_mul)
#define TIMELY_RAI   5 
#define TIMELY_A 875
#define TIMELY_B 800
static const uint64_t timely_min_rate = 10; // 10MB/s is min rate 



static inline int timely_check_credit(struct rxe_qp *qp, int payload) {
    return qp->timely_rate * (rdtsc() - qp->timely_timer) < timely_rdtsc_mul * payload ? 0 : 1;
}

static inline void timely_send_check(struct rxe_qp *qp, struct rxe_pkt_info *pkt, int payload) {
    // don't need timely send check!
    if (rxe_opcode[pkt->opcode].offset[RXE_TIMELY_TIMESTAMP] == 0) {
        return;
    }
    while (!timely_check_credit(qp, payload)) {}
    qp->timely_timer = rdtsc();
    *(uint64_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_TIMELY_TIMESTAMP]) = qp->timely_timer;
}

static inline void timely_recv_pkt(struct rxe_qp *qp, struct rxe_pkt_info *pkt, struct rxe_pkt_info *ack_pkt) {
    // add timestamp for calculate
    // need to add timestamp
    uint64_t pkt_timestamp;
    if (rxe_opcode[pkt->opcode].offset[RXE_TIMELY_TIMESTAMP] == 0) {
        return;
    }
    pkt_timestamp = *(uint64_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_TIMELY_TIMESTAMP]);
    *(uint64_t *)(ack_pkt->hdr + rxe_opcode[ack_pkt->opcode].offset[RXE_TIMELY_TIMESTAMP]) = pkt_timestamp;
}

static inline void timely_recv_ack(struct rxe_qp *qp, struct rxe_pkt_info *pkt) {
    uint64_t new_rtt, new_rate;
    int64_t new_rtt_diff;
    // don't need timely recv ack!
    if (qp->timely_timer == 0) {
        return;
    }

    new_rtt = rdtsc() - *(uint64_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_TIMELY_TIMESTAMP]);
    new_rtt_diff = new_rtt - qp->timely_prev_rtt;
    qp->timely_prev_rtt = new_rtt;

    qp->timely_rtt_diff = (qp->timely_rtt_diff * (1000 - TIMELY_A) + new_rtt_diff * TIMELY_A) / 1000;

    if (new_rtt < TIMELY_T_LOW) {
        qp->timely_rate += TIMELY_RAI;
    } else if (new_rtt > TIMELY_T_HIGH) {
        new_rate = qp->timely_rate * (1000 - TIMELY_B + TIMELY_B * TIMELY_T_HIGH / new_rtt) / 1000;
        qp->timely_rate = max(new_rate, timely_min_rate);
    } else if (qp->timely_rtt_diff <= 0) {
        qp->timely_rate += 5 * TIMELY_RAI;
    } else {
        new_rate = (1000 * qp->timely_rate - TIMELY_B * qp->timely_rate * qp->timely_rtt_diff / TIMELY_MINRTT) / 1000;
        qp->timely_rate = max(new_rate, timely_min_rate);
    }
}

#endif