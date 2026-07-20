#ifndef RXE_DCQCN
#define RXE_DCQCN

#define DCQCN_BC_EXPIRES 1024*1024*10
#define DCQCN_RAI 5
// DCQCN_A = 25600, all scale 25600
#define DCQCN_G 100

static const uint64_t dcqcn_rdtsc_mul = 2200; // 1ns = 2.2 cpu cycles

static const uint64_t dcqcn_min_rate = 10; // 10MB/s is min rate 

static inline int dcqcn_check_credit(struct rxe_qp *qp, int payload) {
    return qp->dcqcn_rate * (rdtsc() - qp->dcqcn_timer) < dcqcn_rdtsc_mul * payload ? 0 : 1;
}

static inline void dcqcn_send_check(struct rxe_qp *qp, struct rxe_pkt_info *pkt, int payload) {
    // don't need dcqcn send check
    if (rxe_opcode[pkt->opcode].offset[RXE_DCQCN_ECN] == 0) {
        return;
    }
    while (!dcqcn_check_credit(qp, payload)) {}
    qp->dcqcn_timer = rdtsc();
    *(uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_DCQCN_ECN]) = 0;

    qp->dcqcn_byte_count += pkt->paylen - rxe_opcode[pkt->opcode].offset[RXE_PAYLOAD] - 4;
    if (qp->dcqcn_byte_count >= DCQCN_BC_EXPIRES) {
        qp->dcqcn_bc++;
        qp->dcqcn_byte_count = 0;
        if (qp->dcqcn_t < 5 && qp->dcqcn_bc < 5) {
            qp->dcqcn_rate = max((qp->dcqcn_rt + qp->dcqcn_rate) / 2, dcqcn_min_rate);
        } else {
            qp->dcqcn_rt += 5;
            qp->dcqcn_rate = max((qp->dcqcn_rt + qp->dcqcn_rate) / 2, dcqcn_min_rate);
        }
    }
}

static inline void dcqcn_recv_pkt(struct rxe_qp *qp, struct rxe_pkt_info *pkt, struct rxe_pkt_info *ack_pkt) {
    uint32_t *ecn_ptr, *ack_ecn_ptr;
    // uint32_t in_depth, out_depth, in_timestamp, out_timestamp, seq_num, *pkt_data_ptr;

    if (rxe_opcode[pkt->opcode].offset[RXE_DCQCN_ECN] == 0) {
        return;
    }

    ecn_ptr = (uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_DCQCN_ECN]);
    ack_ecn_ptr = (uint32_t *)(ack_pkt->hdr + rxe_opcode[ack_pkt->opcode].offset[RXE_DCQCN_ECN]);
    // pkt_data_ptr = (uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_PAYLOAD]);

    // in_depth = ntohl(pkt_data_ptr[0]);
    // out_depth = ntohl(pkt_data_ptr[2]);
    // in_timestamp = ntohl(pkt_data_ptr[1]);
    // out_timestamp = ntohl(pkt_data_ptr[3]);
    // seq_num = ntohl(pkt_data_ptr[4]);

    // if (*ecn_ptr == 0 && out_depth > 2048) {
    //     pr_info("%u %u %u %u %u %u\n", *ecn_ptr, in_depth, out_depth, in_timestamp, out_timestamp, seq_num);
    // }

    // TODO
    if (*ecn_ptr == 0xffffffff && rdtsc() - qp->dcqcn_now_time >= 50 * dcqcn_rdtsc_mul) {
        *ack_ecn_ptr = 0xffffffff;
        qp->dcqcn_now_time = rdtsc();
    } else {
        *ack_ecn_ptr = 0;
    }
}

static inline void dcqcn_recv_ack(struct rxe_qp *qp, struct rxe_pkt_info *pkt) {
    if (qp->dcqcn_timer == 0) {
        return;
    }

    if (*(uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_DCQCN_ECN]) == 0xffffffff) {
        qp->dcqcn_rt = qp->dcqcn_rate;
        qp->dcqcn_rate = max(qp->dcqcn_rt - (qp->dcqcn_A * qp->dcqcn_rt / 25600 / 2), dcqcn_min_rate);
        qp->dcqcn_A = (25600 * qp->dcqcn_A - (qp->dcqcn_A * DCQCN_G) + DCQCN_G) / 25600;
        qp->dcqcn_byte_count = 0;
        qp->dcqcn_last_time = rdtsc();
        qp->dcqcn_t = 0;
        qp->dcqcn_bc = 0;
    }

    if (rdtsc() - qp->dcqcn_last_time >= 55 * dcqcn_rdtsc_mul) {
        qp->dcqcn_A = (25600 * qp->dcqcn_A - (qp->dcqcn_A * DCQCN_G)) / 25600;
        qp->dcqcn_last_time = rdtsc();
        qp->dcqcn_t++;
        if (qp->dcqcn_t < 5 && qp->dcqcn_bc < 5) {
            qp->dcqcn_rate = max((qp->dcqcn_rt + qp->dcqcn_rate) / 2, dcqcn_min_rate);
        } else {
            qp->dcqcn_rt += 5;
            qp->dcqcn_rate = max((qp->dcqcn_rt + qp->dcqcn_rate) / 2, dcqcn_min_rate);
        }
    }
}

#endif