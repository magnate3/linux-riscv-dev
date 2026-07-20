#ifndef RXE_HPCC
#define RXE_HPCC
// static const uint64_t hpcc_rdtsc_mul = 2200; // 1ns = 2.2 cpu cycles
#define HPCC_SNT_NXT 10
#define HPCC_T (30 * 1000)
#define HPCC_WAI 16
#define HPCC_MAXSTAGE 5
#define HPCC_N 9500


static const uint64_t hpcc_min_window = 16384;

static inline int hpcc_check_credit(struct rxe_qp *qp, int payload) {
    return atomic64_read(&qp->hpcc_window) < payload + atomic64_read(&qp->hpcc_flying_bytes) ? 0 : 1;
}

static inline void hpcc_send_check(struct rxe_qp *qp, struct rxe_pkt_info *pkt, int payload) {
    uint32_t *hpcc_header;
    // don't need hpcc send check!
    if (rxe_opcode[pkt->opcode].offset[RXE_HPCC_HEADER] == 0) {
        return;
    }
    while (!hpcc_check_credit(qp, payload)) {}
    atomic64_add(payload, &qp->hpcc_flying_bytes);
    hpcc_header = (uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_HPCC_HEADER]);
    hpcc_header[0] = qp->hpcc_seq;
    hpcc_header[4] = qp->hpcc_B;
    hpcc_header[5] = payload;
    qp->hpcc_seq++;

    // pr_info("send core %d\n", smp_processor_id());
    // pr_info("hpcc_send_pkt: %u %u %u %u %u %u %u\n", hpcc_header[0], hpcc_header[1], hpcc_header[2], hpcc_header[3], hpcc_header[4], hpcc_header[5], hpcc_header[6]);
}


static inline void hpcc_recv_pkt(struct rxe_qp *qp, struct rxe_pkt_info *pkt, struct rxe_pkt_info *ack_pkt) {
    uint32_t *hpcc_header, *ack_hpcc_header;

    // just copy all hpcc header to ack_pkt
    if (rxe_opcode[pkt->opcode].offset[RXE_HPCC_HEADER] == 0) {
        return;
    }
    hpcc_header = (uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_HPCC_HEADER]);
    ack_hpcc_header = (uint32_t *)(ack_pkt->hdr + rxe_opcode[ack_pkt->opcode].offset[RXE_HPCC_HEADER]);
    ack_hpcc_header[0] = hpcc_header[0];
    // these fields are filled by P4
    ack_hpcc_header[1] = ntohl(hpcc_header[1]);
    ack_hpcc_header[2] = ntohl(hpcc_header[2]);
    ack_hpcc_header[3] = ntohl(hpcc_header[3]);
    ack_hpcc_header[4] = hpcc_header[4];
    ack_hpcc_header[5] = hpcc_header[5];
    ack_hpcc_header[6] = ntohl(hpcc_header[6]);

    // uint32_t udp_src_port = *(uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_PAYLOAD]);
    // uint16_t *udp_src_hdr = (uint16_t *)(ack_pkt->hdr - 8);
    // if (udp_src_port != 0) {
    //     *udp_src_hdr = htons((uint16_t)udp_src_port);
    // }
    // pr_info("recv core %d\n", smp_processor_id());

    // pr_info("hpcc_recv_pkt: %u %u %u %u %u %u %u\n", ack_hpcc_header[0], ack_hpcc_header[1], ack_hpcc_header[2], ack_hpcc_header[3], ack_hpcc_header[4], ack_hpcc_header[5], ack_hpcc_header[6]);
}

static inline void hpcc_recv_ack(struct rxe_qp *qp, struct rxe_pkt_info *pkt) {
    uint64_t hpcc_u, hpcc_ts_ns, hpcc_tx_rate, hpcc_tmp_window;
    uint32_t *hpcc_header;

    if (rxe_opcode[pkt->opcode].offset[RXE_HPCC_HEADER] == 0) {
        return;
    }

    hpcc_header = (uint32_t *)(pkt->hdr + rxe_opcode[pkt->opcode].offset[RXE_HPCC_HEADER]);
    if (hpcc_header[0] > qp->hpcc_last_update_seq) {
        hpcc_u = 0;
        hpcc_ts_ns = (hpcc_header[6] - qp->hpcc_ts_carry) * (1 << 18) + hpcc_header[3] - qp->hpcc_ts;
        hpcc_tx_rate = 1000 * (hpcc_header[2] - qp->hpcc_txbyte) / hpcc_ts_ns;
        if (hpcc_header[1] < qp->hpcc_qlen) {
            hpcc_u = 10000 * 80 * 1000 * hpcc_header[1] / (HPCC_T * qp->hpcc_B) + 10000 * hpcc_tx_rate / qp->hpcc_B;
        } else {
            hpcc_u = 10000 * 80 * 1000 * qp->hpcc_qlen / (HPCC_T * qp->hpcc_B) + 10000 * hpcc_tx_rate / qp->hpcc_B;
        }
        if (hpcc_ts_ns > HPCC_T) {
            hpcc_ts_ns = HPCC_T;
        }
        qp->hpcc_U = qp->hpcc_U - qp->hpcc_U * hpcc_ts_ns / HPCC_T + hpcc_ts_ns * hpcc_u / HPCC_T;
        if (qp->hpcc_U >= HPCC_N || qp->hpcc_inc_stage >= HPCC_MAXSTAGE) {
            hpcc_tmp_window = max((uint64_t)atomic64_read(&qp->hpcc_window) * HPCC_N / qp->hpcc_U + HPCC_WAI, hpcc_min_window);
            atomic64_set(&qp->hpcc_window, hpcc_tmp_window);
            qp->hpcc_inc_stage = 0;
        } else {
            hpcc_tmp_window = max((uint64_t)atomic64_read(&qp->hpcc_window) + HPCC_WAI, hpcc_min_window);
            atomic64_set(&qp->hpcc_window, hpcc_tmp_window);
            qp->hpcc_inc_stage++;
        }
        atomic64_sub(hpcc_header[5], &qp->hpcc_flying_bytes);
        qp->hpcc_last_update_seq += HPCC_SNT_NXT;
    } else {
        atomic64_sub(hpcc_header[5], &qp->hpcc_flying_bytes);
    }
    qp->hpcc_txbyte = hpcc_header[2];
    qp->hpcc_qlen = hpcc_header[1];
    qp->hpcc_ts = hpcc_header[3];
    qp->hpcc_B = hpcc_header[4];
    qp->hpcc_ts_carry = hpcc_header[6];

    // uint16_t *udp_src_hdr = (uint16_t *)(pkt->hdr - 8);
    // pr_info("ack core %d %u\n", smp_processor_id(), ntohs(*udp_src_hdr));
}
#endif