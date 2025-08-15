#include "util_rdma.h"
#include "eqds.h"
#include "transport.h"
#include "transport_config.h"
#include "util/util.h"
#include "util_timer.h"
#include <glog/logging.h>
#include <infiniband/verbs.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <sys/mman.h>

namespace uccl {

// RDMAFactory rdma_ctl;
std::shared_ptr<RDMAFactory> rdma_ctl;

int RDMAFactory::init_devs() {
  int num_devs;
  struct ibv_device** devices;

  static std::once_flag init_flag;
  std::call_once(init_flag,
                 []() { rdma_ctl = std::make_shared<RDMAFactory>(); });

  // TODO: Move env vars to a unified place
  char* ib_hca = getenv("NCCL_IB_HCA");
  char* if_name = getenv("NCCL_SOCKET_IFNAME");

  struct ib_dev user_ifs[MAX_IB_DEVS];
  bool searchNot = ib_hca && ib_hca[0] == '^';
  if (searchNot) ib_hca++;

  bool searchExact = ib_hca && ib_hca[0] == '=';
  if (searchExact) ib_hca++;

  int num_ifs = parse_interfaces(ib_hca, user_ifs, MAX_IB_DEVS);
  devices = ibv_get_device_list(&num_devs);
  if (devices == nullptr || num_devs == 0) {
    perror("ibv_get_device_list");
    goto error;
  }

  for (int d = 0; d < num_devs && __num_devices < MAX_IB_DEVS; d++) {
    struct ibv_context* context = ibv_open_device(devices[d]);
    if (context == nullptr) {
      printf("NET/IB : Unable to open device %s", devices[d]->name);
      continue;
    }

    struct ibv_device_attr dev_attr;
    memset(&dev_attr, 0, sizeof(dev_attr));
    if (ibv_query_device(context, &dev_attr)) {
      ibv_close_device(context);
      continue;
    }

    for (int port_num = 1; port_num <= dev_attr.phys_port_cnt; port_num++) {
      struct ibv_port_attr port_attr;
      if (ibv_query_port(context, port_num, &port_attr)) {
        printf("NET/IB : Unable to query port_num %d", port_num);
        ibv_close_device(context);
        continue;
      }

      // Check against user specified HCAs/ports
      if (!(match_if_list(devices[d]->name, port_num, user_ifs, num_ifs,
                          searchExact) ^
            searchNot)) {
        ibv_close_device(context);
        continue;
      }

      if (port_attr.state != IBV_PORT_ACTIVE) {
        ibv_close_device(context);
        continue;
      }

      // Initialize Dev
      struct FactoryDevice dev;
      strncpy(dev.ib_name, devices[d]->name, sizeof(devices[d]->name));

      if (if_name) {
        // Iterate over all interfaces in the list
        auto* if_name_dup = strdup(if_name);
        char* next_intf = strtok(if_name_dup, ",");
        while (next_intf) {
          dev.local_ip_str = get_dev_ip(next_intf);
          if (dev.local_ip_str != "") {
            break;
          }
          next_intf = strtok(nullptr, ",");
        }
        UCCL_INIT_CHECK(dev.local_ip_str != "",
                        "No IP address found for interface");
      } else {
        DCHECK(util_rdma_get_ip_from_ib_name(dev.ib_name, &dev.local_ip_str) ==
               0);
      }

      dev.numa_node = get_dev_numa_node(dev.ib_name);
      dev.dev_attr = dev_attr;
      dev.port_attr = port_attr;
      dev.ib_port_num = port_num;

      double link_bw = (ncclIbSpeed(port_attr.active_speed) *
                        ncclIbWidth(port_attr.active_width)) *
                       1e6 / 8;
      dev.link_bw = link_bw;

      if (port_attr.link_layer == IBV_LINK_LAYER_ETHERNET) {
        dev.gid_idx = ucclParamROCE_GID_IDX();
      } else if (port_attr.link_layer == IBV_LINK_LAYER_INFINIBAND) {
        dev.gid_idx = ucclParamIB_GID_IDX();
      } else {
        printf("Unknown link layer: %d\n", port_attr.link_layer);
        ibv_close_device(context);
        continue;
      }

      dev.context = context;

      if (ibv_query_gid(context, port_num, dev.gid_idx, &dev.gid)) {
        perror("ibv_query_gid");
        ibv_close_device(context);
        continue;
      }

      // Allocate a PD for this device
      dev.pd = ibv_alloc_pd(context);
      if (dev.pd == nullptr) {
        perror("ibv_alloc_pd");
        ibv_close_device(context);
        continue;
      }

      // Detect DMA-BUF support
      {
        struct ibv_pd* pd = ibv_alloc_pd(context);
        if (pd == nullptr) {
          perror("ibv_alloc_pd");
          ibv_close_device(context);
          continue;
        }

        // Test kernel DMA-BUF support with a dummy call (fd=-1)
        (void)ibv_reg_dmabuf_mr(pd, 0ULL /*offset*/, 0ULL /*len*/,
                                0ULL /*iova*/, -1 /*fd*/, 0 /*flags*/);
        dev.dma_buf_support =
            !((errno == EOPNOTSUPP) || (errno == EPROTONOSUPPORT));
        ibv_dealloc_pd(pd);

        UCCL_LOG_RE << "DMA-BUF support: " << dev.dma_buf_support;
      }

      rdma_ctl->devices_.push_back(dev);
      printf("Initialized %s\n", devices[d]->name);
      __num_devices++;
    }
  }
  ibv_free_device_list(devices);
  return __num_devices;

error:
  throw std::runtime_error("Failed to initialize RDMAFactory");
}

/**
 * @brief Create a new RDMA context for a given device running on a specific
 * engine.
 *
 * @param dev
 * @param meta
 * @return RDMAContext*
 */
RDMAContext* RDMAFactory::CreateContext(TimerManager* rto,
                                        uint32_t* engine_unacked_bytes,
                                        eqds::EQDS* eqds, int dev,
                                        uint32_t engine_offset,
                                        union CtrlMeta meta,
                                        SharedIOContext* io_ctx) {
  RDMAContext* ctx = nullptr;

  if constexpr (kReceiverCCA == RECEIVER_CCA_EQDS)
    ctx = new EQDSRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                              engine_offset, meta, io_ctx);
  else if constexpr (kSenderCCA == SENDER_CCA_TIMELY)
    ctx = new TimelyRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                                engine_offset, meta, io_ctx);
  else if constexpr (kSenderCCA == SENDER_CCA_SWIFT)
    ctx = new SwiftRDMAContext(rto, engine_unacked_bytes, eqds, dev,
                               engine_offset, meta, io_ctx);

  CHECK(ctx != nullptr);
  return ctx;
}

std::pair<uint64_t, uint32_t> TXTracking::ack_rc_transmitted_chunks(
    void* subflow_context, RDMAContext* rdma_ctx, UINT_CSN csn, uint64_t now,
    uint32_t* flow_unacked_bytes, uint32_t* engine_outstanding_bytes) {
  auto* subflow = reinterpret_cast<SubUcclFlow*>(subflow_context);
  uint64_t tx_timestamp;
  uint32_t qpidx;

  uint32_t acked_bytes = 0;

  // Traverse unacked_chunks_
  // TODO: we can do more efficiently here.
  for (auto chunk = unacked_chunks_.begin(); chunk != unacked_chunks_.end();
       chunk++) {
    if (chunk->csn == csn.to_uint32()) {
      // We find it!
      chunk->ureq->send.acked_bytes += chunk->wr_ex->sge.length;

      acked_bytes += chunk->wr_ex->sge.length;

      if (chunk->ureq->send.acked_bytes == chunk->ureq->send.data_len) {
        auto poll_ctx = chunk->ureq->poll_ctx;
        // Wakeup app thread waiting one endpoint
        uccl_wakeup(poll_ctx);
        UCCL_LOG_IO << "RC TX message complete";
      }

      *flow_unacked_bytes -= chunk->wr_ex->sge.length;
      *engine_outstanding_bytes -= chunk->wr_ex->sge.length;

      tx_timestamp = chunk->timestamp;
      qpidx = chunk->wr_ex->qpidx;

      // Free wr_ex here.
      rdma_ctx->wr_ex_pool_->free_buff(
          reinterpret_cast<uint64_t>(chunk->wr_ex));

      unacked_chunks_.erase(chunk);
      break;
    }
  }

  auto newrtt_tsc = now - tx_timestamp;

  subflow->pcb.timely_cc.update_rate(now, newrtt_tsc, kEwmaAlpha);

  subflow->pcb.swift_cc.adjust_wnd(to_usec(newrtt_tsc, freq_ghz), acked_bytes);

  return std::make_pair(tx_timestamp, qpidx);
}

uint64_t TXTracking::ack_transmitted_chunks(void* subflow_context,
                                            RDMAContext* rdma_ctx,
                                            uint32_t num_acked_chunks,
                                            uint64_t t5, uint64_t t6,
                                            uint64_t remote_queueing_tsc,
                                            uint32_t* flow_unacked_bytes) {
  DCHECK(num_acked_chunks <= unacked_chunks_.size());

  auto* subflow = reinterpret_cast<SubUcclFlow*>(subflow_context);

  uint64_t t1 = 0;
  uint32_t seg_size = 0;

  while (num_acked_chunks) {
    auto& chunk = unacked_chunks_.front();
    if (chunk.last_chunk) {
      auto poll_ctx = chunk.ureq->poll_ctx;
      // Wakeup app thread waiting one endpoint
      uccl_wakeup(poll_ctx);
      UCCL_LOG_IO << "UC Tx message complete";
    }

    // Record timestamp of the oldest unacked chunk.
    if (t1 == 0) t1 = chunk.timestamp;

    seg_size += chunk.wr_ex->sge.length;

    *flow_unacked_bytes -= chunk.wr_ex->sge.length;

    // Free wr_ex here.
    rdma_ctx->wr_ex_pool_->free_buff(reinterpret_cast<uint64_t>(chunk.wr_ex));

    unacked_chunks_.erase(unacked_chunks_.begin());
    num_acked_chunks--;
  }

  if (unlikely(t5 <= t1)) {
    // Invalid timestamp.
    // We have found that t5 (transferred from NIC timestamp) may be
    // occasionally smaller than t1 (timestamp of the oldest unacked chunk).
    // When this happens, we use software timestamp to fix it.
    t5 = rdtsc();
  }

  auto endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
  auto fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;
  // Make RTT independent of segment size.
  auto serial_delay_tsc =
      us_to_cycles(seg_size * 1e6 / rdma_ctx->link_speed, freq_ghz);
  if (fabric_delay_tsc > serial_delay_tsc ||
      to_usec(fabric_delay_tsc, freq_ghz) < kMAXRTTUS)
    fabric_delay_tsc -= serial_delay_tsc;
  else {
    // Invalid timestamp.
    // Recalculate delay.
    t5 = rdtsc();
    endpoint_delay_tsc = t6 - t5 + remote_queueing_tsc;
    fabric_delay_tsc = (t6 - t1) - endpoint_delay_tsc;
    if (fabric_delay_tsc > serial_delay_tsc)
      fabric_delay_tsc -= serial_delay_tsc;
    else {
      // This may be caused by clock synchronization.
      fabric_delay_tsc = 0;
    }
  }

  UCCL_LOG_IO << "Total: " << to_usec(t6 - t1, freq_ghz)
              << ", Endpoint delay: " << to_usec(endpoint_delay_tsc, freq_ghz)
              << ", Fabric delay: " << to_usec(fabric_delay_tsc, freq_ghz);

  // LOG_EVERY_N(INFO, 10000) << "Host: " <<
  // std::round(to_usec(endpoint_delay_tsc, freq_ghz)) <<
  //     ", Fabric: " << std::round(to_usec(fabric_delay_tsc, freq_ghz));

#ifdef TEST_TURNAROUND_ESTIMATION
  static bool first = true;
  static double avg_turnaround_delay = 0.0;
  static int count = 0;
  auto turnaround_delay = to_usec(remote_queueing_tsc, freq_ghz);

  if (turnaround_delay <
          500 /* filter wrong values (probabaly due to clock sync) */
      && count++ > 5000 /* warmup */) {
    if (first) {
      avg_turnaround_delay = turnaround_delay;
      first = false;
    } else {
      avg_turnaround_delay =
          (avg_turnaround_delay * count + turnaround_delay) / (count + 1);
    }
    LOG_EVERY_N(INFO, 1000)
        << "Turnaround delay: " << turnaround_delay
        << "us, Average turnaround delay: " << avg_turnaround_delay << "us";
  }
#endif

  if (fabric_delay_tsc) {
    // Update global cwnd.
    subflow->pcb.timely_cc.update_rate(t6, fabric_delay_tsc, kEwmaAlpha);
    // TODO: seperate enpoint delay and fabric delay.
    subflow->pcb.swift_cc.adjust_wnd(to_usec(fabric_delay_tsc, freq_ghz),
                                     seg_size);
  }

  return fabric_delay_tsc;
}

void SharedIOContext::check_ctrl_rq(bool force) {
  auto n_post_ctrl_rq = get_post_ctrl_rq_cnt();
  if (!force && n_post_ctrl_rq < kPostRQThreshold) return;

  int post_batch = std::min(kPostRQThreshold, (uint32_t)n_post_ctrl_rq);

  for (int i = 0; i < post_batch; i++) {
    auto chunk_addr = pop_ctrl_chunk();
    ctrl_recv_wrs_.recv_sges[i].addr = chunk_addr;

    CQEDesc* cqe_desc = pop_cqe_desc();
    cqe_desc->data = (uint64_t)chunk_addr;
    ctrl_recv_wrs_.recv_wrs[i].wr_id = (uint64_t)cqe_desc;
    ctrl_recv_wrs_.recv_wrs[i].next =
        (i == post_batch - 1) ? nullptr : &ctrl_recv_wrs_.recv_wrs[i + 1];
  }

  struct ibv_recv_wr* bad_wr;
  DCHECK(ibv_post_recv(ctrl_qp_, &ctrl_recv_wrs_.recv_wrs[0], &bad_wr) == 0);
  UCCL_LOG_IO << "Posted " << post_batch << " recv requests for Ctrl QP";
  dec_post_ctrl_rq(post_batch);
}

void SharedIOContext::check_srq(bool force) {
  auto n_post_srq = get_post_srq_cnt();
  if (!force && n_post_srq < kPostRQThreshold) return;

  int post_batch = std::min(kPostRQThreshold, (uint32_t)n_post_srq);

  for (int i = 0; i < post_batch; i++) {
    if (!is_rc_mode()) {
      auto chunk_addr = pop_retr_chunk();
      dp_recv_wrs_.recv_sges[i].addr = chunk_addr;
      dp_recv_wrs_.recv_sges[i].length = kRetrChunkSize;
      dp_recv_wrs_.recv_sges[i].lkey = get_retr_chunk_lkey();
      dp_recv_wrs_.recv_wrs[i].num_sge = 1;
      dp_recv_wrs_.recv_wrs[i].sg_list = &dp_recv_wrs_.recv_sges[i];
      dp_recv_wrs_.recv_wrs[i].next =
          (i == post_batch - 1) ? nullptr : &dp_recv_wrs_.recv_wrs[i + 1];

      CQEDesc* cqe_desc = pop_cqe_desc();
      cqe_desc->data = (uint64_t)chunk_addr;
      dp_recv_wrs_.recv_wrs[i].wr_id = (uint64_t)cqe_desc;
    } else {
      dp_recv_wrs_.recv_wrs[i].num_sge = 0;
      dp_recv_wrs_.recv_wrs[i].sg_list = nullptr;
      dp_recv_wrs_.recv_wrs[i].next =
          (i == post_batch - 1) ? nullptr : &dp_recv_wrs_.recv_wrs[i + 1];
      dp_recv_wrs_.recv_wrs[i].wr_id = 0;
    }
  }

  struct ibv_recv_wr* bad_wr;
  DCHECK(ibv_post_srq_recv(srq_, &dp_recv_wrs_.recv_wrs[0], &bad_wr) == 0);
  UCCL_LOG_IO << "Posted " << post_batch << " recv requests for SRQ";
  dec_post_srq(post_batch);
}

#ifdef USE_CQ_EX
int SharedIOContext::poll_ctrl_cq(void) {
  auto cq_ex = ctrl_cq_ex_;
  int work = 0;

  int budget = kMaxBatchCQ << 1;

  while (1) {
    struct ibv_poll_cq_attr poll_cq_attr = {};
    if (ibv_start_poll(cq_ex, &poll_cq_attr)) return work;
    int cq_budget = 0;

    while (1) {
      if (cq_ex->status != IBV_WC_SUCCESS) {
        DCHECK(false) << "Ctrl CQ state error: " << cq_ex->status << ", "
                      << ibv_wc_read_opcode(cq_ex)
                      << ", ctrl_chunk_pool_size: " << ctrl_chunk_pool_->size();
      }

      CQEDesc* cqe_desc = reinterpret_cast<CQEDesc*>(cq_ex->wr_id);
      auto chunk_addr = (uint64_t)cqe_desc->data;

      auto opcode = ibv_wc_read_opcode(cq_ex);
      if (opcode == IBV_WC_RECV) {
        auto imm_data = ntohl(ibv_wc_read_imm_data(cq_ex));
        auto num_ack = imm_data;
        UCCL_LOG_IO << "Receive " << num_ack
                    << " ACKs, Chunk addr: " << chunk_addr
                    << ", byte_len: " << ibv_wc_read_byte_len(cq_ex);
        auto base_addr = chunk_addr + UD_ADDITION;
        for (int i = 0; i < num_ack; i++) {
          auto pkt_addr = base_addr + i * CtrlChunkBuffPool::kPktSize;

          auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);
          auto fid = ucclsackh->fid.value();
          auto peer_id = ucclsackh->peer_id.value();
          auto* rdma_ctx = find_rdma_ctx(peer_id, fid);

          rdma_ctx->uc_rx_ack(cq_ex, ucclsackh);
        }
        inc_post_ctrl_rq();
      } else {
        inflight_ctrl_wrs_--;
      }

      push_ctrl_chunk(chunk_addr);

      push_cqe_desc(cqe_desc);

      if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;

      if (opcode == IBV_WC_SEND) {
        // We don't count send WRs in budget.
        cq_budget--;
      }
    }
    ibv_end_poll(cq_ex);

    work += cq_budget;

    check_ctrl_rq(false);

    if (cq_budget < budget) break;
  }

  return work;
}
#else
int SharedIOContext::poll_ctrl_cq(void) {
  auto cq = ibv_cq_ex_to_cq(ctrl_cq_ex_);
  struct ibv_wc wcs[kMaxBatchCQ];

  int cq_budget = 0;
  int budget = kMaxBatchCQ << 1;

  while (1) {
    int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);
    if (nr_wcs == 0) break;

    for (int i = 0; i < nr_wcs; i++) {
      auto* wc = wcs + i;
      DCHECK(wc->status == IBV_WC_SUCCESS)
          << "Ctrl CQ state error: " << wc->status;
      CQEDesc* cqe_desc = (CQEDesc*)wc->wr_id;
      auto chunk_addr = (uint64_t)cqe_desc->data;

      auto opcode = wc->opcode;

      if (opcode == IBV_WC_RECV) {
        auto imm_data = ntohl(wc->imm_data);
        auto num_ack = imm_data;
        UCCL_LOG_IO << "Receive " << num_ack
                    << " ACKs, Chunk addr: " << chunk_addr
                    << ", byte_len: " << wc->byte_len;
        auto base_addr = chunk_addr + UD_ADDITION;
        for (int i = 0; i < num_ack; i++) {
          auto pkt_addr = base_addr + i * CtrlChunkBuffPool::kPktSize;

          auto* ucclsackh = reinterpret_cast<UcclSackHdr*>(pkt_addr);
          auto fid = ucclsackh->fid.value();
          auto peer_id = ucclsackh->peer_id.value();
          auto* rdma_ctx = find_rdma_ctx(peer_id, fid);

          rdma_ctx->uc_rx_ack(ucclsackh);
        }
        inc_post_ctrl_rq();
      } else {
        inflight_ctrl_wrs_--;
      }

      push_ctrl_chunk(chunk_addr);

      push_cqe_desc(cqe_desc);

      if (opcode == IBV_WC_SEND) {
        // We don't count send WRs in budget.
        cq_budget--;
      }
    }

    cq_budget += nr_wcs;

    check_ctrl_rq(false);

    if (cq_budget >= budget) break;
  }

  return cq_budget;
}
#endif

void SharedIOContext::flush_acks() {
  if (nr_tx_ack_wr_ == 0) return;

  tx_ack_wr_[nr_tx_ack_wr_ - 1].next = nullptr;

  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(ctrl_qp_, tx_ack_wr_, &bad_wr);
  DCHECK(ret == 0) << ret << ", nr_tx_ack_wr_: " << nr_tx_ack_wr_;

  UCCL_LOG_IO << "Flush " << nr_tx_ack_wr_ << " ACKs";

  inflight_ctrl_wrs_ += nr_tx_ack_wr_;

  nr_tx_ack_wr_ = 0;
}

#ifdef USE_CQ_EX
int SharedIOContext::rc_poll_recv_cq(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    rdma_ctx->rc_rx_chunk(cq_ex);

    inc_post_srq();

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::rc_poll_send_cq(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    rdma_ctx->rc_rx_ack(cq_ex);

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::uc_poll_send_cq(void) {
  auto cq_ex = send_cq_ex_;
  int cq_budget = 0;
  int budget = kMaxBatchCQ << 1;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;

    if (cqe_desc) {
      // Completion signal from rtx.
      auto retr_hdr = (uint64_t)cqe_desc->data;
      push_retr_hdr(retr_hdr);
      push_cqe_desc(cqe_desc);
    }

    if (++cq_budget == budget || ibv_next_poll(cq_ex)) break;
  }

  ibv_end_poll(cq_ex);

  return cq_budget;
}

int SharedIOContext::uc_poll_recv_cq(void) {
  auto cq_ex = recv_cq_ex_;
  int cq_budget = 0;

  struct ibv_poll_cq_attr poll_cq_attr = {};
  if (ibv_start_poll(cq_ex, &poll_cq_attr)) return 0;

  std::vector<RDMAContext*> rdma_ctxs;

  while (1) {
    if (cq_ex->status != IBV_WC_SUCCESS) {
      DCHECK(false) << "data path CQ state error: " << cq_ex->status
                    << " from QP:" << ibv_wc_read_qp_num(cq_ex);
    }

    auto* rdma_ctx = qpn_to_rdma_ctx(ibv_wc_read_qp_num(cq_ex));

    auto* cqe_desc = (CQEDesc*)cq_ex->wr_id;
    auto chunk_addr = (uint64_t)cqe_desc->data;
    auto opcode = ibv_wc_read_opcode(cq_ex);

    if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
      // Common case.
      rdma_ctx->uc_rx_chunk(cq_ex);
    } else {
      // Rare case.
      rdma_ctx->uc_rx_rtx_chunk(cq_ex, chunk_addr);
    }

    rdma_ctxs.push_back(rdma_ctx);

    push_retr_chunk(chunk_addr);

    push_cqe_desc(cqe_desc);

    inc_post_srq();

    if (++cq_budget == kMaxBatchCQ || ibv_next_poll(cq_ex)) break;
  }
  ibv_end_poll(cq_ex);

  for (auto rdma_ctx : rdma_ctxs) {
    rdma_ctx->uc_post_acks();
  }

  flush_acks();

  return cq_budget;
}
#else
int SharedIOContext::rc_poll_send_cq(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(send_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "RC send CQ state error: " << wc->status;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);
    rdma_ctx->rc_rx_ack(wc);
  }

  return nr_wcs;
}

int SharedIOContext::rc_poll_recv_cq(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(recv_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "RC recv CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);
    rdma_ctx->rc_rx_chunk(wc->byte_len, wc->imm_data);

    inc_post_srq();
  }

  return nr_wcs;
}

int SharedIOContext::uc_poll_send_cq(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(send_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "UC send CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;

    if (cqe_desc) {
      // Completion signal from rtx.
      auto retr_hdr = (uint64_t)cqe_desc->data;
      push_retr_hdr(retr_hdr);
      push_cqe_desc(cqe_desc);
    }
  }

  return nr_wcs;
}

int SharedIOContext::uc_poll_recv_cq(void) {
  struct ibv_wc wcs[kMaxBatchCQ];

  auto* cq = ibv_cq_ex_to_cq(recv_cq_ex_);

  int nr_wcs = ibv_poll_cq(cq, kMaxBatchCQ, wcs);

  std::vector<RDMAContext*> rdma_ctxs;

  for (int i = 0; i < nr_wcs; i++) {
    auto* wc = wcs + i;
    DCHECK(wc->status == IBV_WC_SUCCESS)
        << "UC recv CQ state error: " << wc->status;
    auto* cqe_desc = (CQEDesc*)wc->wr_id;
    auto* rdma_ctx = qpn_to_rdma_ctx(wc->qp_num);

    auto chunk_addr = (uint64_t)cqe_desc->data;
    auto opcode = wc->opcode;

    if (likely(opcode == IBV_WC_RECV_RDMA_WITH_IMM)) {
      // Common case.
      rdma_ctx->uc_rx_chunk(wc);
    } else {
      // Rare case.
      rdma_ctx->uc_rx_rtx_chunk(wc, chunk_addr);
    }

    rdma_ctxs.push_back(rdma_ctx);

    push_retr_chunk(chunk_addr);

    push_cqe_desc(cqe_desc);

    inc_post_srq();
  }

  for (auto rdma_ctx : rdma_ctxs) {
    rdma_ctx->uc_post_acks();
  }

  flush_acks();

  return nr_wcs;
}
#endif

}  // namespace uccl
