



```
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
```

# timing_wheel of  Carouse
[refer to 谷歌流量整形技术Carousel解读](https://zhuanlan.zhihu.com/p/693128127)     
```
 


  bool EventOnQueueData(SubUcclFlow* subflow, struct wr_ex* wr_ex,
                        uint32_t full_chunk_size, uint64_t now) override {
    return wheel_.queue_on_timing_wheel(
        subflow->pcb.timely_cc.rate_,
        &subflow->pcb.timely_cc.prev_desired_tx_tsc_, now, wr_ex,
        full_chunk_size, subflow->in_wheel_cnt_ == 0);
  }
  // Queue a work request (i.e., one chunk) on the timing wheel.
  // Returns true if the work request was queued on the wheel.
  // Otherwise, the timing wheel was bypassed and the caller can transmit
  // directly.
  inline bool queue_on_timing_wheel()
```


# class SubUcclFlow


```
class IMMData {
 public:
  // HINT: Indicates whether the last chunk of a message.
  // CSN:  Chunk Sequence Number. subflow->pcb.get_snd_nxt
  // RID:  Request ID.
  // FID:  Flow Index.
  // High-----------------32bit------------------Low
  //  | HINT |  RESERVED  |  CSN  |  RID  |  FID  |
  //    1bit      8bit       8bit    7bit    8bit
```

```
 IMMData imm_data(0);

      imm_data.SetFID(flow->flowid());
      if ((*sent_offset + chunk_size == size)) {
        // Last chunk of the message.
        imm_data.SetHINT(1);
      }
      imm_data.SetRID(ureq->send.rid);

      imm_data.SetCSN(subflow->pcb.get_snd_nxt().to_uint32());

      wr->imm_data = htonl(imm_data.GetImmData());
```

```
void RDMAContext::uc_rx_chunk(struct ibv_wc* wc) {
  auto now = rdtsc();
  auto byte_len = wc->byte_len;
  auto imm_data = IMMData(ntohl(wc->imm_data));
  auto qp_num = wc->qp_num;
  auto qpidx = qpn2idx_[qp_num];

  auto last_chunk = imm_data.GetHINT();
  auto csn = imm_data.GetCSN();
  auto rid = imm_data.GetRID();
  auto fid = imm_data.GetFID();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(receiver_flow_tbl_[fid]);
  DCHECK(flow) << fid << ", RDMAContext ptr: " << this;
  auto* subflow = flow->sub_flows_[engine_offset_];

  UCCL_LOG_IO << "Received chunk: (byte_len, csn, rid, fid): " << byte_len
              << ", " << csn << ", " << rid << ", " << fid << " from QP#"
              << qpidx;

  // Locate request by rid
  DCHECK(rid < kMaxReq);
  auto req = get_recvreq_by_id(rid);
  if (req->type != RecvRequest::RECV || req->ureq->context != flow) {
    UCCL_LOG_IO << "Can't find corresponding request or this request is "
                   "invalid for this chunk. Dropping. ";
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  // Compare CSN with the expected CSN.
  auto ecsn = subflow->pcb.rcv_nxt;
  auto distance = UINT_CSN(csn) - ecsn;

  if (UINT_CSN::uintcsn_seqno_lt(UINT_CSN(csn), ecsn)) {
    UCCL_LOG_IO << "Chunk lag behind. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  if (distance.to_uint32() > kReassemblyMaxSeqnoDistance) {
    UCCL_LOG_IO << "Chunk too far ahead. Dropping as we can't handle SACK. "
                << "csn: " << csn << ", ecsn: " << ecsn.to_uint32();
    subflow->pcb.stats_chunk_drop++;
    return;
  }

  // Always use the latest timestamp.
  subflow->pcb.t_remote_nic_rx = now;

  subflow->pcb.sack_bitmap_bit_set(distance.to_uint32());

  auto* msg_size = &req->ureq->recv.elems[0].size;
  uint32_t* received_bytes = req->received_bytes;
  received_bytes[0] += byte_len;

  if (!last_chunk) {
    req = nullptr;
  }

  subflow->rxtracking.ready_csn_.insert({csn, req});

  try_update_csn(subflow);

  if (distance.to_uint32()) {
    subflow->rxtracking.encounter_ooo();
#ifdef STATS
    subflow->pcb.stats_ooo++;
    subflow->pcb.stats_maxooo =
        std::max(subflow->pcb.stats_maxooo, distance.to_uint32());
    if (subflow->rxtracking.real_ooo()) subflow->pcb.stats_real_ooo++;
#endif
  }

  subflow->rxtracking.cumulate_wqe();
  subflow->rxtracking.cumulate_bytes(byte_len);

  if (list_empty(&subflow->ack.ack_link))
    list_add_tail(&subflow->ack.ack_link, &ack_list_);
  subflow->next_ack_path_ = qpidx;

  // Send ACK if needed.
  if (subflow->rxtracking.need_imm_ack()) {
    auto chunk_addr = io_ctx_->pop_ctrl_chunk();
    craft_ack(subflow, chunk_addr, 0);
    try_post_acks(1, chunk_addr, true);

    subflow->rxtracking.clear_imm_ack();
    list_del(&subflow->ack.ack_link);
  }

  EventOnRxData(subflow, &imm_data);
}
```

```
bool RDMAContext::senderCC_tx_message(struct ucclRequest* ureq) {
      auto* flow = reinterpret_cast<UcclFlow*>(ureq->context);
      DCHECK(flow);
      auto* subflow = flow->sub_flows_[engine_offset_];
     // Select QP.
      qpidx = select_qpidx_pot(chunk_size, subflow);
      auto qpw = &dp_qps_[qpidx];

      wr->send_flags = 0;
      if (qpw->signal_cnt_++ % kSignalInterval == 0) {
        wr->send_flags = IBV_SEND_SIGNALED;
      }
      // if (size <= kMaxInline) {
      //   wr->send_flags |= IBV_SEND_INLINE;
      // }
      wr_ex->qpidx = qpidx;

      struct ibv_send_wr* bad_wr;
      DCHECK(ibv_post_send(qpw->qp, wr, &bad_wr) == 0);
```

# use bitmap for tcp Sliding Window 


```
  // SACK bitmap at the receiver side.
  uint64_t sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{};
  uint8_t sack_bitmap_count{0};
  // SACK bitmap at the sender side.
  uint64_t tx_sack_bitmap[kSackBitmapSize / kSackBitmapBucketSize]{};
  uint8_t tx_sack_bitmap_count{0};
  // The starting CSN of the copy of SACK bitmap.
  uint32_t tx_sack_bitmap_base{0};
```

pcb.tx_sack_bitmap   
```
void RDMAContext::uc_rx_ack(struct ibv_cq_ex* cq_ex, UcclSackHdr* ucclsackh) {
  uint64_t t5;
  auto t6 = rdtsc();

  auto fid = ucclsackh->fid.value();
  auto qpidx = ucclsackh->path.value();
  auto ackno = ucclsackh->ackno.value();

  DCHECK(fid < MAX_FLOW);
  auto* flow = reinterpret_cast<UcclFlow*>(sender_flow_tbl_[fid]);
  auto* subflow = flow->sub_flows_[engine_offset_];

  bool update_sackbitmap = false;

  if (UINT_CSN::uintcsn_seqno_lt(ackno, subflow->pcb.snd_una)) {
    UCCL_LOG_IO << "Received old ACK " << ackno << " for flow" << fid << "/"
                << flow << " by Ctrl QP";
  } else if (UINT_CSN::uintcsn_seqno_gt(ackno, subflow->pcb.snd_nxt)) {
    UCCL_LOG_IO << "Received ACK for untransmitted data "
                << "ackno: " << ackno
                << ", snd_nxt: " << subflow->pcb.snd_nxt.to_uint32()
                << " for flow" << fid << "/" << flow << " by Ctrl QP";
  } else if (UINT_CSN::uintcsn_seqno_eq(ackno, subflow->pcb.snd_una)) {
    UCCL_LOG_IO << "Received duplicate ACK " << ackno << " for flow" << fid
                << "/" << flow
                << ", snd_una: " << subflow->pcb.snd_una.to_uint32()
                << " by Ctrl QP";

    EventOnRxNACK(subflow, ucclsackh);

    update_sackbitmap = true;

    subflow->pcb.duplicate_acks++;
    subflow->pcb.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();
    int fast_rexmit_thres = ((is_roce()) ? ROCE_DUP_ACK_THRES : 65536);

    if (subflow->pcb.duplicate_acks < fast_rexmit_thres) {
      // We have not reached the threshold yet, so we do not do
      // retransmission.
    } else if (subflow->pcb.duplicate_acks == fast_rexmit_thres) {
      // Fast retransmit.
      fast_retransmit_for_flow(subflow);
    } else {
      // We have already done the fast retransmit, so we are now
      // in the fast recovery phase.
      auto sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
      // We check the SACK bitmap to see if there are more undelivered
      // chunks. In fast recovery mode we get after a fast
      // retransmit, we will retransmit all missing chunks that we
      // find from the SACK bitmap, when enumerating the SACK bitmap
      // for up to sack_bitmap_count ACKs.
      uint32_t index = 0;
      while (sack_bitmap_count && index < kSackBitmapSize &&
             !subflow->txtracking.empty()) {
        auto bucket_idx = index / PCB::kSackBitmapBucketSize;
        auto sack_bitmap = ucclsackh->sack_bitmap[bucket_idx].value();

        auto cursor = index % PCB::kSackBitmapBucketSize;

        if ((sack_bitmap & (1ULL << cursor)) == 0) {
          // We found a hole.
          auto seqno = subflow->pcb.snd_una + index;
          auto chunk = subflow->txtracking.get_unacked_chunk_from_idx(index);
          if (seqno == chunk.csn) {
            auto wr_ex = chunk.wr_ex;
            if (try_retransmit_chunk(subflow, wr_ex)) {
              subflow->pcb.stats_fast_rexmits++;
            } else {
              // We can't retransmit the chunk due to lack of
              // credits. Quit the loop.
              index = kSackBitmapSize;
            }
          }
          // Rearm timer for Retransmission.
          rearm_timer_for_flow(subflow);
        } else {
          sack_bitmap_count--;
        }
        index++;
      }
    }

  } else {
    UCCL_LOG_IO << "Received valid ACK " << ackno << " for flow" << fid << "/"
                << flow << " by Ctrl QP";

    EventOnRxACK(subflow, ucclsackh);

    update_sackbitmap = true;
    auto num_acked_chunks = UINT_CSN(ackno) - subflow->pcb.snd_una;
    auto remote_queueing_tsc =
        us_to_cycles((ucclsackh->remote_queueing.value()), freq_ghz);
    if constexpr (kTestNoHWTimestamp)
      t5 = t6;
    else
      t5 = convert_nic_to_host(ibv_wc_read_completion_ts(cq_ex));

    DCHECK(engine_offset_ < ucclParamNUM_ENGINES());
    auto reduced_bytes = subflow->unacked_bytes_;
    auto newrtt_tsc = subflow->txtracking.ack_transmitted_chunks(
        subflow, this, num_acked_chunks.to_uint32(), t5, t6,
        remote_queueing_tsc, &subflow->unacked_bytes_);
    reduced_bytes -= subflow->unacked_bytes_;
    *engine_unacked_bytes_ -= reduced_bytes;
    if (qpidx < port_entropy_)
      subflow->update_scoreboard_rtt(newrtt_tsc, qpidx);
    else {
      // This ack is for retransmitted chunk.
      // Don't update scoreboard for retransmitted chunks.
    }

    subflow->pcb.snd_una = ackno;
    subflow->pcb.duplicate_acks = 0;
    subflow->pcb.snd_ooo_acks = 0;
    subflow->pcb.rto_rexmits_consectutive = 0;
    if (!subflow->txtracking.empty()) {
      // Rearm timer if we still have unacked chunks.
      rearm_timer_for_flow(subflow);
    } else {
      disarm_timer_for_flow(subflow);
    }
  }

  // For duplicate ACKs and valid ACKs, we may need to update the SACK bitmap
  // at the sender side.
  if (update_sackbitmap) {
    for (int i = 0; i < kSackBitmapSize / PCB::kSackBitmapBucketSize; i++)
      subflow->pcb.tx_sack_bitmap[i] = ucclsackh->sack_bitmap[i].value();
    subflow->pcb.tx_sack_bitmap_count = ucclsackh->sack_bitmap_count.value();
    subflow->pcb.tx_sack_bitmap_base = ackno;
  }
}
```

# out of order    

```
subflow->pcb.duplicate_acks++;
subflow->pcb.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();
```

#  duplicate_acks
```
subflow->pcb.duplicate_acks++;
subflow->pcb.snd_ooo_acks = ucclsackh->sack_bitmap_count.value();
```

# rtt estimate

```
    DCHECK(engine_offset_ < ucclParamNUM_ENGINES());
    auto reduced_bytes = subflow->unacked_bytes_;
    auto newrtt_tsc = subflow->txtracking.ack_transmitted_chunks(
        subflow, this, num_acked_chunks.to_uint32(), t5, t6,
        remote_queueing_tsc, &subflow->unacked_bytes_);
    reduced_bytes -= subflow->unacked_bytes_;
    *engine_unacked_bytes_ -= reduced_bytes;
    if (qpidx < port_entropy_)
      subflow->update_scoreboard_rtt(newrtt_tsc, qpidx);
    else {
      // This ack is for retransmitted chunk.
      // Don't update scoreboard for retransmitted chunks.
    }
```
 +  how to deal witch ack for  retransmitted chunks?            

```
  inline void update_scoreboard_rtt(uint64_t newrtt_tsc, uint32_t qpidx) {
    scoreboard_rtt_[qpidx] = (1 - kPPEwmaAlpha) * scoreboard_rtt_[qpidx] +
                             kPPEwmaAlpha * to_usec(newrtt_tsc, freq_ghz);
  }
```


# rto retransmit

```
  inline void rto_retransmit_for_flow(void* context) {
    if (is_roce() || kTestLoss) {
      __retransmit_for_flow(context, true);
    }
  }
  
    void __retransmit_for_flow(void* context, bool rto);
  inline void fast_retransmit_for_flow(void* context) {
    if (is_roce() || kTestLoss) {
      __retransmit_for_flow(context, false);
    }
  }
  inline void rto_retransmit_for_flow(void* context) {
    if (is_roce() || kTestLoss) {
      __retransmit_for_flow(context, true);
    }
  }
```