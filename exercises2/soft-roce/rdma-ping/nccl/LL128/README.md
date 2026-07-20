
NCCL 使用 3 种不同的协议，分别是 LL、LL128 和 Simple，分别具有不同延迟（1us，2us 和 ~6us），不同带宽（50%、95% 和 100%）以及其他影响其性能的差异。时延越长带宽越高，时延越高带宽越长，不同协议在带宽和延迟之间做出了不同的权衡 。

```
 // SIGNAL
  wr[1].opcode                  = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wr[1].send_flags              = IBV_SEND_SIGNALED;
  wr[1].wr_id                   = req - comm->base.reqs;  // used for matching completions with request
  wr[1].next                    = NULL;
  wr[1].wr.atomic.remote_addr   = (uint64_t)signalPtr;
  wr[1].wr.atomic.compare_add   = signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : signalValue;
  wr[1].wr.atomic.rkey          = signalRkey;
  wr[1].sg_list = &sge[1];
  wr[1].num_sge = 1;

  sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;
  sge[1].length = sizeof(comm->putSignalScratchpad);
  sge[1].lkey = comm->devs[devIndex].putSignalScratchpadMr->lkey;
```