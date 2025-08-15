

# rdma_rw_ctx_post  rdma_rw_ctx_wrs

```
static bool nvmet_rdma_execute_command(struct nvmet_rdma_rsp *rsp)
{
        struct nvmet_rdma_queue *queue = rsp->queue;

        if (unlikely(atomic_sub_return(1 + rsp->n_rdma,
                        &queue->sq_wr_avail) < 0)) {
                pr_debug("IB send queue full (needed %d): queue %u cntlid %u\n",
                                1 + rsp->n_rdma, queue->idx,
                                queue->nvme_sq.ctrl->cntlid);
                atomic_add(1 + rsp->n_rdma, &queue->sq_wr_avail);
                return false;
        }

        if (nvmet_rdma_need_data_in(rsp)) {
                if (rdma_rw_ctx_post(&rsp->rw, queue->qp,
                                queue->cm_id->port_num, &rsp->read_cqe, NULL))
                        nvmet_req_complete(&rsp->req, NVME_SC_DATA_XFER_ERROR);
        } else {
                rsp->req.execute(&rsp->req);
        }

        return true;
}
```
# rdma_rw_ctx_signature_init
nvmet_rdma_map_sgl_keyed -->  nvmet_rdma_rw_ctx_init --> rdma_rw_ctx_signature_init    
Keyed SGL Data Block descriptor 是一个 Data Block descriptor，它包括一个用作主机内存访问一部分的密钥。可在 Keyed SGL Data Block descriptor 中指定的最大长度为 (16 MiB – 1)。  
```

        ctx->reg->reg_wr.wr.opcode = IB_WR_REG_MR_INTEGRITY;
        ctx->reg->reg_wr.wr.wr_cqe = NULL;
        ctx->reg->reg_wr.wr.num_sge = 0;
        ctx->reg->reg_wr.wr.send_flags = 0;
        ctx->reg->reg_wr.access = IB_ACCESS_LOCAL_WRITE;
        if (rdma_protocol_iwarp(qp->device, port_num))
                ctx->reg->reg_wr.access |= IB_ACCESS_REMOTE_WRITE;
        ctx->reg->reg_wr.mr = ctx->reg->mr;
        ctx->reg->reg_wr.key = ctx->reg->mr->lkey;
        count++;

        ctx->reg->sge.addr = ctx->reg->mr->iova;
        ctx->reg->sge.length = ctx->reg->mr->length;
        if (sig_attrs->wire.sig_type == IB_SIG_TYPE_NONE)
                ctx->reg->sge.length -= ctx->reg->mr->sig_attrs->meta_length;

        rdma_wr = &ctx->reg->wr;
        rdma_wr->wr.sg_list = &ctx->reg->sge;
        rdma_wr->wr.num_sge = 1;
        rdma_wr->remote_addr = remote_addr;
        rdma_wr->rkey = rkey;
        if (dir == DMA_TO_DEVICE)
                rdma_wr->wr.opcode = IB_WR_RDMA_WRITE;
        else
                rdma_wr->wr.opcode = IB_WR_RDMA_READ;
        ctx->reg->reg_wr.wr.next = &rdma_wr->wr;
```
设置opcodeIB_WR_RDMA_READ or IB_WR_RDMA_WRITE;       

## client write
target向client发起了 Opcode: Reliable Connection (RC) - RDMA READ Request 操作   
client 产生 Reliable Connection (RC) - RDMA READ Response 操作   

## client read
target产生Reliable Connection (RC) - RDMA WRITE XX(比如First)    
# referneces
[NVMe Linux驱动系列二：target端](https://blog.csdn.net/lincolnjunior_lj/article/details/132492082)   