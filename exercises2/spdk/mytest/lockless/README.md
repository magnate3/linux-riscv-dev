
# examples/nvme/perf
创建多个qpair   

```
  for (i = 0; i < ns_ctx->u.nvme.num_all_qpairs; i++) {
                ns_ctx->u.nvme.qpair[i] = spdk_nvme_ctrlr_alloc_io_qpair(entry->u.nvme.ctrlr, &opts,
                                          sizeof(opts));
                qpair = ns_ctx->u.nvme.qpair[i];
                if (!qpair) {
                        printf("ERROR: spdk_nvme_ctrlr_alloc_io_qpair failed\n");
                        goto qpair_failed;
                }

                if (spdk_nvme_poll_group_add(group, qpair)) {
                        printf("ERROR: unable to add I/O qpair to poll group.\n");
                        spdk_nvme_ctrlr_free_io_qpair(qpair);
                        goto qpair_failed;
                }

                if (spdk_nvme_ctrlr_connect_io_qpair(entry->u.nvme.ctrlr, qpair)) {
                        printf("ERROR: unable to connect I/O qpair.\n");
                        spdk_nvme_ctrlr_free_io_qpair(qpair);
                        goto qpair_failed;
                }
        }
```

# nvme over rdma

nvme over rdma,一个connection只产生一个qpair   
 

```

static int
nvmf_rdma_connect(struct spdk_nvmf_transport *transport, struct rdma_cm_event *event)
{
        rqpair = calloc(1, sizeof(struct spdk_nvmf_rdma_qpair));
        if (rqpair == NULL) {
                SPDK_ERRLOG("Could not allocate new connection.\n");
                nvmf_rdma_event_reject(event->id, SPDK_NVMF_RDMA_ERROR_NO_RESOURCES);
                return -1;
        }

        rqpair->device = port->device;
        rqpair->max_queue_depth = max_queue_depth;
        rqpair->max_read_depth = max_read_depth;
        rqpair->cm_id = event->id;
        rqpair->listen_id = event->listen_id;
        rqpair->qpair.transport = transport;
        STAILQ_INIT(&rqpair->ibv_events);
}
```

#  spdk对hardware queue pair的封装
1） IO请求提交函数   
spdk_nvme_ns_cmd_read  
2）生产request  
_nvme_ns_cmd_rw(ns, qpair, &payload, …)   
3）提交请求   
```
nvme_qpair_submmit_request()  
-->nvme_transport_qpair_submit_request(qpair, request)
-->nvme_pcie_qpair_submit_request (qpair, request)
-->nvme_pcie_qpair_build_contig_request(qpair, req, tr);
-->nvme_pcie_qpair_submit_tracker(qpair, tr)
```


```
nvme_pcie_qpair_submit_tracker(struct spdk_nvme_qpair *qpair, struct nvme_tracker *tr)
{
        struct nvme_request     *req;
        struct nvme_pcie_qpair  *pqpair = nvme_pcie_qpair(qpair);
        struct nvme_pcie_ctrlr  *pctrlr = nvme_pcie_ctrlr(qpair->ctrlr);

        req = tr->req;
        assert(req != NULL);
        req->timed_out = false;
        if (spdk_unlikely(pctrlr->ctrlr.timeout_enabled)) {
                req->submit_tick = spdk_get_ticks();
        } else {
                req->submit_tick = 0;
        }

        pqpair->tr[tr->cid].active = true;

        /* Copy the command from the tracker to the submission queue. */
        nvme_pcie_copy_command(&pqpair->cmd[pqpair->sq_tail], &req->cmd);

        if (spdk_unlikely(++pqpair->sq_tail == pqpair->num_entries)) {
                pqpair->sq_tail = 0;
        }

        if (spdk_unlikely(pqpair->sq_tail == pqpair->sq_head)) {
                SPDK_ERRLOG("sq_tail is passing sq_head!\n");
        }

        spdk_wmb();
        if (spdk_likely(nvme_pcie_qpair_update_mmio_required(qpair,
                        pqpair->sq_tail,
                        pqpair->sq_shadow_tdbl,
                        pqpair->sq_eventidx))) {
                g_thread_mmio_ctrlr = pctrlr;
                spdk_mmio_write_4(pqpair->sq_tdbl, pqpair->sq_tail); // <---
                g_thread_mmio_ctrlr = NULL;
        }
}
```
通过上面code path可以看到，spdk nvme 需要操作硬件寄存器，并且一路无锁和原子操作。需要上层保护。
 