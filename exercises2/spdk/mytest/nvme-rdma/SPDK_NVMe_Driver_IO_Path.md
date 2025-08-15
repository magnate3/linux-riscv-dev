# 1 IO Request Submission

```C
// include/spdk/nvme.h
int spdk_nvme_ns_cmd_read(
  struct spdk_nvme_ns *ns, struct spdk_nvme_qpair *qpair, void *payload,
  uint64_t lba, uint32_t lba_count, spdk_nvme_cmd_cb cb_fn,
  void *cb_arg, uint32_t io_flags);

// lib/nvme/nvme_ns_cmd.c
int spdk_nvme_ns_cmd_read(struct spdk_nvme_ns *ns,
                          struct spdk_nvme_qpair *qpair, void *buffer,
                          uint64_t lba, uint32_t lba_count,
                          spdk_nvme_cmd_cb cb_fn, void *cb_arg,
                          uint32_t io_flags) {
  struct nvme_request *req;
  struct nvme_payload payload;

  payload.type = NVME_PAYLOAD_TYPE_CONTIG;
  payload.u.contig = buffer;
  payload.md = NULL;

  req = _nvme_ns_cmd_rw(ns, qpair, &payload, 0, 0, lba, lba_count, cb_fn,
                        cb_arg, SPDK_NVME_OPC_READ, io_flags, 0, 0, true);
  if (req != NULL) {
    return nvme_qpair_submit_request(qpair, req);
  } else {
    return -ENOMEM;
  }
}
```

   Specifically, _nvme_ns_cmd_rw will prepare the payload, and invokes **nvme_allocate_request, which is a slot in device submission queue. If the size of the IO request is large than the namespace’s sectors_per_max_io**, this function will split the request to several child IO commands sent to the device.   

    Now, let’s look at the nvme_qpair_submit_request() function. Because we are using the NVMe SSD, it will go into spdk_pcie_qpair_submit_request, which allocate a tracker from preallocated tracker list.   
	
```
// lib/nvme/nvme_qpair.c
int nvme_qpair_submit_request(struct spdk_nvme_qpair *qpair,
                              struct nvme_request *req) {
  // some handling of child requests ...
  return nvme_transport_qpair_submit_request(qpair, req);
}

// lib/nvme/nvme_transport.c
int nvme_transport_qpair_submit_request(struct spdk_nvme_qpair *qpair,
                                        struct nvme_request *req) {
  NVME_TRANSPORT_CALL(qpair->trtype, qpair_submit_request, (qpair, req));
}

#define NVME_TRANSPORT_CALL(trtype, func_name, args) \
  do {                                               \
    switch (trtype) {                                \
      TRANSPORT_PCIE(func_name, args)                \
      TRANSPORT_FABRICS_RDMA(func_name, args)        \
      TRANSPORT_DEFAULT(trtype)                      \
    }                                                \
    SPDK_UNREACHABLE();                              \
  } while (0)

#define TRANSPORT_PCIE(func_name, args) \
  case SPDK_NVME_TRANSPORT_PCIE:        \
    return nvme_pcie_##func_name args;
```

调用nvme_pcie_qpair_submit_request  
````

// lib/nvme/nvme_pcie.c
int nvme_pcie_qpair_submit_request(struct spdk_nvme_qpair *qpair,
                                   struct nvme_request *req) {

  // ...

  tr = TAILQ_FIRST(&pqpair->free_tr);

  if (tr == NULL || !pqpair->is_enabled) {
    /*
     * No tracker is available, or the qpair is disabled due to
     *  an in-progress controller-level reset.
     *
     * Put the request on the qpair's request queue to be
     *  processed when a tracker frees up via a command
     *  completion or when the controller reset is
     *  completed.
     */
    STAILQ_INSERT_TAIL(&qpair->queued_req, req, stailq);
    goto exit;
  }

  if (tr == NULL || !pqpair->is_enabled) {
    /*
     * No tracker is available, or the qpair is disabled due to
     *  an in-progress controller-level reset.
     *
     * Put the request on the qpair's request queue to be
     *  processed when a tracker frees up via a command
     *  completion or when the controller reset is
     *  completed.
     */
    STAILQ_INSERT_TAIL(&qpair->queued_req, req, stailq);
    goto exit;
  }
  TAILQ_REMOVE(&pqpair->free_tr, tr, tq_list); /* remove tr from free_tr */
  TAILQ_INSERT_TAIL(&pqpair->outstanding_tr, tr, tq_list);
  tr->req = req;
  req->cmd.cid = tr->cid;

  if (req->payload_size && req->payload.md) {
    md_payload = req->payload.md + req->md_offset;
    // mptr is: metadata pointer
    tr->req->cmd.mptr = spdk_vtophys(md_payload);
    if (tr->req->cmd.mptr == SPDK_VTOPHYS_ERROR) {
      nvme_pcie_fail_request_bad_vtophys(qpair, tr);
      rc = -EINVAL;
      goto exit;
    }
  }

  // some SGL request processing

  nvme_pcie_qpair_submit_tracker(qpair, tr);

  // ...
  return rc;
}

static void nvme_pcie_qpair_submit_tracker(struct spdk_nvme_qpair *qpair,
                                           struct nvme_tracker *tr) {
  struct nvme_request *req;
  struct nvme_pcie_qpair *pqpair = nvme_pcie_qpair(qpair);
  struct nvme_pcie_ctrlr *pctrlr = nvme_pcie_ctrlr(qpair->ctrlr);

  tr->timed_out = 0;
  if (spdk_unlikely(qpair->active_proc &&
                    qpair->active_proc->timeout_cb_fn != NULL)) {
    // NOTE: here it does not invoke get_ticks() everytime in hot path
    tr->submit_tick = spdk_get_ticks();
  }

  req = tr->req;
  pqpair->tr[tr->cid].active = true;

  /* Copy the command from the tracker to the submission queue. */
  nvme_pcie_copy_command(&pqpair->cmd[pqpair->sq_tail], &req->cmd);

  if (++pqpair->sq_tail == pqpair->num_entries) {
    pqpair->sq_tail = 0;
  }

  if (pqpair->sq_tail == pqpair->sq_head) {
    SPDK_ERRLOG("sq_tail is passing sq_head!\n");
  }

  /** Write memory barrier */
  // in x86_64, it is using write_barrier (not full read/write barrier)
  // #define spdk_wmb()	__asm volatile("sfence" ::: "memory")
  // but not __asm volatile("mfence" ::: "memory")
  spdk_wmb();
  g_thread_mmio_ctrlr = pctrlr;
  if (spdk_likely(nvme_pcie_qpair_update_mmio_required(qpair, pqpair->sq_tail,
                                                       pqpair->sq_shadow_tdbl,
                                                       pqpair->sq_eventidx))) {
    // sq_tdbl: Submission queue tail doorbell
    spdk_mmio_write_4(pqpair->sq_tdbl, pqpair->sq_tail);
  }
  g_thread_mmio_ctrlr = NULL;
}

static inline void spdk_mmio_write_4(volatile uint32_t *addr, uint32_t val) {
  /** Compiler memory barrier */
  // #define spdk_compiler_barrier() __asm volatile("" ::: "memory")
  spdk_compiler_barrier();
  *addr = val;
}
```

The spdk_nvme_tracker is defined as:   
```
struct nvme_tracker {
  TAILQ_ENTRY(nvme_tracker) tq_list;

  struct nvme_request *req;
  uint16_t cid;

  uint16_t rsvd1 : 14;
  uint16_t timed_out : 1;
  uint16_t active : 1;

  uint32_t rsvd2;

  /* The value of spdk_get_ticks() when the tracker was submitted to the
   * hardware. */
  uint64_t submit_tick;

  uint64_t prp_sgl_bus_addr;

  union {
    uint64_t prp[NVME_MAX_PRP_LIST_ENTRIES];
    struct spdk_nvme_sgl_descriptor sgl[NVME_MAX_SGL_DESCRIPTORS];
  } u;
};
/*
 * struct nvme_tracker must be exactly 4K so that the prp[] array does not cross
 * a page boundary and so that there is no padding required to meet alignment
 * requirements.
 */
SPDK_STATIC_ASSERT(sizeof(struct nvme_tracker) == 4096,
                   "nvme_tracker is not 4K");
SPDK_STATIC_ASSERT((offsetof(struct nvme_tracker, u.sgl) & 7) == 0,
                   "SGL must be Qword aligned");
```
 mentions that Each I/O has 3 data structure associated with it:   
struct perf_task - A data structure which contains the callback that user supplies.
struct nvme_request - transport-agnostic data structure created by the NVMe driver for each I/O. It has three cache lines that are touched in the main I/O path.   
struct nvme_tracker - This is a data structure created by NVMe driver’s PCIe transport for each I/O. It has one cache line that is touched in the main I/O path.   

```
struct nvme_request {
  struct spdk_nvme_cmd cmd;

  /**
   * Data payload for this request's command.
   */
  struct nvme_payload payload;

  uint8_t retries;

  /**
   * Number of children requests still outstanding for this
   *  request which was split into multiple child requests.
   */
  uint16_t num_children;
  uint32_t payload_size;

  /**
   * Offset in bytes from the beginning of payload for this request.
   * This is used for I/O commands that are split into multiple requests.
   */
  uint32_t payload_offset;
  uint32_t md_offset;

  spdk_nvme_cmd_cb cb_fn;
  void *cb_arg;
  STAILQ_ENTRY(nvme_request) stailq;

  struct spdk_nvme_qpair *qpair;

  /**
   * The active admin request can be moved to a per process pending
   *  list based on the saved pid to tell which process it belongs
   *  to. The cpl saves the original completion information which
   *  is used in the completion callback.
   *  these below two fields are only used for admin request.
   */
  pid_t pid;
  struct spdk_nvme_cpl cpl;

  /**
   * The following members should not be reordered with members
   *  above.  These members are only needed when splitting
   *  requests which is done rarely, and the driver is careful
   *  to not touch the following fields until a split operation is
   *  needed, to avoid touching an extra cacheline.
   */

  /**
   * Points to the outstanding child requests for a parent request.
   *  Only valid if a request was split into multiple children
   *  requests, and is not initialized for non-split requests.
   */
  TAILQ_HEAD(, nvme_request) children;

  /**
   * Linked-list pointers for a child request in its parent's list.
   */
  TAILQ_ENTRY(nvme_request) child_tailq;

  /**
   * Points to a parent request if part of a split request,
   *   NULL otherwise.
   */
  struct nvme_request *parent;

  /**
   * Completion status for a parent request.  Initialized to all 0's
   *  (SUCCESS) before child requests are submitted.  If a child
   *  request completes with error, the error status is copied here,
   *  to ensure that the parent request is also completed with error
   *  status once all child requests are completed.
   */
  struct spdk_nvme_cpl parent_status;

  /**
   * The user_cb_fn and user_cb_arg fields are used for holding the original
   * callback data when using nvme_allocate_request_user_copy.
   */
  spdk_nvme_cmd_cb user_cb_fn;
  void *user_cb_arg;
  void *user_buffer;
};

// qpair's submission queue is actually array of spdk_nvme_cmd
struct spdk_nvme_cmd {
  /* dword 0 */
  uint16_t opc : 8;  /* opcode */
  uint16_t fuse : 2; /* fused operation */
  uint16_t rsvd1 : 4;
  uint16_t psdt : 2;
  uint16_t cid; /* command identifier */

  /* dword 1 */
  uint32_t nsid; /* namespace identifier */

  /* dword 2-3 */
  uint32_t rsvd2;
  uint32_t rsvd3;

  /* dword 4-5 */
  uint64_t mptr; /* metadata pointer */

  /* dword 6-9: data pointer */
  union {
    struct {
      uint64_t prp1; /* prp entry 1 */
      uint64_t prp2; /* prp entry 2 */
    } prp;

    struct spdk_nvme_sgl_descriptor sgl1;
  } dptr;

  /* dword 10-15 */
  uint32_t cdw10; /* command-specific */
  uint32_t cdw11; /* command-specific */
  uint32_t cdw12; /* command-specific */
  uint32_t cdw13; /* command-specific */
  uint32_t cdw14; /* command-specific */
  uint32_t cdw15; /* command-specific */
};
```
# IO Completion Polling
Now, let’s look at the completion polling path. The main processing of completion is in nvme_pcie_qpair_process_completions. By default, it will walk through the completion queue entries in an NVMe queue pair’s completion queue and checks for entries whose phase bit has flipper. (See while(1) part). For each entry found, we need to update the completion queue head doorbell.   


```
// lib/nvme/nvme_qpair.c
int32_t spdk_nvme_qpair_process_completions(struct spdk_nvme_qpair *qpair,
                                            uint32_t max_completions) {
  int32_t ret;

  if (qpair->ctrlr->is_failed) {
    nvme_qpair_fail(qpair);
    return 0;
  }

  qpair->in_completion_context = 1;
  ret = nvme_transport_qpair_process_completions(qpair, max_completions);
  qpair->in_completion_context = 0;
  if (qpair->delete_after_completion_context) {
    /*
     * A request to delete this qpair was made in the context of this completion
     *  routine - so it is safe to delete it now.
     */
    spdk_nvme_ctrlr_free_io_qpair(qpair);
  }
  return ret;
}

// lib/nvme/nvme_pcie.c
int32_t nvme_pcie_qpair_process_completions(struct spdk_nvme_qpair *qpair,
                                            uint32_t max_completions) {
  struct nvme_pcie_qpair *pqpair = nvme_pcie_qpair(qpair);
  struct nvme_pcie_ctrlr *pctrlr = nvme_pcie_ctrlr(qpair->ctrlr);
  struct nvme_tracker *tr;
  struct spdk_nvme_cpl *cpl;
  uint32_t num_completions = 0;
  struct spdk_nvme_ctrlr *ctrlr = qpair->ctrlr;

  if (spdk_unlikely(!nvme_pcie_qpair_check_enabled(qpair))) {
    /*
     * qpair is not enabled, likely because a controller reset is
     *  is in progress.  Ignore the interrupt - any I/O that was
     *  associated with this interrupt will get retried when the
     *  reset is complete.
     */
    return 0;
  }

  if (spdk_unlikely(nvme_qpair_is_admin_queue(qpair))) {
    nvme_robust_mutex_lock(&ctrlr->ctrlr_lock);
  }

  if (max_completions == 0 || max_completions > pqpair->max_completions_cap) {
    /*
     * max_completions == 0 means unlimited, but complete at most
     * max_completions_cap batch of I/O at a time so that the completion
     * queue doorbells don't wrap around.
     */
    max_completions = pqpair->max_completions_cap;
  }

  while (1) {
    cpl = &pqpair->cpl[pqpair->cq_head];

    if (cpl->status.p != pqpair->phase) {
      // check the phase bit
      break;
    }

    tr = &pqpair->tr[cpl->cid];
    pqpair->sq_head = cpl->sqhd;

    if (tr->active) {
      nvme_pcie_qpair_complete_tracker(qpair, tr, cpl, true);
    } else {
      SPDK_ERRLOG("cpl does not map to outstanding cmd\n");
      nvme_qpair_print_completion(qpair, cpl);
      assert(0);
    }

    if (spdk_unlikely(++pqpair->cq_head == pqpair->num_entries)) {
      pqpair->cq_head = 0;
      pqpair->phase = !pqpair->phase;
    }

    if (++num_completions == max_completions) {
      break;
    }
  }

  if (num_completions > 0) {
    g_thread_mmio_ctrlr = pctrlr;
    if (spdk_likely(nvme_pcie_qpair_update_mmio_required(
            qpair, pqpair->cq_head, pqpair->cq_shadow_hdbl,
            pqpair->cq_eventidx))) {
      // ring the qpair's completion head doorbell
      spdk_mmio_write_4(pqpair->cq_hdbl, pqpair->cq_head);
    }
    g_thread_mmio_ctrlr = NULL;
  }

  /* We don't want to expose the admin queue to the user,
   * so when we're timing out admin commands set the
   * qpair to NULL.
   */
  if (!nvme_qpair_is_admin_queue(qpair) &&
      spdk_unlikely(qpair->active_proc->timeout_cb_fn != NULL) &&
      qpair->ctrlr->state == NVME_CTRLR_STATE_READY) {
    /*
     * User registered for timeout callback
     */
    nvme_pcie_qpair_check_timeout(qpair);
  }

  /* Before returning, complete any pending admin request. */
  if (spdk_unlikely(nvme_qpair_is_admin_queue(qpair))) {
    nvme_pcie_qpair_complete_pending_admin_request(qpair);

    nvme_robust_mutex_unlock(&ctrlr->ctrlr_lock);
  }

  return num_completions;
}

// nvme_spec.h
// Completion queue entry
struct spdk_nvme_cpl {
	/* dword 0 */
	uint32_t		cdw0;	/* command-specific */
	/* dword 1 */
	uint32_t		rsvd1;
	/* dword 2 */
	uint16_t		sqhd;	/* submission queue head pointer */
	uint16_t		sqid;	/* submission queue identifier */
	/* dword 3 */
	uint16_t		cid;	/* command identifier */
	struct spdk_nvme_status	status;
};

// and the phase tag is in status
struct spdk_nvme_status {
  uint16_t p : 1;   /* phase tag */
  uint16_t sc : 8;  /* status code */
  uint16_t sct : 3; /* status code type */
  uint16_t rsvd2 : 2;
  uint16_t m : 1;   /* more */
  uint16_t dnr : 1; /* do not retry */
};

// Each qpair has its *cpl* field as completion queue
pqpair->cpl =
    spdk_dma_zmalloc(pqpair->num_entries * sizeof(struct spdk_nvme_cpl),
                     page_size, &pqpair->cpl_bus_addr)
```
nvme_pcie_qpair_complete_tracker is the main component to process this completion. And it corresponds to the submission path’s nvme_pcie_qpair_submit_tracker, the definition of the function is shown bellow. The NVMe completion queue is an array of completion queue entries. Inside those entries is a CID value that SPDK provided on command submission. SPDK allocates an array of tracker objects where the index is this CID. Remember, SPDK allows users to queue up more requests than there are actual slots in the submission queue, and that NVMe allows commands to complete in any order after submission.   

```
// nvme_pcie.c
static void nvme_pcie_qpair_complete_tracker(struct spdk_nvme_qpair *qpair,
                                             struct nvme_tracker *tr,
                                             struct spdk_nvme_cpl *cpl,
                                             bool print_on_error) {
  struct nvme_pcie_qpair *pqpair = nvme_pcie_qpair(qpair);
  struct nvme_request *req;
  bool retry, error, was_active;
  bool req_from_current_proc = true;

  req = tr->req;

  assert(req != NULL);

  error = spdk_nvme_cpl_is_error(cpl);
  retry = error && nvme_completion_is_retry(cpl) &&
          req->retries < spdk_nvme_retry_count;

  if (error && print_on_error) {
    nvme_qpair_print_command(qpair, &req->cmd);
    nvme_qpair_print_completion(qpair, cpl);
  }

  was_active = pqpair->tr[cpl->cid].active;
  pqpair->tr[cpl->cid].active = false;

  assert(cpl->cid == req->cmd.cid);

  if (retry) {
    req->retries++;
    nvme_pcie_qpair_submit_tracker(qpair, tr);
  } else {
    if (was_active) {
      /* Only check admin requests from different processes. */
      if (nvme_qpair_is_admin_queue(qpair) && req->pid != getpid()) {
        req_from_current_proc = false;
        nvme_pcie_qpair_insert_pending_admin_request(qpair, req, cpl);
      } else {
        if (req->cb_fn) {
          // NOTE: this is the callback invocation
          req->cb_fn(req->cb_arg, cpl);
        }
      }
    }

    if (req_from_current_proc == true) {
      nvme_free_request(req);
    }

    tr->req = NULL;

    TAILQ_REMOVE(&pqpair->outstanding_tr, tr, tq_list);
    TAILQ_INSERT_HEAD(&pqpair->free_tr, tr, tq_list);

    /*
     * If the controller is in the middle of resetting, don't
     *  try to submit queued requests here - let the reset logic
     *  handle that instead.
     */
    if (!STAILQ_EMPTY(&qpair->queued_req) && !qpair->ctrlr->is_resetting) {
      req = STAILQ_FIRST(&qpair->queued_req);
      STAILQ_REMOVE_HEAD(&qpair->queued_req, stailq);
      nvme_qpair_submit_request(qpair, req);
    }
  }
}
```
# references

[SPDK NVMe Driver I/O Path](https://jingliu.me/posts/2021-01-20-spdkio.html)