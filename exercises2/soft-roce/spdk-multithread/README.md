
# api

```
rctrlr->cm_channel = rdma_create_event_channel();
rdma_get_cm_event：
    STAILQ_INSERT_TAIL(&rctrlr->pending_cm_events, entry, link);
```
处理rctrlr->pending_cm_events   
```
nvme_rdma_poll_events(struct nvme_rdma_ctrlr *rctrlr)
{
	struct nvme_rdma_cm_event_entry	*entry, *tmp;
	struct nvme_rdma_qpair		*event_qpair;
	struct rdma_cm_event		*event;
	struct rdma_event_channel	*channel = rctrlr->cm_channel;

	STAILQ_FOREACH_SAFE(entry, &rctrlr->pending_cm_events, link, tmp) {
		event_qpair = entry->evt->id->context;
		if (event_qpair->evt == NULL) {
			event_qpair->evt = entry->evt;
			STAILQ_REMOVE(&rctrlr->pending_cm_events, entry, nvme_rdma_cm_event_entry, link);
			STAILQ_INSERT_HEAD(&rctrlr->free_cm_events, entry, link);
		}
	}
```