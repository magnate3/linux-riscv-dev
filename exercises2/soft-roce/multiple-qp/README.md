

# To create multiple Queue Pairs in RDMA

```
struct rdma_cm_id *cm_client_id = NULL; 
struct rdma_cm_event *cm_event = NULL;

ret = process_rdma_cm_event(cm_event_channel, RDMA_CM_EVENT_CONNECT_REQUEST, &cm_event);
cm_client_id = cm_event->id;
rdma_create_qp(cm_client_id, pd, &qp_init_attr);
```

# server
##  client resources

```setup_client_resources
   	 * 1. Protection Domains (PD): ibv_alloc_pd
	 * 2. Memory Buffers
	 * 3. Completion Queues (CQ) : ibv_create_cq
	 * 4. Queue Pair (QP) :  rdma_create_qp
```