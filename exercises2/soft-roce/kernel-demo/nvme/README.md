# server

## cq

ib_cq_pool_get --> ib_alloc_cqs --> ib_alloc_cq

```C
queue->cq = ib_cq_pool_get(ndev->device, nr_cqe + 1,
                                   queue->comp_vector, IB_POLL_WORKQUEUE);
```

## qp

```C
rdma_create_qp(queue->cm_id, ndev->pd, &qp_attr);
```