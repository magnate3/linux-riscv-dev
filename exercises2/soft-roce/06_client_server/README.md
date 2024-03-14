# RDMA Examples



RDMA example programs I wrote while learning RDMA over Infiniband. 
Using `libibverbs` api and RDMA Connection Manager (RDMA_CM) api



## Examples


| Name                                                                 | Description                                 | Connection Mode     | APIs              |
| -------------------------------------------------------------------- | ------------------------------------------- | ------------------- | ----------------- |
| [echo](https://github.com/jalalmostafa/rdma-examples/tree/main/echo) | Clients sends string, server echoes it back | Reliable Connection | CM + send/receive |

#  rdma_get_cm_event

RDMA_CM_EVENT_ADDR_RESOLVED
Address resolution (rdma_resolve_addr) completed successfully.

RDMA_CM_EVENT_ROUTE_RESOLVED
Route resolution (rdma_resolve_route) completed successfully.
