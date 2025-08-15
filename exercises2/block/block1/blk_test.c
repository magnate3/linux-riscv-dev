active側                      passive側
--------                      ---------
rdma_create_id                rdma_create_id
  ibv_get_device_list           ibv_get_device_list

rdma_resolve_addr             rdma_bind_addr
  ibv_device_open               ibv_device_open
  ibv_alloc_pd                  ibv_alloc_pd

rdma_resolve_route
                     
rdma_create_qp
  ibv_create_comp_channel
  ibv_create_cq
  ibv_create_qp
  ibv_modify_qp(RESET->INIT)

rdma_reg_msgs
  ibv_reg_mr
rdma_post_recv
  ibv_post_recv
                              rdma_listen
rdma_connect  --------------> rdma_get_request
     .
     .                        rdma_create_qp
     .                          ibv_create_comp_channel
     .                          ibv_create_cq
     .                          ibv_create_qp
     .                          ibv_modify_qp(RESET->INIT)
     .
     .                        rdma_reg_msgs
     .                          ibv_reg_mr
     .                        rdma_post_recv
     .                          ibv_post_recv
     .
  rdma_get_cm_event <-------  rdma_accept
    ibv_modify_qp(INIT->RTR)    ibv_modify_qp(INIT->RTR)
    ibv_modify_qp(RTR->RTS)     ibv_modify_qp(RTR->RTS)
  rdma_ack_cm_event

rdma_post_send
  ibv_post_send
rdma_get_send_comp            rdma_get_recv_comp
  ibv_poll_cq                   ibv_poll_cq