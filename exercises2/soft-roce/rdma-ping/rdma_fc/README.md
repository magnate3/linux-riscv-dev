# rdma_fc
[RDMA Flow Control Samples](https://github.com/michaelbe2/rdma_fc/tree/master)



#  dc_read_fc


```
/* send control message */
static int send_control(connection_t *conn, enum packet_type type,
                        unsigned conn_index)
{
    packet_t packet = { .type = type,
                        .conn_index = conn_index };
    int dci, ret;

    dci = rand() % NUM_DCI;
    assert(g_test.dci_outstanding[dci] < g_options.tx_queue_len);

    LOG_TRACE("send_control: ibv_wr_start: qpex = %p\n", g_test.dcis_ex[dci]);
    ibv_wr_start(g_test.dcis_ex[dci]);

    g_test.dcis_ex[dci]->wr_id = (uint64_t)dci;
    g_test.dcis_ex[dci]->wr_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;

    LOG_TRACE("send_control: ibv_wr_send: wr_id=0x%lx, qpex=%p\n", g_test.dcis_ex[dci]->wr_id, g_test.dcis_ex[dci]);
    ibv_wr_send(g_test.dcis_ex[dci]);

    LOG_TRACE("send_control: mlx5dv_wr_set_dc_addr: mqpex=%p, ah=%p, rem_dctn=0x%06x\n",
               g_test.m_dcis_ex[dci], conn->dc_ah, conn->remote_dctn);
    mlx5dv_wr_set_dc_addr(g_test.m_dcis_ex[dci], conn->dc_ah, conn->remote_dctn, DC_KEY);

    LOG_TRACE("send_control: ibv_wr_set_inline_data: qpex=%p, lkey=0, local_buf=%p, size=%u\n",
              g_test.dcis_ex[dci], &packet, (uint32_t)sizeof(packet));
    ibv_wr_set_inline_data(g_test.dcis_ex[dci], &packet, (uint32_t)sizeof(packet));
    
//    LOG_TRACE("send_control: ibv_wr_set_sge: qpex=%p, lkey=0, local_buf=%p, size=%u\n",
//              g_test.dcis_ex[dci], &packet, (uint32_t)sizeof(packet));
//    ibv_wr_set_sge(g_test.dcis_ex[dci], 0 /*mr->lkey*/, (uintptr_t)&packet, (uint32_t)sizeof(packet));

    LOG_TRACE("send_control: ibv_wr_complete: qpex=%p\n", g_test.dcis_ex[dci]);
    ret = ibv_wr_complete(g_test.dcis_ex[dci]);
    if (ret) {
        LOG_ERROR("send_control: ibv_wr_complete failed (error=%d)\n", ret);
        return -1;
    }
    
    ++g_test.dci_outstanding[dci];

    LOG_TRACE("Sent packet %d conn_index %d", packet.type, packet.conn_index);
    return 0;
}

```

+ server


```
./dc_read_fc   -G 3 -p 9999 -n 1 
dc_read_fc.c:1341 INFO  Waiting for 1 connections...
dc_read_fc.c:1299 INFO  Total read bandwidth: 7355.32 MB/s
dc_read_fc.c:1082 INFO  Disconnecting 1 connections
```


+ client     


```
./dc_read_fc 10.22.116.220  -G 3 -p 9999 -n 1 
dc_read_fc.c:1116 INFO  Connection[0] to 10.22.116.220...
dc_read_fc.c:1082 INFO  Disconnecting 1 connections
root@ljtest2:~/rdma-bench/rdma_fc/dc_read_fc# ./dc_read_fc 10.22.116.220  -G 3 -p 9999 -n 1 -v
dc_read_fc.c:1116 INFO  Connection[0] to 10.22.116.220...
dc_read_fc.c:954  DEBUG Got rdma_cm event RDMA_CM_EVENT_ADDR_RESOLVED
dc_read_fc.c:954  DEBUG Got rdma_cm event RDMA_CM_EVENT_ROUTE_RESOLVED
dc_read_fc.c:699  DEBUG Created CQ @0x55d23a5ff540
dc_read_fc.c:716  DEBUG Created SRQ @0x55d23a5ff848
dc_read_fc.c:290  DEBUG Registered buffer 0x55d23a619000 length 1024 lkey 0x1bfefd rkey 0x1bfefd
dc_read_fc.c:290  DEBUG Registered buffer 0x7f75ad017000 length 1048576 lkey 0xa4645 rkey 0xa4645
dc_read_fc.c:675  DEBUG Posted 128 receives
dc_read_fc.c:521  DEBUG mlx5dv_create_qp(0x7f75ad118150,0x7fff2fe47570,0x7fff2fe474d0)
```
