

#  ibv_wr_set_sge_list



#   old    

```
     struct ibv_send_wr wr = {
            .wr_id = (uint64_t)i,
            .sg_list = list,
            .num_sge = 2,
            .opcode = IBV_WR_SEND,
            .send_flags = flags,
        };
```