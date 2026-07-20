# RDMA pingpongs

This directory contains pingpong programs based on RDMA.
The [rdma-core](https://github.com/linux-rdma/rdma-core/tree/master/libibverbs/examples) examples were used as a
guideline to develop these programs, which were deeply modified to integrate with the existing codebase.
Credits to the original authors for the original code.

## Pingpong design

The original pingpong programs in `rdma-core` follow a synchronous model, where the client sends a message and waits for
the server to respond before starting the next pingpong round.

However, in order to have a fair and truthful comparison between different technologies, we prefer having an
asynchronous model: the client has a thread that keeps sending packets at a fixed interval.

## Available programs

- [rc_pingpong](rc_pingpong.c): A RDMA pingpong using the Reliable Connection (RC) transport.

  A Reliable Connection (RC) is a connection-oriented transport that provides reliable, in-order delivery of messages.
  This transport type is very similar to TCP in the TCP/IP stack.

  Using RC transport to do the pingpong experiment in the asynchronous way (i.e., the client sends packets at a fixed
  interval), is pretty tricky: the client must wait for the completion of the previous message before sending another
  one (because of the definition of the reliable connection). So, if the ACK for each message is waited, then the send
  interval will not be the wanted one; if the ACK is not waited, then the pingpong becomes unpredictable, hoping that
  the ACK is received within the time interval. For this reason, RC is not the best transport to use in our case.

- [ud_pingpong](ud_pingpong.c): A RDMA pingpong using the Unreliable Datagram (UD) transport.

  An Unreliable Datagram (UD) is a connectionless transport that provides unreliable, unordered delivery of messages.
  This transport type is very similar to UDP in the TCP/IP stack.

  The UD transport is the best transport to use in our case, since it is connectionless and provides unreliable,
  unordered
  delivery of messages. This means that the client can send packets at a fixed interval without waiting for the
  completion of the previous message.

  It also represents the most fair comparison with other technologies which are based on UDP.

### Unavailable programs

The following transport types exist and are available in the original repository, but they were not adapted (yet) to
our use case:

- `uc_pingpong`: A RDMA pingpong using the Unreliable Connection (UC) transport.

  An Unreliable Connection (UC) is a connection-oriented transport that provides unreliable, unordered delivery of
  messages.
  It is basically a middle ground between RC and UD; a connection is established between QPs, but the delivery of
  messages is unreliable.
- `xrc_pingpong`: A RDMA pingpong using the Extended Reliable Connection (XRC) transport.
- `srq_pingpong`: A RDMA pingpong using the Shared Receive Queue (SRQ) transport.

## Build

The build process follows the same steps as for any CMake project, with a variable SERVER to indicate if the program
is the server or the client.

```bash
cd build
cmake -D SERVER=<0,1> ..
make
```
+ server   
``` 
cmake -D SERVER=0
```


+ client    
```
cmake -D SERVER=1
```
There is the option to build the programs using a `DEBUG` mode, which has extra logging and debugging information:

```bash
cd build
cmake -D DEBUG=1 -D SERVER=<0,1> ..
make
```

## ibv_devinfo   

```
root@test:~/rdma# rdma link
link irdma0/1 state DOWN physical_state DISABLED netdev enp23s0f0 
link irdma1/1 state ACTIVE physical_state LINK_UP netdev enp23s0f1 
root@test:~/rdma# ibv_devinfo -d irdma1  -v
hca_id: irdma1
        transport:                      InfiniBand (0)
        fw_ver:                         1.54
        node_guid:                      6efe:54ff:fe3d:8a39
        sys_image_guid:                 6cfe:543d:8a39:0000
        vendor_id:                      0x8086
        vendor_part_id:                 5522
        hw_ver:                         0x2
        phys_port_cnt:                  1
        max_mr_size:                    0x200000000000
        page_size_cap:                  0x40201000
        max_qp:                         65533
        max_qp_wr:                      4063
        device_cap_flags:               0x00229000
                                        RC_RNR_NAK_GEN
                                        MEM_WINDOW
                                        MEM_MGT_EXTENSIONS
                                        Unknown flags: 0x8000
        max_sge:                        13
        max_sge_rd:                     13
        max_cq:                         131069
        max_cqe:                        1048574
        max_mr:                         2097150
        max_pd:                         262141
        max_qp_rd_atom:                 127
        max_ee_rd_atom:                 0
        max_res_rd_atom:                0
        max_qp_init_rd_atom:            255
        max_ee_init_rd_atom:            0
        atomic_cap:                     ATOMIC_NONE (0)
        max_ee:                         0
        max_rdd:                        0
        max_mw:                         2097150
        max_raw_ipv6_qp:                0
        max_raw_ethy_qp:                0
        max_mcast_grp:                  65536
        max_mcast_qp_attach:            8
        max_total_mcast_qp_attach:      524288
        max_ah:                         131072
        max_fmr:                        0
        max_srq:                        0
        max_pkeys:                      1
        local_ca_ack_delay:             0
        general_odp_caps:
        rc_odp_caps:
                                        NO SUPPORT
        uc_odp_caps:
                                        NO SUPPORT
        ud_odp_caps:
                                        NO SUPPORT
        xrc_odp_caps:
                                        NO SUPPORT
        completion timestamp_mask:                      0x000000000001ffff
        core clock not supported
        device_cap_flags_ex:            0x229000
        tso_caps:
                max_tso:                        0
        rss_caps:
                max_rwq_indirection_tables:                     0
                max_rwq_indirection_table_size:                 0
                rx_hash_function:                               0x0
                rx_hash_fields_mask:                            0x0
        max_wq_type_rq:                 0
        packet_pacing_caps:
                qp_rate_limit_min:      0kbps
                qp_rate_limit_max:      0kbps
        tag matching not supported
        num_comp_vectors:               48
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               1
                        port_lmc:               0x00
                        link_layer:             Ethernet
                        max_msg_sz:             0x7fffffff
                        port_cap_flags:         0x04050000
                        port_cap_flags2:        0x0000
                        max_vl_num:             invalid value (0)
                        bad_pkey_cntr:          0x0
                        qkey_viol_cntr:         0x0
                        sm_sl:                  0
                        pkey_tbl_len:           1
                        gid_tbl_len:            32
                        subnet_timeout:         0
                        init_type_reply:        0
                        active_width:           4X (2)
                        active_speed:           25.0 Gbps (32)
                        phys_state:             LINK_UP (5)
                        GID[  0]:               fe80::6efe:54ff:fe3d:8a39, RoCE v2
                        GID[  1]:               ::ffff
```

```
ibv_devinfo  show irdma1
hca_id: irdma0
        transport:                      InfiniBand (0)
        fw_ver:                         1.54
        node_guid:                      6efe:54ff:fe3d:ba88
        sys_image_guid:                 6cfe:543d:ba88:0000
        vendor_id:                      0x8086
        vendor_part_id:                 5522
        hw_ver:                         0x2
        phys_port_cnt:                  1
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               1
                        port_lmc:               0x00
                        link_layer:             Ethernet

```

## Run

The programs require at least two nodes to run, one acting as the server and the other as the client.

The following example shows how to run the `ud_pingpong` experiment.

First, the server must be started, which will wait a connection from a client:

```bash
./ud_pingpong -d <ib device name> -g <port gid index> -p <pingpong rounds>
# Example:
# ./ud_pingpong -d rocep65s0f0 -g 0 -p 1000
```

Then, the client must be started, which will connect to the server and start the pingpong:

```bash
./ud_pingpong -d <ib device name> -g <port gid index> -p <pingpong rounds> -i <send interval> -s <server IP>
# Example:
# ./ud_pingpong -d rocep65s0f0 -g 0 -p 1000 -i 1000000 -s 10.10.1.2
```


```
./ud_pingpong -d irdma1  -g  1  -p 3  -i 1 -s 10.22.116.221
```

Note that the CLI interface was adapted to the other programs in the project and is different from the one in the
`rdma-core` examples.

## Results

By default, the results of the experiments are saved in a file named after the transport type (e.g. `ud_pingpong`
creates a file `ud.dat`).
In the code, the "persistence agent" can be modified to print results to stdout, to a different file, or to store
different information about the results.