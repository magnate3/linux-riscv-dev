
# ibv_devinfo -d rxe0 -v
```
[root@centos7 RDMA_RC_example]#  ibv_devinfo -d rxe0 -v
hca_id: rxe0
        transport:                      InfiniBand (0)
        fw_ver:                         0.0.0
        node_guid:                      b208:75ff:fe5f:b85e
        sys_image_guid:                 0000:0000:0000:0000
        vendor_id:                      0x0000
        vendor_part_id:                 0
        hw_ver:                         0x0
        phys_port_cnt:                  1
        max_mr_size:                    0xffffffffffffffff
        page_size_cap:                  0xfffff000
        max_qp:                         65536
        max_qp_wr:                      16384
        device_cap_flags:               0x00203c76
                                        BAD_PKEY_CNTR
                                        BAD_QKEY_CNTR
                                        AUTO_PATH_MIG
                                        CHANGE_PHY_PORT
                                        UD_AV_PORT_ENFORCE
                                        PORT_ACTIVE_EVENT
                                        SYS_IMAGE_GUID
                                        RC_RNR_NAK_GEN
                                        SRQ_RESIZE
                                        MEM_MGT_EXTENSIONS
        max_sge:                        32
        max_sge_rd:                     32
        max_cq:                         16384
        max_cqe:                        32767
        max_mr:                         2048
        max_pd:                         32764
        max_qp_rd_atom:                 128
        max_ee_rd_atom:                 0
        max_res_rd_atom:                258048
        max_qp_init_rd_atom:            128
        max_ee_init_rd_atom:            0
        atomic_cap:                     ATOMIC_HCA (1)
        max_ee:                         0
        max_rdd:                        0
        max_mw:                         0
        max_raw_ipv6_qp:                0
        max_raw_ethy_qp:                0
        max_mcast_grp:                  8192
        max_mcast_qp_attach:            56
        max_total_mcast_qp_attach:      458752
        max_ah:                         100
        max_fmr:                        0
        max_srq:                        960
        max_srq_wr:                     16384
        max_srq_sge:                    27
        max_pkeys:                      64
        local_ca_ack_delay:             15
        general_odp_caps:
        rc_odp_caps:
                                        NO SUPPORT
        uc_odp_caps:
                                        NO SUPPORT
        ud_odp_caps:
                                        NO SUPPORT
        completion_timestamp_mask not supported
        core clock not supported
        device_cap_flags_ex:            0x0
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
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
                        max_msg_sz:             0x800000
                        port_cap_flags:         0x00890000
                        port_cap_flags2:        0x0000
                        max_vl_num:             1 (1)
                        bad_pkey_cntr:          0x0
                        qkey_viol_cntr:         0x0
                        sm_sl:                  0
                        pkey_tbl_len:           64
                        gid_tbl_len:            1024
                        subnet_timeout:         0
                        init_type_reply:        0
                        active_width:           1X (1)
                        active_speed:           2.5 Gbps (1)
                        phys_state:             LINK_UP (5)
                        GID[  0]:               fe80:0000:0000:0000:b208:75ff:fe5f:b85e
                        GID[  1]:               0000:0000:0000:0000:0000:ffff:0a0b:0bfb
```

# server

```
[root@centos7 RDMA_RC_example]# ./demo -d rxe0 -g 1
 ------------------------------------------------
 Device name    : "rxe0"
 IB port        : 1
 TCP port       : 19875
 GID index      : 1
 ------------------------------------------------

[Server only] waiting on port 19875 for TCP connection
TCP connection was established
searching for IB devices in host
found 2 device(s)
[Server only] going to send the message: '***  SEND operation        ***'
MR was registered with addr=0x2b5d2d70, lkey=0x6b0, rkey=0x6b0, flags=0x7
QP was created, QP number=0x11

Local LID       = 0x0
Remote address = 0xd518610
Remote rkey = 0x52c
Remote QP number = 0x11
Remote LID = 0x0
Remote GID = 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:0b:0b:52
QP state was change to RTS
Send Request was posted
completion was found in CQ with status 0x0
[Server only] Contents of server buffer: '***  RDMA Write operation  ***'

test result is 0
```

# client

```
./demo  -d rxe0 -g 1 10.11.11.251
 ------------------------------------------------
 Device name    : "rxe0"
 IB port        : 1
[client only] IP        : 10.11.11.251
 TCP port       : 19875
 GID index      : 1
 ------------------------------------------------

TCP connection was established
searching for IB devices in host
found 1 device(s)
MR was registered with addr=0xd518610, lkey=0x52c, rkey=0x52c, flags=0x7
QP was created, QP number=0x11

Local LID       = 0x0
Remote address = 0x2b5d2d70
Remote rkey = 0x6b0
Remote QP number = 0x11
Remote LID = 0x0
Remote GID = 00:00:00:00:00:00:00:00:00:00:ff:ff:0a:0b:0b:fb
Receive Request was posted
QP state was change to RTS
completion was found in CQ with status 0x0
[Client only] Message is: '***  SEND operation        ***'
RDMA Read Request was posted
completion was found in CQ with status 0x0
[Client only] Contents of server's buffer: '***  RDMA Read operation   ***'
[Client only] Now replacing it with: '***  RDMA Write operation  ***'
RDMA Write Request was posted
completion was found in CQ with status 0x0

test result is 0
```