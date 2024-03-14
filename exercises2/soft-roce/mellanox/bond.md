
# bond
RoCE LAG is a feature meant for mimicking Ethernet bonding for IB devices and is available for dual port cards only.  
多个物理的ConnectX4网卡的网口不能做到同一个bond中，否则会有问题

每个网卡只有一个port  
```
ubuntu:/home/ubuntu:~$ ibv_devinfo
hca_id: mlx5_6
        transport:                      InfiniBand (0)
        fw_ver:                         16.27.1016
        node_guid:                      f23b:68ff:feee:b17e
        sys_image_guid:                 08c0:eb03:00da:e2e6
        vendor_id:                      0x02c9
        vendor_part_id:                 4120
        hw_ver:                         0x0
        board_id:                       MT_0000000011
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_1
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:0040:8f1b
        sys_image_guid:                 08c0:eb03:0040:8f1a
        vendor_id:                      0x02c9
        vendor_part_id:                 4117
        hw_ver:                         0x0
        board_id:                       MT_2420110004
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_3
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:0040:917e
        sys_image_guid:                 08c0:eb03:0040:917e
        vendor_id:                      0x02c9
        vendor_part_id:                 4117
        hw_ver:                         0x0
        board_id:                       MT_2420110004
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_5
        transport:                      InfiniBand (0)
        fw_ver:                         20.30.1004
        node_guid:                      08c0:eb03:00ea:50e6
        sys_image_guid:                 08c0:eb03:00ea:50e6
        vendor_id:                      0x02c9
        vendor_part_id:                 4123
        hw_ver:                         0x0
        board_id:                       MT_0000000226
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             4096 (5)
                        sm_lid:                 4
                        port_lid:               18
                        port_lmc:               0x00
                        link_layer:             InfiniBand

hca_id: mlx5_7
        transport:                      InfiniBand (0)
        fw_ver:                         16.27.1016
        node_guid:                      a202:67ff:fecb:a429
        sys_image_guid:                 08c0:eb03:00da:e2e6
        vendor_id:                      0x02c9
        vendor_part_id:                 4120
        hw_ver:                         0x0
        board_id:                       MT_0000000011
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:0040:8f1a
        sys_image_guid:                 08c0:eb03:0040:8f1a
        vendor_id:                      0x02c9
        vendor_part_id:                 4117
        hw_ver:                         0x0
        board_id:                       MT_2420110004
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_2
        transport:                      InfiniBand (0)
        fw_ver:                         16.27.1016
        node_guid:                      08c0:eb03:00da:e2e6
        sys_image_guid:                 08c0:eb03:00da:e2e6
        vendor_id:                      0x02c9
        vendor_part_id:                 4119
        hw_ver:                         0x0
        board_id:                       MT_0000000011
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet

hca_id: mlx5_4
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:0040:917f
        sys_image_guid:                 08c0:eb03:0040:917e
        vendor_id:                      0x02c9
        vendor_part_id:                 4117
        hw_ver:                         0x0
        board_id:                       MT_2420110004
        phys_port_cnt:                  1
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
```


```
ubuntu:/home/ubuntu:~$ ibdev2netdev 
mlx5_0 port 1 ==> ens4f0 (Down)
mlx5_1 port 1 ==> ens4f1 (Down)
mlx5_2 port 1 ==> ens6 (Up)
mlx5_3 port 1 ==> ens14f0 (Up)
mlx5_4 port 1 ==> ens14f1 (Down)
mlx5_5 port 1 ==> ib0 (Up)
mlx5_6 port 1 ==> ens6f1 (Down)
mlx5_7 port 1 ==> ens6f2 (Down)
root@ubuntu:/home/ubuntu# ibdev2netdev  -v
0000:31:00.0 mlx5_0 (MT4117 - MT2138K10121) CX4121A - ConnectX-4 LX SFP28 fw 14.27.1016 port 1 (DOWN  ) ==> ens4f0 (Down)
0000:31:00.1 mlx5_1 (MT4117 - MT2138K10121) CX4121A - ConnectX-4 LX SFP28 fw 14.27.1016 port 1 (DOWN  ) ==> ens4f1 (Down)
0000:4b:00.0 mlx5_2 (MT4119 - MCX515A-CCAT) CX515A - ConnectX-5 QSFP28 fw 16.27.1016 port 1 (ACTIVE) ==> ens6 (Up)
0000:98:00.0 mlx5_3 (MT4117 - MT2138K10336) CX4121A - ConnectX-4 LX SFP28 fw 14.27.1016 port 1 (ACTIVE) ==> ens14f0 (Up)
0000:98:00.1 mlx5_4 (MT4117 - MT2138K10336) CX4121A - ConnectX-4 LX SFP28 fw 14.27.1016 port 1 (DOWN  ) ==> ens14f1 (Down)
0000:b1:00.0 mlx5_5 (MT4123 - MCX654105A-HCAT) ConnectX-6 VPI adapter card, HDR IB (200Gb/s) and 200GbE, single-port QSFP56, Socket Direct                                                                                            fw 20.30.1004 port 1 (ACTIVE) ==> ib0 (Up)
0000:4b:00.1 mlx5_6 (MT4120 - NA)  fw 16.27.1016 port 1 (DOWN  ) ==> ens6f1 (Down)
0000:4b:00.2 mlx5_7 (MT4120 - NA)  fw 16.27.1016 port 1 (DOWN  ) ==> ens6f2 (Down)
root@ubuntu:/home/ubuntu# 
ubuntu:/home/ubuntu:~$ 
```

```
root@ubuntu:/home/ubuntu# ibv_devinfo -v

hca_id: mlx5_0
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:0040:8f1a
        sys_image_guid:                 08c0:eb03:0040:8f1a
        vendor_id:                      0x02c9
        vendor_part_id:                 4117
        hw_ver:                         0x0
        board_id:                       MT_2420110004
        phys_port_cnt:                  1
        max_mr_size:                    0xffffffffffffffff
        page_size_cap:                  0xfffffffffffff000
        max_qp:                         262144
        max_qp_wr:                      32768
        device_cap_flags:               0xa5721c36
                                        BAD_PKEY_CNTR
                                        BAD_QKEY_CNTR
                                        AUTO_PATH_MIG
                                        CHANGE_PHY_PORT
                                        PORT_ACTIVE_EVENT
                                        SYS_IMAGE_GUID
                                        RC_RNR_NAK_GEN
                                        MEM_WINDOW
                                        XRC
                                        MEM_MGT_EXTENSIONS
                                        MEM_WINDOW_TYPE_2B
                                        MANAGED_FLOW_STEERING
                                        Unknown flags: 0x84400000
        device_cap_exp_flags:           0x520DF8F000000000
                                        EXP_CROSS_CHANNEL
                                        EXP_MR_ALLOCATE
                                        EXT_ATOMICS
                                        EXT_SEND NOP
                                        EXP_UMR
                                        EXP_ODP
                                        EXP_RX_CSUM_TCP_UDP_PKT
                                        EXP_RX_CSUM_IP_PKT
                                        EXP_MASKED_ATOMICS
                                        EXP_RX_TCP_UDP_PKT_TYPE
                                        EXP_SCATTER_FCS
                                        EXP_WQ_DELAY_DROP
                                        EXP_PHYSICAL_RANGE_MR
                                        EXP_UMR_FIXED_SIZE
                                        Unknown flags: 0x200000000000
        max_sge:                        30
        max_sge_rd:                     30
        max_cq:                         16777216
        max_cqe:                        4194303
        max_mr:                         16777216
        max_pd:                         16777216
        max_qp_rd_atom:                 16
        max_ee_rd_atom:                 0
        max_res_rd_atom:                4194304
        max_qp_init_rd_atom:            16
        max_ee_init_rd_atom:            0
        atomic_cap:                     ATOMIC_HCA (1)
        log atomic arg sizes (mask)             0x8
        masked_log_atomic_arg_sizes (mask)      0x3c
        masked_log_atomic_arg_sizes_network_endianness (mask)   0x34
        max fetch and add bit boundary  64
        log max atomic inline           5
        max_ee:                         0
        max_rdd:                        0
        max_mw:                         16777216
        max_raw_ipv6_qp:                0
        max_raw_ethy_qp:                0
        max_mcast_grp:                  2097152
        max_mcast_qp_attach:            240
        max_total_mcast_qp_attach:      503316480
        max_ah:                         2147483647
        max_fmr:                        0
        max_srq:                        8388608
        max_srq_wr:                     32767
        max_srq_sge:                    31
        max_pkeys:                      128
        local_ca_ack_delay:             16
        hca_core_clock:                 156250
        max_klm_list_size:              65536
        max_send_wqe_inline_klms:       20
        max_umr_recursion_depth:        4
        max_umr_stride_dimension:       1
        general_odp_caps:
                                        ODP_SUPPORT
                                        ODP_SUPPORT_IMPLICIT
        max_size:                       0xFFFFFFFFFFFFFFFF
        rc_odp_caps:
                                        SUPPORT_SEND
                                        SUPPORT_RECV
                                        SUPPORT_WRITE
                                        SUPPORT_READ
                                        SUPPORT_SRQ_RECV
        uc_odp_caps:
                                        NO SUPPORT
        ud_odp_caps:
                                        SUPPORT_SEND
        dc_odp_caps:
                                        SUPPORT_SEND
                                        SUPPORT_WRITE
                                        SUPPORT_READ
                                        SUPPORT_SRQ_RECV
        xrc_odp_caps:
                                        NO SUPPORT
        raw_eth_odp_caps:
                                        NO SUPPORT
        max_dct:                        0
        max_device_ctx:                 1020
        Multi-Packet RQ supported
                Supported for objects type:
                        IBV_EXP_MP_RQ_SUP_TYPE_WQ_RQ
                Supported payload shifts:
                        2 bytes
                Log number of strides for single WQE: 9 - 16
                Log number of bytes in single stride: 6 - 13

        VLAN offloads caps:
                                        C-VLAN stripping offload
                                        C-VLAN insertion offload
        rx_pad_end_addr_align:  64
        tso_caps:
        max_tso:                        262144
        supported_qp:
                                        SUPPORT_RAW_PACKET
        packet_pacing_caps:
        qp_rate_limit_min:              0kbps
        qp_rate_limit_max:              0kbps
        ooo_caps:
        ooo_rc_caps  = 0x0
        ooo_xrc_caps = 0x0
        ooo_dc_caps  = 0x0
        ooo_ud_caps  = 0x0
        sw_parsing_caps:
                                        SW_PARSING
                                        SW_PARSING_CSUM
                                        SW_PARSING_LSO
        supported_qp:
                                        SUPPORT_RAW_PACKET
        tag matching not supported
        tunnel_offloads_caps:
                                        TUNNEL_OFFLOADS_VXLAN
                                        TUNNEL_OFFLOADS_GRE
                                        TUNNEL_OFFLOADS_GENEVE
        UMR fixed size:
                max entity size:        2147483648
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
                        max_msg_sz:             0x40000000
                        port_cap_flags:         0x04010000
                        max_vl_num:             invalid value (0)
                        bad_pkey_cntr:          0x0
                        qkey_viol_cntr:         0x0
                        sm_sl:                  0
                        pkey_tbl_len:           1
                        gid_tbl_len:            256
                        subnet_timeout:         0
                        init_type_reply:        0
                        active_width:           4X (2)
                        active_speed:           10.0 Gbps (4)
                        phys_state:             DISABLED (3)
                        GID[  0]:               fe80:0000:0000:0000:0ac0:ebff:fe40:8f1a
                        GID[  1]:               fe80:0000:0000:0000:0ac0:ebff:fe40:8f1a

hca_id: mlx5_2
        transport:                      InfiniBand (0)
        fw_ver:                         16.27.1016
        node_guid:                      08c0:eb03:00da:e2e6
        sys_image_guid:                 08c0:eb03:00da:e2e6
        vendor_id:                      0x02c9
        vendor_part_id:                 4119
        hw_ver:                         0x0
        board_id:                       MT_0000000011
        phys_port_cnt:                  1
        max_mr_size:                    0xffffffffffffffff
        page_size_cap:                  0xfffffffffffff000
        max_qp:                         262144
        max_qp_wr:                      32768
        device_cap_flags:               0xe5721c36
                                        BAD_PKEY_CNTR
                                        BAD_QKEY_CNTR
                                        AUTO_PATH_MIG
                                        CHANGE_PHY_PORT
                                        PORT_ACTIVE_EVENT
                                        SYS_IMAGE_GUID
                                        RC_RNR_NAK_GEN
                                        MEM_WINDOW
                                        XRC
                                        MEM_MGT_EXTENSIONS
                                        MEM_WINDOW_TYPE_2B
                                        MANAGED_FLOW_STEERING
                                        Unknown flags: 0xc4400000
        device_cap_exp_flags:           0x520DF8F100000000
                                        EXP_DC_TRANSPORT
                                        EXP_CROSS_CHANNEL
                                        EXP_MR_ALLOCATE
                                        EXT_ATOMICS
                                        EXT_SEND NOP
                                        EXP_UMR
                                        EXP_ODP
                                        EXP_RX_CSUM_TCP_UDP_PKT
                                        EXP_RX_CSUM_IP_PKT
                                        EXP_MASKED_ATOMICS
                                        EXP_RX_TCP_UDP_PKT_TYPE
                                        EXP_SCATTER_FCS
                                        EXP_WQ_DELAY_DROP
                                        EXP_PHYSICAL_RANGE_MR
                                        EXP_UMR_FIXED_SIZE
                                        Unknown flags: 0x200000000000
        max_sge:                        30
        max_sge_rd:                     30
        max_cq:                         16777216
        max_cqe:                        4194303
        max_mr:                         16777216
        max_pd:                         16777216
        max_qp_rd_atom:                 16
        max_ee_rd_atom:                 0
        max_res_rd_atom:                4194304
        max_qp_init_rd_atom:            16
        max_ee_init_rd_atom:            0
        atomic_cap:                     ATOMIC_HCA (1)
        log atomic arg sizes (mask)             0x8
        masked_log_atomic_arg_sizes (mask)      0x3c
        masked_log_atomic_arg_sizes_network_endianness (mask)   0x34
        max fetch and add bit boundary  64
        log max atomic inline           5
        max_ee:                         0
        max_rdd:                        0
        max_mw:                         16777216
        max_raw_ipv6_qp:                0
        max_raw_ethy_qp:                0
        max_mcast_grp:                  2097152
        max_mcast_qp_attach:            240
        max_total_mcast_qp_attach:      503316480
        max_ah:                         2147483647
        max_fmr:                        0
        max_srq:                        8388608
        max_srq_wr:                     32767
        max_srq_sge:                    31
        max_pkeys:                      128
        local_ca_ack_delay:             16
        hca_core_clock:                 78125
        max_klm_list_size:              65536
        max_send_wqe_inline_klms:       20
        max_umr_recursion_depth:        4
        max_umr_stride_dimension:       1
        general_odp_caps:
                                        ODP_SUPPORT
                                        ODP_SUPPORT_IMPLICIT
        max_size:                       0xFFFFFFFFFFFFFFFF
        rc_odp_caps:
                                        SUPPORT_SEND
                                        SUPPORT_RECV
                                        SUPPORT_WRITE
                                        SUPPORT_READ
                                        SUPPORT_SRQ_RECV
        uc_odp_caps:
                                        NO SUPPORT
        ud_odp_caps:
                                        SUPPORT_SEND
        dc_odp_caps:
                                        SUPPORT_SEND
                                        SUPPORT_WRITE
                                        SUPPORT_READ
                                        SUPPORT_SRQ_RECV
        xrc_odp_caps:
                                        NO SUPPORT
        raw_eth_odp_caps:
                                        NO SUPPORT
        max_dct:                        262144
        max_device_ctx:                 1020
        Multi-Packet RQ supported
                Supported for objects type:
                        IBV_EXP_MP_RQ_SUP_TYPE_SRQ_TM
                        IBV_EXP_MP_RQ_SUP_TYPE_WQ_RQ
                Supported payload shifts:
                        2 bytes
                Log number of strides for single WQE: 3 - 16
                Log number of bytes in single stride: 6 - 13

        VLAN offloads caps:
                                        C-VLAN stripping offload
                                        C-VLAN insertion offload
        rx_pad_end_addr_align:  64
        tso_caps:
        max_tso:                        262144
        supported_qp:
                                        SUPPORT_RAW_PACKET
        packet_pacing_caps:
        qp_rate_limit_min:              1kbps
        qp_rate_limit_max:              100000000kbps
        supported_qp:
                                        SUPPORT_RAW_PACKET
        support_burst_control:          YES
        ooo_caps:
        ooo_rc_caps  = 0x1
        ooo_xrc_caps = 0x1
        ooo_dc_caps  = 0x1
        ooo_ud_caps  = 0x0
                                        SUPPORT_RC_RW_DATA_PLACEMENT
                                        SUPPORT_XRC_RW_DATA_PLACEMENT
                                        SUPPORT_DC_RW_DATA_PLACEMENT
        sw_parsing_caps:
                                        SW_PARSING
                                        SW_PARSING_CSUM
                                        SW_PARSING_LSO
        supported_qp:
                                        SUPPORT_RAW_PACKET
        tag matching not supported
        tunnel_offloads_caps:
                                        TUNNEL_OFFLOADS_VXLAN
                                        TUNNEL_OFFLOADS_GRE
                                        TUNNEL_OFFLOADS_GENEVE
        UMR fixed size:
                max entity size:        2147483648
        Device ports:
                port:   1
                        state:                  PORT_ACTIVE (4)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
                        max_msg_sz:             0x40000000
                        port_cap_flags:         0x04010000
                        max_vl_num:             invalid value (0)
                        bad_pkey_cntr:          0x0
                        qkey_viol_cntr:         0x0
                        sm_sl:                  0
                        pkey_tbl_len:           1
                        gid_tbl_len:            256
                        subnet_timeout:         0
                        init_type_reply:        0
                        active_width:           4X (2)
                        active_speed:           25.0 Gbps (32)
                        phys_state:             LINK_UP (5)
                        GID[  0]:               fe80:0000:0000:0000:0ac0:ebff:feda:e2e6
                        GID[  1]:               fe80:0000:0000:0000:0ac0:ebff:feda:e2e6
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:ac11:f22f
                        GID[  3]:               0000:0000:0000:0000:0000:ffff:ac11:f22f

hca_id: mlx5_4
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:0040:917f
        sys_image_guid:                 08c0:eb03:0040:917e
        vendor_id:                      0x02c9
        vendor_part_id:                 4117
        hw_ver:                         0x0
        board_id:                       MT_2420110004
        phys_port_cnt:                  1
        max_mr_size:                    0xffffffffffffffff
        page_size_cap:                  0xfffffffffffff000
        max_qp:                         262144
        max_qp_wr:                      32768
        device_cap_flags:               0xa5721c36
                                        BAD_PKEY_CNTR
                                        BAD_QKEY_CNTR
                                        AUTO_PATH_MIG
                                        CHANGE_PHY_PORT
                                        PORT_ACTIVE_EVENT
                                        SYS_IMAGE_GUID
                                        RC_RNR_NAK_GEN
                                        MEM_WINDOW
                                        XRC
                                        MEM_MGT_EXTENSIONS
                                        MEM_WINDOW_TYPE_2B
                                        MANAGED_FLOW_STEERING
                                        Unknown flags: 0x84400000
        device_cap_exp_flags:           0x520DF8F000000000
                                        EXP_CROSS_CHANNEL
                                        EXP_MR_ALLOCATE
                                        EXT_ATOMICS
                                        EXT_SEND NOP
                                        EXP_UMR
                                        EXP_ODP
                                        EXP_RX_CSUM_TCP_UDP_PKT
                                        EXP_RX_CSUM_IP_PKT
                                        EXP_MASKED_ATOMICS
                                        EXP_RX_TCP_UDP_PKT_TYPE
                                        EXP_SCATTER_FCS
                                        EXP_WQ_DELAY_DROP
                                        EXP_PHYSICAL_RANGE_MR
                                        EXP_UMR_FIXED_SIZE
                                        Unknown flags: 0x200000000000
        max_sge:                        30
        max_sge_rd:                     30
        max_cq:                         16777216
        max_cqe:                        4194303
        max_mr:                         16777216
        max_pd:                         16777216
        max_qp_rd_atom:                 16
        max_ee_rd_atom:                 0
        max_res_rd_atom:                4194304
        max_qp_init_rd_atom:            16
        max_ee_init_rd_atom:            0
        atomic_cap:                     ATOMIC_HCA (1)
        log atomic arg sizes (mask)             0x8
        masked_log_atomic_arg_sizes (mask)      0x3c
        masked_log_atomic_arg_sizes_network_endianness (mask)   0x34
        max fetch and add bit boundary  64
        log max atomic inline           5
        max_ee:                         0
        max_rdd:                        0
        max_mw:                         16777216
        max_raw_ipv6_qp:                0
        max_raw_ethy_qp:                0
        max_mcast_grp:                  2097152
        max_mcast_qp_attach:            240
        max_total_mcast_qp_attach:      503316480
        max_ah:                         2147483647
        max_fmr:                        0
        max_srq:                        8388608
        max_srq_wr:                     32767
        max_srq_sge:                    31
        max_pkeys:                      128
        local_ca_ack_delay:             16
        hca_core_clock:                 156250
        max_klm_list_size:              65536
        max_send_wqe_inline_klms:       20
        max_umr_recursion_depth:        4
        max_umr_stride_dimension:       1
        general_odp_caps:
                                        ODP_SUPPORT
                                        ODP_SUPPORT_IMPLICIT
        max_size:                       0xFFFFFFFFFFFFFFFF
        rc_odp_caps:
                                        SUPPORT_SEND
                                        SUPPORT_RECV
                                        SUPPORT_WRITE
                                        SUPPORT_READ
                                        SUPPORT_SRQ_RECV
        uc_odp_caps:
                                        NO SUPPORT
        ud_odp_caps:
                                        SUPPORT_SEND
        dc_odp_caps:
                                        SUPPORT_SEND
                                        SUPPORT_WRITE
                                        SUPPORT_READ
                                        SUPPORT_SRQ_RECV
        xrc_odp_caps:
                                        NO SUPPORT
        raw_eth_odp_caps:
                                        NO SUPPORT
        max_dct:                        0
        max_device_ctx:                 1020
        Multi-Packet RQ supported
                Supported for objects type:
                        IBV_EXP_MP_RQ_SUP_TYPE_WQ_RQ
                Supported payload shifts:
                        2 bytes
                Log number of strides for single WQE: 9 - 16
                Log number of bytes in single stride: 6 - 13

        VLAN offloads caps:
                                        C-VLAN stripping offload
                                        C-VLAN insertion offload
        rx_pad_end_addr_align:  64
        tso_caps:
        max_tso:                        262144
        supported_qp:
                                        SUPPORT_RAW_PACKET
        packet_pacing_caps:
        qp_rate_limit_min:              0kbps
        qp_rate_limit_max:              0kbps
        ooo_caps:
        ooo_rc_caps  = 0x0
        ooo_xrc_caps = 0x0
        ooo_dc_caps  = 0x0
        ooo_ud_caps  = 0x0
        sw_parsing_caps:
                                        SW_PARSING
                                        SW_PARSING_CSUM
                                        SW_PARSING_LSO
        supported_qp:
                                        SUPPORT_RAW_PACKET
        tag matching not supported
        tunnel_offloads_caps:
                                        TUNNEL_OFFLOADS_VXLAN
                                        TUNNEL_OFFLOADS_GRE
                                        TUNNEL_OFFLOADS_GENEVE
        UMR fixed size:
                max entity size:        2147483648
        Device ports:
                port:   1
                        state:                  PORT_DOWN (1)
                        max_mtu:                4096 (5)
                        active_mtu:             1024 (3)
                        sm_lid:                 0
                        port_lid:               0
                        port_lmc:               0x00
                        link_layer:             Ethernet
                        max_msg_sz:             0x40000000
                        port_cap_flags:         0x04010000
                        max_vl_num:             invalid value (0)
                        bad_pkey_cntr:          0x0
                        qkey_viol_cntr:         0x0
                        sm_sl:                  0
                        pkey_tbl_len:           1
                        gid_tbl_len:            256
                        subnet_timeout:         0
                        init_type_reply:        0
                        active_width:           4X (2)
                        active_speed:           10.0 Gbps (4)
                        phys_state:             DISABLED (3)
                        GID[  0]:               fe80:0000:0000:0000:0ac0:ebff:fe40:917f
                        GID[  1]:               fe80:0000:0000:0000:0ac0:ebff:fe40:917f
```

```
root@ubuntu:/home/ubuntu# ibswitches 
Switch  : 0x08c0eb0300add04e ports 41 "Quantum Mellanox Technologies" base port 0 lid 10 lmc 0
Switch  : 0x08c0eb0300f6fc26 ports 41 "MF0;d32-03-1:MQM8700/U1" enhanced port 0 lid 4 lmc 0
root@ubuntu:/home/ubuntu# ibhosts
Ca      : 0x08c0eb0300add056 ports 1 "Mellanox Technologies Aggregation Node"
Ca      : 0xe8ebd303003a3cb8 ports 1 "node1 HCA-2"
Ca      : 0xe8ebd303003a3b68 ports 1 "node2 HCA-2"
Ca      : 0xe8ebd303003a3b60 ports 1 "node4 HCA-2"
Ca      : 0xe8ebd303003a3a60 ports 1 "node3 HCA-2"
Ca      : 0x08c0eb0300f6fc2e ports 1 "Mellanox Technologies Aggregation Node"
Ca      : 0xe8ebd303003a3cb9 ports 1 "node1 HCA-3"
Ca      : 0xe8ebd303003a3b69 ports 1 "node2 HCA-3"
Ca      : 0xe8ebd303003a3a61 ports 1 "node3 HCA-3"
Ca      : 0xb8cef603008bc144 ports 1 "mlu1000-2 HCA-1"
Ca      : 0xb8cef603008bcd48 ports 1 "mlu1000-1 HCA-1"
Ca      : 0x08c0eb0300cb3d52 ports 1 "dpu7 HCA-1"
Ca      : 0x08c0eb030050fc4a ports 1 "dpu6 HCA-1"
Ca      : 0x08c0eb030050f1c2 ports 1 "dpu5 HCA-1"
Ca      : 0x08c0eb0300b6a0b4 ports 1 "CPUGL12 HCA-6"
Ca      : 0x08c0eb0300ea512e ports 1 "cpugl11 HCA-6"
Ca      : 0x08c0eb0300b6a0ec ports 1 "CPUGL09 HCA-6"
Ca      : 0xb8cef6030087756e ports 1 "CPUGL10 HCA-6"
Ca      : 0x08c0eb0300ea50ee ports 1 "CPUGL06 HCA-6"
Ca      : 0x08c0eb0300b6a0b8 ports 1 "CPUGL05 HCA-6"
Ca      : 0x08c0eb0300ea50f6 ports 1 "CPUGL04 HCA-6"
Ca      : 0x08c0eb0300ea510e ports 1 "CPUGL08 HCA-6"
Ca      : 0x08c0eb0300ea512a ports 1 "CPUGL03 HCA-6"
Ca      : 0x08c0eb030050fc36 ports 1 "dpu4 HCA-1"
Ca      : 0x08c0eb0300cb36de ports 1 "dpu3 HCA-1"
Ca      : 0x08c0eb0300cb36e2 ports 1 "dpu2 HCA-1"
Ca      : 0x08c0eb030050f81a ports 1 "dpu1 HCA-1"
Ca      : 0x08c0eb0300b6a0c4 ports 1 "CPUGL02 HCA-6"
Ca      : 0x08c0eb0300b6a0cc ports 1 "cpugl1 HCA-6"
Ca      : 0x08c0eb0300cb36ea ports 1 "fpga2 HCA-1"
Ca      : 0x08c0eb0300ea50e6 ports 1 "CPUGL07 HCA-6"
root@ubuntu:/home/ubuntu# 
```

```
root@ubuntu:/home/ubuntu# ibswitches 
Switch  : 0x08c0eb0300add04e ports 41 "Quantum Mellanox Technologies" base port 0 lid 10 lmc 0
Switch  : 0x08c0eb0300f6fc26 ports 41 "MF0;d32-03-1:MQM8700/U1" enhanced port 0 lid 4 lmc 0
root@ubuntu:/home/ubuntu# 
root@ubuntu:/home/ubuntu# iblinkinfo -S 0x08c0eb0300add04e
Switch: 0x08c0eb0300add04e Quantum Mellanox Technologies:
          10    1[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    2[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    3[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    4[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    5[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    6[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    7[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    8[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10    9[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   10[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   11[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   12[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   13[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   14[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   15[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   16[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   17[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   18[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   19[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   20[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   21[  ] ==( 4X      25.78125 Gbps Active/  LinkUp)==>      40    1[  ] "node3 HCA-2" ( Could be 53.125 Gbps)
          10   22[  ] ==( 4X      25.78125 Gbps Active/  LinkUp)==>      35    1[  ] "node4 HCA-2" ( Could be 53.125 Gbps)
          10   23[  ] ==( 4X      25.78125 Gbps Active/  LinkUp)==>      41    1[  ] "node2 HCA-2" ( Could be 53.125 Gbps)
          10   24[  ] ==( 4X      25.78125 Gbps Active/  LinkUp)==>      37    1[  ] "node1 HCA-2" ( Could be 53.125 Gbps)
          10   25[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   26[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   27[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   28[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   29[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   30[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   31[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   32[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   33[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   34[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   35[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   36[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   37[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   38[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   39[  ] ==(                Down/ Polling)==>             [  ] "" ( )
          10   40[  ] ==( 4X        53.125 Gbps Active/  LinkUp)==>       4   40[  ] "MF0;d32-03-1:MQM8700/U1" ( )
          10   41[  ] ==( 4X        53.125 Gbps Active/  LinkUp)==>      23    1[  ] "Mellanox Technologies Aggregation Node" ( )
root@ubuntu:/home/ubuntu# 
```