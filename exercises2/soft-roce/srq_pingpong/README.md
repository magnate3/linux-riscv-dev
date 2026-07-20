
# server

```Shell
ibdev2netdev 
mlx5_0 port 1 ==> ens4f0 (Down)
mlx5_1 port 1 ==> ens4f1 (Up)
mlx5_2 port 1 ==> ens6 (Up)
mlx5_3 port 1 ==> ens14f0 (Up)
mlx5_4 port 1 ==> ens14f1 (Down)
mlx5_5 port 1 ==> ib0 (Up)
```

```Shell
 rdma link
0/1: mlx5_0/1: state DOWN physical_state DISABLED 
2/1: mlx5_2/1: state ACTIVE physical_state LINK_UP 
4/1: mlx5_4/1: state DOWN physical_state DISABLED 
5/1: mlx5_5/1: subnet_prefix fe80:0000:0000:0000 lid 12 sm_lid 4 lmc 0 state ACTIVE physical_state LINK_UP 
7/1: mlx5_3/1: state ACTIVE physical_state LINK_UP 
16/1: mlx5_1/1: state ACTIVE physical_state LINK_UP 
```

```Shell
ibv_devinfo -d mlx5_3 -v
hca_id: mlx5_3
        transport:                      InfiniBand (0)
        fw_ver:                         14.27.1016
        node_guid:                      08c0:eb03:003b:353c
        sys_image_guid:                 08c0:eb03:003b:353c
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
                        active_width:           1X (1)
                        active_speed:           10.0 Gbps (4)
                        phys_state:             LINK_UP (5)
                        GID[  0]:               fe80:0000:0000:0000:0ac0:ebff:fe3b:353c
                        GID[  1]:               fe80:0000:0000:0000:0ac0:ebff:fe3b:353c
                        GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:f238
                        GID[  3]:               0000:0000:0000:0000:0000:ffff:c0a8:f238

```
 GID[  2]:               0000:0000:0000:0000:0000:ffff:c0a8:f238 是ip地址   
```Shell
./srq_pingpong  -d mlx5_3  -g 2
  local address:  LID 0x0000, QPN 0x001241, PSN 0x5bf358, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001242, PSN 0xaf1662, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001243, PSN 0x721504, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001244, PSN 0x4b2a03, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001245, PSN 0xf54b5a, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001246, PSN 0xb446b4, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001247, PSN 0x219403, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001248, PSN 0xcff340, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001249, PSN 0xe4ebe4, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x00124a, PSN 0x5dc596, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x00124b, PSN 0xcb1a82, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x00124c, PSN 0x5dd5ed, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x00124d, PSN 0xabf1d5, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x00124e, PSN 0x233f5e, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x00124f, PSN 0x17f3c4, GID ::ffff:192.168.242.56
  local address:  LID 0x0000, QPN 0x001250, PSN 0x2eb238, GID ::ffff:192.168.242.56
  remote address:  LID 0x0000, QPN 0x000239, PSN 0x77489a,   local address:  LID 0x0000, QPN 0x001241, PSN 0x5bf358, 
  remote address:  LID 0x0000, QPN 0x00023a, PSN 0x3a22b3,   local address:  LID 0x0000, QPN 0x001242, PSN 0xaf1662, 
  remote address:  LID 0x0000, QPN 0x00023b, PSN 0xeca64b,   local address:  LID 0x0000, QPN 0x001243, PSN 0x721504, 
  remote address:  LID 0x0000, QPN 0x00023c, PSN 0xfe5c07,   local address:  LID 0x0000, QPN 0x001244, PSN 0x4b2a03, 
  remote address:  LID 0x0000, QPN 0x00023d, PSN 0x57a245,   local address:  LID 0x0000, QPN 0x001245, PSN 0xf54b5a, 
  remote address:  LID 0x0000, QPN 0x00023e, PSN 0xd7f78d,   local address:  LID 0x0000, QPN 0x001246, PSN 0xb446b4, 
  remote address:  LID 0x0000, QPN 0x00023f, PSN 0xcc6332,   local address:  LID 0x0000, QPN 0x001247, PSN 0x219403, 
  remote address:  LID 0x0000, QPN 0x000240, PSN 0x8bf10d,   local address:  LID 0x0000, QPN 0x001248, PSN 0xcff340, 
  remote address:  LID 0x0000, QPN 0x000241, PSN 0x7bb6f7,   local address:  LID 0x0000, QPN 0x001249, PSN 0xe4ebe4, 
  remote address:  LID 0x0000, QPN 0x000242, PSN 0x12db76,   local address:  LID 0x0000, QPN 0x00124a, PSN 0x5dc596, 
  remote address:  LID 0x0000, QPN 0x000243, PSN 0x831e18,   local address:  LID 0x0000, QPN 0x00124b, PSN 0xcb1a82, 
  remote address:  LID 0x0000, QPN 0x000244, PSN 0xae9402,   local address:  LID 0x0000, QPN 0x00124c, PSN 0x5dd5ed, 
  remote address:  LID 0x0000, QPN 0x000245, PSN 0xb94b90,   local address:  LID 0x0000, QPN 0x00124d, PSN 0xabf1d5, 
  remote address:  LID 0x0000, QPN 0x000246, PSN 0x5e02c6,   local address:  LID 0x0000, QPN 0x00124e, PSN 0x233f5e, 
  remote address:  LID 0x0000, QPN 0x000247, PSN 0x030a42,   local address:  LID 0x0000, QPN 0x00124f, PSN 0x17f3c4, 
  remote address:  LID 0x0000, QPN 0x000248, PSN 0x1aed14,   local address:  LID 0x0000, QPN 0x001250, PSN 0x2eb238, 
  remote address: LID 0x0000, QPN 0x000239, PSN 0x77489a, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x00023a, PSN 0x3a22b3, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x00023b, PSN 0xeca64b, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x00023c, PSN 0xfe5c07, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x00023d, PSN 0x57a245, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x00023e, PSN 0xd7f78d, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x00023f, PSN 0xcc6332, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000240, PSN 0x8bf10d, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000241, PSN 0x7bb6f7, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000242, PSN 0x12db76, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000243, PSN 0x831e18, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000244, PSN 0xae9402, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000245, PSN 0xb94b90, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000246, PSN 0x5e02c6, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000247, PSN 0x030a42, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x000248, PSN 0x1aed14, GID ::ffff:192.168.242.57
8192000 bytes in 0.01 seconds = 5757.86 Mbit/sec
1000 iters in 0.01 seconds = 11.38 usec/iter
```

# client

```Shell
./srq_pingpong  -d mlx5_3  192.168.242.56 -g 2
  local address:  LID 0x0000, QPN 0x000239, PSN 0x77489a, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x00023a, PSN 0x3a22b3, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x00023b, PSN 0xeca64b, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x00023c, PSN 0xfe5c07, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x00023d, PSN 0x57a245, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x00023e, PSN 0xd7f78d, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x00023f, PSN 0xcc6332, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000240, PSN 0x8bf10d, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000241, PSN 0x7bb6f7, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000242, PSN 0x12db76, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000243, PSN 0x831e18, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000244, PSN 0xae9402, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000245, PSN 0xb94b90, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000246, PSN 0x5e02c6, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000247, PSN 0x030a42, GID ::ffff:192.168.242.57
  local address:  LID 0x0000, QPN 0x000248, PSN 0x1aed14, GID ::ffff:192.168.242.57
  remote address: LID 0x0000, QPN 0x001241, PSN 0x5bf358, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001242, PSN 0xaf1662, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001243, PSN 0x721504, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001244, PSN 0x4b2a03, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001245, PSN 0xf54b5a, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001246, PSN 0xb446b4, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001247, PSN 0x219403, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001248, PSN 0xcff340, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001249, PSN 0xe4ebe4, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x00124a, PSN 0x5dc596, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x00124b, PSN 0xcb1a82, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x00124c, PSN 0x5dd5ed, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x00124d, PSN 0xabf1d5, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x00124e, PSN 0x233f5e, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x00124f, PSN 0x17f3c4, GID ::ffff:192.168.242.56
  remote address: LID 0x0000, QPN 0x001250, PSN 0x2eb238, GID ::ffff:192.168.242.56
8192000 bytes in 0.01 seconds = 9088.34 Mbit/sec
1000 iters in 0.01 seconds = 7.21 usec/iter
```

# spdk

## rdma_create_qp
```
spdk_rdma_qp_create(struct rdma_cm_id *cm_id, struct spdk_rdma_qp_init_attr *qp_attr)
{
	struct spdk_rdma_qp *spdk_rdma_qp;
	int rc;
	struct ibv_qp_init_attr attr = {
		.qp_context = qp_attr->qp_context,
		.send_cq = qp_attr->send_cq,
		.recv_cq = qp_attr->recv_cq,
		.srq = qp_attr->srq,
		.cap = qp_attr->cap,
		.qp_type = IBV_QPT_RC
	};

	spdk_rdma_qp = calloc(1, sizeof(*spdk_rdma_qp));
	if (!spdk_rdma_qp) {
		SPDK_ERRLOG("qp memory allocation failed\n");
		return NULL;
	}

	if (qp_attr->stats) {
		spdk_rdma_qp->stats = qp_attr->stats;
		spdk_rdma_qp->shared_stats = true;
	} else {
		spdk_rdma_qp->stats = calloc(1, sizeof(*spdk_rdma_qp->stats));
		if (!spdk_rdma_qp->stats) {
			SPDK_ERRLOG("qp statistics memory allocation failed\n");
			free(spdk_rdma_qp);
			return NULL;
		}
	}

	rc = rdma_create_qp(cm_id, qp_attr->pd, &attr);
	if (rc) {
		SPDK_ERRLOG("Failed to create qp, errno %s (%d)\n", spdk_strerror(errno), errno);
		free(spdk_rdma_qp);
		return NULL;
	}

	qp_attr->cap = attr.cap;
	spdk_rdma_qp->qp = cm_id->qp;
	spdk_rdma_qp->cm_id = cm_id;

	return spdk_rdma_qp;
}

```

## ibv_create_srq

```C
struct spdk_rdma_srq *
spdk_rdma_srq_create(struct spdk_rdma_srq_init_attr *init_attr)
{
	assert(init_attr);
	assert(init_attr->pd);

	struct spdk_rdma_srq *rdma_srq = calloc(1, sizeof(*rdma_srq));

	if (!rdma_srq) {
		SPDK_ERRLOG("Can't allocate memory for SRQ handle\n");
		return NULL;
	}

	if (init_attr->stats) {
		rdma_srq->stats = init_attr->stats;
		rdma_srq->shared_stats = true;
	} else {
		rdma_srq->stats = calloc(1, sizeof(*rdma_srq->stats));
		if (!rdma_srq->stats) {
			SPDK_ERRLOG("SRQ statistics memory allocation failed");
			free(rdma_srq);
			return NULL;
		}
	}

	rdma_srq->srq = ibv_create_srq(init_attr->pd, &init_attr->srq_init_attr);
	if (!rdma_srq->srq) {
		if (!init_attr->stats) {
			free(rdma_srq->stats);
		}
		SPDK_ERRLOG("Unable to create SRQ, errno %d (%s)\n", errno, spdk_strerror(errno));
		free(rdma_srq);
		return NULL;
	}

	return rdma_srq;
}

```

# 根据qp_num 获取qp

nvme_rdma_process_recv_completion -->  get_rdma_qpair_from_wc
```
static inline int
nvme_rdma_process_recv_completion(struct nvme_rdma_poller *poller, struct ibv_wc *wc,
				  struct nvme_rdma_wr *rdma_wr)
{
	struct nvme_rdma_qpair		*rqpair;
	struct spdk_nvme_rdma_req	*rdma_req;
	struct spdk_nvme_rdma_rsp	*rdma_rsp;

	rdma_rsp = SPDK_CONTAINEROF(rdma_wr, struct spdk_nvme_rdma_rsp, rdma_wr);

	if (poller && poller->srq) {
		rqpair = get_rdma_qpair_from_wc(poller->group, wc);
		if (spdk_unlikely(!rqpair)) {
			/* Since we do not handle the LAST_WQE_REACHED event, we do not know when
			 * a Receive Queue in a QP, that is associated with an SRQ, is flushed.
			 * We may get a WC for a already destroyed QP.
			 *
			 * However, for the SRQ, this is not any error. Hence, just re-post the
			 * receive request to the SRQ to reuse for other QPs, and return 0.
			 */
			spdk_rdma_srq_queue_recv_wrs(poller->srq, rdma_rsp->recv_wr);
			return 0;
		}
	} 
```

```C
static struct spdk_nvmf_rdma_qpair *
get_rdma_qpair_from_wc(struct spdk_nvmf_rdma_poller *rpoller, struct ibv_wc *wc)
{
	struct spdk_nvmf_rdma_qpair find;

	find.qp_num = wc->qp_num;

	return RB_FIND(qpairs_tree, &rpoller->qpairs, &find);
}
```

```C
#define NVME_RDMA_POLL_GROUP_CHECK_QPN(_rqpair, qpn)				\
	((_rqpair)->rdma_qp && (_rqpair)->rdma_qp->qp->qp_num == (qpn))	\
```


```C
static struct nvme_rdma_qpair *
get_rdma_qpair_from_wc(struct nvme_rdma_poll_group *group, struct ibv_wc *wc)
{
	struct spdk_nvme_qpair *qpair;
	struct nvme_rdma_qpair *rqpair;

	STAILQ_FOREACH(qpair, &group->group.connected_qpairs, poll_group_stailq) {
		rqpair = nvme_rdma_qpair(qpair);
		if (NVME_RDMA_POLL_GROUP_CHECK_QPN(rqpair, wc->qp_num)) {
			return rqpair;
		}
	}

	STAILQ_FOREACH(qpair, &group->group.disconnected_qpairs, poll_group_stailq) {
		rqpair = nvme_rdma_qpair(qpair);
		if (NVME_RDMA_POLL_GROUP_CHECK_QPN(rqpair, wc->qp_num)) {
			return rqpair;
		}
	}

	return NULL;
}
```
