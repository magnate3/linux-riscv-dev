#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>

#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

#include "mlx5_ifc.h"

enum {
	MLX5_REG_PAOS            = 0x5006,
	MLX5_REG_HOST_ENDIANNESS = 0x7004,
	MLX5_CMD_OP_ACCESS_REG                    = 0x805,
};

enum mlx5_port_status {
	MLX5_PORT_UP        = 1,
	MLX5_PORT_DOWN      = 2,
};

struct mlx5_ifc_access_register_out_bits {
	u8         status[0x8];
	u8         reserved_at_8[0x18];

	u8         syndrome[0x20];

	u8         reserved_at_40[0x40];

	u8         register_data[][0x20];
};

enum {
	MLX5_ACCESS_REGISTER_IN_OP_MOD_WRITE  = 0x0,
	MLX5_ACCESS_REGISTER_IN_OP_MOD_READ   = 0x1,
};

struct mlx5_ifc_access_register_in_bits {
	u8         opcode[0x10];
	u8         reserved_at_10[0x10];

	u8         reserved_at_20[0x10];
	u8         op_mod[0x10];

	u8         reserved_at_40[0x10];
	u8         register_id[0x10];

	u8         argument[0x20];

	u8         register_data[][0x20];
};

struct mlx5_ifc_paos_reg_bits {
	u8         swid[0x8];
	u8         local_port[0x8];
	u8         reserved_at_10[0x4];
	u8         admin_status[0x4];
	u8         reserved_at_18[0x4];
	u8         oper_status[0x4];

	u8         ase[0x1];
	u8         ee[0x1];
	u8         reserved_at_22[0x1c];
	u8         e[0x2];

	u8         reserved_at_40[0x40];
};

/* Management base versions */
#define IB_MGMT_BASE_VERSION                    1
#define OPA_MGMT_BASE_VERSION                   0x80

#define OPA_SM_CLASS_VERSION                    0x80

/* Management classes */
#define IB_MGMT_CLASS_SUBN_LID_ROUTED           0x01
#define IB_MGMT_CLASS_SUBN_DIRECTED_ROUTE       0x81
#define IB_MGMT_CLASS_SUBN_ADM                  0x03
#define IB_MGMT_CLASS_PERF_MGMT                 0x04
#define IB_MGMT_CLASS_BM                        0x05
#define IB_MGMT_CLASS_DEVICE_MGMT               0x06
#define IB_MGMT_CLASS_CM                        0x07
#define IB_MGMT_CLASS_SNMP                      0x08
#define IB_MGMT_CLASS_DEVICE_ADM                0x10
#define IB_MGMT_CLASS_BOOT_MGMT                 0x11
#define IB_MGMT_CLASS_BIS                       0x12
#define IB_MGMT_CLASS_CONG_MGMT                 0x21
#define IB_MGMT_CLASS_VENDOR_RANGE2_START       0x30
#define IB_MGMT_CLASS_VENDOR_RANGE2_END         0x4F

/* Management methods */
#define IB_MGMT_METHOD_GET                      0x01
#define IB_MGMT_METHOD_SET                      0x02
#define IB_MGMT_METHOD_GET_RESP                 0x81
#define IB_MGMT_METHOD_SEND                     0x03
#define IB_MGMT_METHOD_TRAP                     0x05
#define IB_MGMT_METHOD_REPORT                   0x06
#define IB_MGMT_METHOD_REPORT_RESP              0x86
#define IB_MGMT_METHOD_TRAP_REPRESS             0x07

#define IB_MGMT_METHOD_RESP                     0x80
#define IB_BM_ATTR_MOD_RESP                     cpu_to_be32(1)

#define IB_MGMT_MAX_METHODS                     128


#define IB_SMP_DATA_SIZE                        64
#define IB_SMP_MAX_PATH_HOPS                    64
struct ib_smp {
	u8      base_version;
	u8      mgmt_class;
	u8      class_version;
	u8      method;
	__be16  status;
	u8      hop_ptr;
	u8      hop_cnt;
	__be64  tid;
	__be16  attr_id;
	__be16  resv;
	__be32  attr_mod;
	__be64  mkey;
	__be16  dr_slid;
	__be16  dr_dlid;
	u8      reserved[28];
	u8      data[IB_SMP_DATA_SIZE];
	u8      initial_path[IB_SMP_MAX_PATH_HOPS];
	u8      return_path[IB_SMP_MAX_PATH_HOPS];
}; /* __packed */

enum {
	IB_MGMT_MAD_HDR = 24,
	IB_MGMT_MAD_DATA = 232,
	IB_MGMT_RMPP_HDR = 36,
	IB_MGMT_RMPP_DATA = 220,
	IB_MGMT_VENDOR_HDR = 40,
	IB_MGMT_VENDOR_DATA = 216,
	IB_MGMT_SA_HDR = 56,
	IB_MGMT_SA_DATA = 200,
	IB_MGMT_DEVICE_HDR = 64,
	IB_MGMT_DEVICE_DATA = 192,
	IB_MGMT_MAD_SIZE = IB_MGMT_MAD_HDR + IB_MGMT_MAD_DATA,
	OPA_MGMT_MAD_DATA = 2024,
	OPA_MGMT_RMPP_DATA = 2012,
	OPA_MGMT_MAD_SIZE = IB_MGMT_MAD_HDR + OPA_MGMT_MAD_DATA,
};

struct ib_mad_hdr {
	u8      base_version;
	u8      mgmt_class;
	u8      class_version;
	u8      method;
	__be16  status;
	__be16  class_specific;
	__be64  tid;
	__be16  attr_id;
	__be16  resv;
	__be32  attr_mod;
};

struct ib_mad {
	struct ib_mad_hdr       mad_hdr;
	u8                      data[IB_MGMT_MAD_DATA];
};

static inline void init_query_mad(struct ib_smp *mad)
{
	mad->base_version  = 1;
	mad->mgmt_class    = IB_MGMT_CLASS_SUBN_LID_ROUTED;
	mad->class_version = 1;
	mad->method        = IB_MGMT_METHOD_GET;
}

#define cpu_to_be16	htobe16
#define cpu_to_be32	htobe32

/* Subnet management attributes */
#define IB_SMP_ATTR_NOTICE                      cpu_to_be16(0x0002)
#define IB_SMP_ATTR_NODE_DESC                   cpu_to_be16(0x0010)
#define IB_SMP_ATTR_NODE_INFO                   cpu_to_be16(0x0011)
#define IB_SMP_ATTR_SWITCH_INFO                 cpu_to_be16(0x0012)
#define IB_SMP_ATTR_GUID_INFO                   cpu_to_be16(0x0014)
#define IB_SMP_ATTR_PORT_INFO                   cpu_to_be16(0x0015)
#define IB_SMP_ATTR_PKEY_TABLE                  cpu_to_be16(0x0016)
#define IB_SMP_ATTR_SL_TO_VL_TABLE              cpu_to_be16(0x0017)
#define IB_SMP_ATTR_VL_ARB_TABLE                cpu_to_be16(0x0018)
#define IB_SMP_ATTR_LINEAR_FORWARD_TABLE        cpu_to_be16(0x0019)
#define IB_SMP_ATTR_RANDOM_FORWARD_TABLE        cpu_to_be16(0x001A)
#define IB_SMP_ATTR_MCAST_FORWARD_TABLE         cpu_to_be16(0x001B)
#define IB_SMP_ATTR_SM_INFO                     cpu_to_be16(0x0020)
#define IB_SMP_ATTR_VENDOR_DIAG                 cpu_to_be16(0x0030)
#define IB_SMP_ATTR_LED_INFO                    cpu_to_be16(0x0031)
#define IB_SMP_ATTR_VENDOR_MASK                 cpu_to_be16(0xFF00)


struct mlx5_ifc_mad_ifc_out_bits {
	u8         status[0x8];
	u8         reserved_at_8[0x18];

	u8         syndrome[0x20];

	u8         reserved_at_40[0x40];

	u8         response_mad_packet[256][0x8];
};

struct mlx5_ifc_mad_ifc_in_bits {
	u8         opcode[0x10];
	u8         reserved_at_10[0x10];

	u8         reserved_at_20[0x10];
	u8         op_mod[0x10];

	u8         remote_lid[0x10];
	u8         reserved_at_50[0x8];
	u8         port[0x8];

	u8         reserved_at_60[0x20];

	u8         mad[256][0x8];
};

int pp_query_hca_cap(struct pp_context *ppc);
int pp_config_port(struct ibv_context *ctx, enum mlx5_port_status status);
int pp_query_mad_ifc_port(struct ibv_context *ctx, uint8_t port, struct ibv_port_attr *attr);

#endif
