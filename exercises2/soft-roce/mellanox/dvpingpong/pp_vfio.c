#include "pp_common.h"
#include "pp_vfio.h"

static int mlx5_core_access_reg(struct ibv_context *ctx, void *data_in,
				int size_in, void *data_out, int size_out,
				uint32_t reg_id, int arg, int write)
{
	int outlen = DEVX_ST_SZ_BYTES(access_register_out) + size_out;
	int inlen = DEVX_ST_SZ_BYTES(access_register_in) + size_in;
	int err = -ENOMEM;
	uint32_t *out = NULL;
	uint32_t *in = NULL;
	void *data;

	in = calloc(1, inlen);
	out = calloc(1, outlen);
	if (!in || !out)
		goto out;

	data = DEVX_ADDR_OF(access_register_in, in, register_data);
	memcpy(data, data_in, size_in);

	DEVX_SET(access_register_in, in, opcode, MLX5_CMD_OP_ACCESS_REG);
	DEVX_SET(access_register_in, in, op_mod, !write);
	DEVX_SET(access_register_in, in, argument, arg);
	DEVX_SET(access_register_in, in, register_id, reg_id);

	err = mlx5dv_devx_general_cmd(ctx, in, inlen, out, outlen);
	if (err)
		goto out;

	data = DEVX_ADDR_OF(access_register_out, out, register_data);
	memcpy(data_out, data, size_out);

out:
	free(out);
	free(in);
	if (err)
		ERR("access_reg failed, reg_id %d, arg %d, write %d\n", reg_id, arg, write);

	return err;
}

static int mlx5_set_port_admin_status(struct ibv_context *ctx, enum mlx5_port_status status)
{
	uint32_t in[DEVX_ST_SZ_DW(paos_reg)] = {0};
	uint32_t out[DEVX_ST_SZ_DW(paos_reg)];
	int ret;

	/* Query */
	DEVX_SET(paos_reg, in, local_port, 1);
	ret = mlx5_core_access_reg(ctx, in, sizeof(in), out, sizeof(out),
				   MLX5_REG_PAOS, 0, 0);
	if (ret) {
		ERR("Query PAOS failed\n");
		return ret;
	}
	INFO("Query PAOS succeeds, status %d\n", DEVX_GET(paos_reg, out, admin_status));

	/* Test set */
	DEVX_SET(paos_reg, in, local_port, 1);
	DEVX_SET(paos_reg, in, admin_status, status);
	DEVX_SET(paos_reg, in, ase, 1);
	ret = mlx5_core_access_reg(ctx, in, sizeof(in), out, sizeof(out),
				   MLX5_REG_PAOS, 0, 1);
	if (ret) {
		ERR("Set port status to %d failed\n", status);
		return ret;
	}

	/* Query again */
	DEVX_SET(paos_reg, in, local_port, 1);
	ret = mlx5_core_access_reg(ctx, in, sizeof(in), out, sizeof(out),
				   MLX5_REG_PAOS, 0, 0);
	if (ret) {
		ERR("Query(2nd) PAOS failed\n");
		return ret;
	}
	if (DEVX_GET(paos_reg, out, admin_status) != status) {
		ERR("Set failed? Set %d get %d\n", status, DEVX_GET(paos_reg, out, admin_status));
		return -1;
	}

	INFO("Set port status %d succeeds\n", status);
	return 0;
}

int pp_config_port(struct ibv_context *ctx, enum mlx5_port_status status)
{
	mlx5_set_port_admin_status(ctx, status);  // MLX5_PORT_DOWN, MLX5_PORT_UP
	return 0;
}

enum {
	MLX5_IB_VENDOR_CLASS1 = 0x9,
	MLX5_IB_VENDOR_CLASS2 = 0xa,
};

static int mlx5_cmd_mad_ifc(struct ibv_context *ctx, const void *inb, void *outb,
			    u16 opmod, u8 port)
{
	int outlen = DEVX_ST_SZ_BYTES(mad_ifc_out);
	int inlen = DEVX_ST_SZ_BYTES(mad_ifc_in);
	int err = -ENOMEM;
	void *data;
	void *resp;
	u32 *out;
	u32 *in;

	in = calloc(inlen, 1);
	out = calloc(outlen, 1);
	if (!in || !out)
		goto out;

	DEVX_SET(mad_ifc_in, in, opcode, MLX5_CMD_OP_MAD_IFC);
	DEVX_SET(mad_ifc_in, in, op_mod, opmod);
	DEVX_SET(mad_ifc_in, in, port, port);

	data = DEVX_ADDR_OF(mad_ifc_in, in, mad);
	memcpy(data, inb, DEVX_FLD_SZ_BYTES(mad_ifc_in, mad));

	//err = mlx5_cmd_exec_inout(dev, mad_ifc, in, out);
	err = mlx5dv_devx_general_cmd(ctx, in, inlen, out, outlen);
	if (err) {
		ERR("MAD_IFC failed %d\n", err);
		goto out;
	}

	resp = DEVX_ADDR_OF(mad_ifc_out, out, response_mad_packet);
	memcpy(outb, resp,
	       DEVX_FLD_SZ_BYTES(mad_ifc_out, response_mad_packet));

out:
	free(out);
	free(in);
	return err;
}

static bool can_do_mad_ifc(struct ibv_context *ctx,  uint8_t port_num,
			   struct ib_mad *in_mad)
{
	/* FIXME: todo */
	/*
	if (in_mad->mad_hdr.mgmt_class != IB_MGMT_CLASS_SUBN_LID_ROUTED &&
	    in_mad->mad_hdr.mgmt_class != IB_MGMT_CLASS_SUBN_DIRECTED_ROUTE)
		return true;

	return dev->port_caps[port_num - 1].has_smi;
	*/
	return true;
}

static int mlx5_MAD_IFC(struct ibv_context *ctx, int ignore_mkey,
			//int ignore_bkey, uint8_t port, const struct ib_wc *in_wc,
			//const struct ib_grh *in_grh, const void *in_mad,
			int ignore_bkey, uint8_t port, const void *in_wc,
			const void *in_grh, const void *in_mad,
			void *response_mad)
{
	u8 op_modifier = 0;

	if (!can_do_mad_ifc(ctx, port, (struct ib_mad *)in_mad))
		return -EPERM;

	/* Key check traps can't be generated unless we have in_wc to
         * tell us where to send the trap.
         */
	if (ignore_mkey || !in_wc)
		op_modifier |= 0x1;
	if (ignore_bkey || !in_wc)
		op_modifier |= 0x2;

	return mlx5_cmd_mad_ifc(ctx, in_mad, response_mad, op_modifier, port);
}


int pp_query_mad_ifc_port(struct ibv_context *ctx, uint8_t port, struct ibv_port_attr *attr)
{
	struct ib_smp *in_mad, *out_mad;
	int err = ENOMEM;

	in_mad = calloc(sizeof(*in_mad), 1);
	out_mad = calloc(sizeof(*in_mad), 1);
	if (!in_mad || !out_mad)
		goto out;

	init_query_mad(in_mad);
	in_mad->attr_id = IB_SMP_ATTR_PORT_INFO;
	in_mad->attr_mod = cpu_to_be32(port);
	err = mlx5_MAD_IFC(ctx, 1, 1, port, NULL, NULL, in_mad, out_mad);
	if (err) {
		ERR("mlx5_MAD_IFC failed %d\n", err);
		goto out;
	}

	attr->lid = be16toh(*((__be16 *)(out_mad->data + 16)));
	attr->lmc = out_mad->data[34] & 0x7;
	attr->sm_lid = be16toh(*((__be16 *)(out_mad->data + 18)));
	attr->sm_sl = out_mad->data[36] & 0xf;
	attr->state = out_mad->data[32] & 0xf;
	attr->phys_state =  out_mad->data[33] >> 4;
	attr->port_cap_flags = be16toh(*((__be32 *)(out_mad->data + 20)));
	attr->active_width     = out_mad->data[31] & 0xf;
	attr->active_speed     = out_mad->data[35] >> 4;
	attr->max_mtu          = out_mad->data[41] & 0xf;
	attr->active_mtu       = out_mad->data[36] >> 4;
	attr->subnet_timeout   = out_mad->data[51] & 0x1f;
	attr->max_vl_num       = out_mad->data[37] >> 4;
	attr->init_type_reply  = out_mad->data[41] >> 4;

	DBG("state %d phys_state %d lid %d sm_lid %d max_mtu %d active_mtu %d port_cap_flags 0x%x\n",
	    attr->state, attr->phys_state, attr->lid, attr->sm_lid, attr->max_mtu, attr->active_mtu, attr->port_cap_flags);
out:
	free(in_mad);
	free(out_mad);
	return err;
}

int pp_query_hca_cap(struct pp_context *ppc)
{
        uint16_t opmod = MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE |
		HCA_CAP_OPMOD_GET_CUR;
	uint32_t out[DEVX_ST_SZ_DW(query_hca_cap_out)] = {};
	uint32_t in[DEVX_ST_SZ_DW(query_hca_cap_in)] = {};
	int ret;

	DEVX_SET(query_hca_cap_in, in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
	DEVX_SET(query_hca_cap_in, in, op_mod, opmod);

	ret = mlx5dv_devx_general_cmd(ppc->ibctx, in, sizeof(in), out, sizeof(out));
	if (ret) {
		ERR("mlx5dv_devx_general_cmd failed: %d, errno %d\n", ret, errno);
		return ret;
	}

	INFO("Test devx_general_cmd(query_hca_cap): qos %d, log_max_qp_sz %d, log_max_qp %d, log_max_cq_sz %d, log_max_cq %d, port_type %d, ib_virt %d\n",
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.qos),
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.log_max_qp_sz),
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.log_max_qp),
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.log_max_cq_sz),
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.log_max_cq),
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.port_type),
	     DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.ib_virt));

	return 0;
}
