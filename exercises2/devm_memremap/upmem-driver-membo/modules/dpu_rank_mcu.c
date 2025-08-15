/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/kernel.h>
#include <linux/string.h>

#include <dpu_region.h>
#include <dpu_rank_mcu.h>
#include <dpu_vpd_structures.h>
#include <dpu_mcu_ci_commands.h>
#include <dpu_mcu_ci_protocol.h>

#define VPD_HEADER_OFF 0x1E000

int dpu_rank_mcu_set_vdd_dpu(struct dpu_rank_t *rank, uint32_t vdd_mv)
{
	struct ec_params_vdd vdd = { .vdd_mv = vdd_mv };
	const char *pn = rank->part_number;
	int ret;

	dev_dbg(&rank->dev, "%s\n", __func__);

	/* This command does not exist for E19 FW */
	if (!strcmp(pn, "") || !strncmp(pn, "UPMEM-E19", strlen("UPMEM-E19"))) {
		return 0;
	}

	ret = dpu_control_interface_mcu_command(rank, EC_CMD_DIMM_SET_VDD, 0,
						&vdd, sizeof(vdd), NULL, 0);
	if (ret != 0) {
		dev_warn(&rank->dev, "cannot set VDD DPU using MCU\n");
		return ret;
	}

	return 0;
}

int dpu_rank_mcu_get_version(struct dpu_rank_t *rank)
{
	char ec_mcu_version[128];
	int ret;

	dev_dbg(&rank->dev, "%s\n", __func__);

	ret = dpu_control_interface_mcu_command(rank, EC_CMD_GET_BUILD_INFO, 0,
						NULL, 0, ec_mcu_version,
						sizeof(ec_mcu_version));
	if (ret == 0) {
		dev_dbg(&rank->dev, "MCU version: %s\n", ec_mcu_version);

		strncpy(rank->mcu_version, ec_mcu_version,
			sizeof(rank->mcu_version) - 1);
	} else {
		dev_warn(&rank->dev, "cannot request MCU version\n");
		return -EAGAIN;
	}

	return 0;
}

int dpu_rank_mcu_get_frequency_and_clock_div(struct dpu_rank_t *rank)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	struct ec_params_osc p_osc = { .set_fck_mhz = OSC_FREQ_DONT_SET };
	struct ec_response_osc r_osc;
	int ret;

	dev_dbg(&rank->dev, "%s\n", __func__);

	ret = dpu_control_interface_mcu_command(rank, EC_CMD_OSC_FREQ, 0,
						&p_osc, sizeof(p_osc), &r_osc,
						sizeof(r_osc));
	if (ret == 0) {
		dev_dbg(&rank->dev,
			"FCK frequency %u MHz (min/max %u/%u MHz)\n",
			r_osc.fck_mhz, r_osc.fck_min_mhz, r_osc.fck_max_mhz);
		dev_dbg(&rank->dev, "Divider /%u = %u Mhz -> /%u = %u Mhz\n",
			r_osc.div_max, r_osc.fck_mhz / r_osc.div_max,
			r_osc.div_min, r_osc.fck_mhz / r_osc.div_min);

		tr->desc.timings.fck_frequency_in_mhz = r_osc.fck_mhz;
		tr->desc.timings.clock_division = r_osc.div_min;
	} else {
		dev_warn(&rank->dev,
			 "cannot request FCK frequency / Divider from MCU\n");
		return -EAGAIN;
	}

	return 0;
}

int dpu_rank_mcu_set_rank_name(struct dpu_rank_t *rank)
{
	struct ec_params_dimm_id p_dimm;
	struct ec_response_dimm_id r_dimm;
	int ret;

	dev_dbg(&rank->dev, "%s\n", __func__);

	p_dimm.id_index = DIMM_ID_DEV_NAME;
	strncpy(p_dimm.id_string, dev_name(&rank->dev),
		sizeof(p_dimm.id_string) - 1);
	p_dimm.id_string[sizeof(p_dimm.id_string) - 1] = '\0';
	ret = dpu_control_interface_mcu_command(
		rank, EC_CMD_DIMM_SET_ID, 0, &p_dimm, sizeof(p_dimm), NULL, 0);
	if (ret < 0) {
		dev_warn(&rank->dev, "cannot inform MCU about rank name\n");
		return -EAGAIN;
	}

	ret = dpu_control_interface_mcu_command(rank, EC_CMD_DIMM_ID, 0, NULL,
						0, &r_dimm, sizeof(r_dimm));
	if (ret == 0) {
		dev_dbg(&rank->dev, "Module part number: %s S/N: %s\n",
			r_dimm.part_number, r_dimm.serial_number);
		dev_dbg(&rank->dev, "Device name: %s Sticker name: %s\n",
			r_dimm.dev_name, r_dimm.pretty_name);

		strncpy(rank->part_number, r_dimm.part_number,
			sizeof(rank->part_number) - 1);
		strncpy(rank->serial_number, r_dimm.serial_number,
			sizeof(rank->serial_number) - 1);
	} else {
		dev_warn(
			&rank->dev,
			"cannot request part number / serial number from MCU\n");
		return -EAGAIN;
	}

	return 0;
}

int dpu_rank_mcu_get_rank_index_count(struct dpu_rank_t *rank)
{
	struct ec_response_rank_info r_rank;
	int ret;

	dev_dbg(&rank->dev, "%s\n", __func__);

	ret = dpu_control_interface_mcu_command(rank, EC_CMD_RANK_INFO, 0, NULL,
						0, &r_rank, sizeof(r_rank));
	if (ret == 0) {
		dev_dbg(&rank->dev, "Rank %u/%u\n", r_rank.rank_index,
			r_rank.rank_total);

		rank->rank_index = r_rank.rank_index;
		rank->rank_count = r_rank.rank_total;
	} else {
		dev_warn(&rank->dev,
			 "cannot request rank index / rank count from MCU\n");
		return -EAGAIN;
	}

	return 0;
}

static int dpu_rank_mcu_validate_vpd_header(struct dpu_rank_t *rank,
					    struct dpu_vpd_header *vpd_header)
{
	/* Basic checks to make sure the VPD file is valid */
	if (memcmp(&vpd_header->struct_id, VPD_STRUCT_ID,
		   sizeof(vpd_header->struct_id)) != 0) {
		dev_dbg(&rank->dev, "invalid VPD structure ID\n");
		return -EINVAL;
	}

	if (vpd_header->struct_ver != VPD_STRUCT_VERSION) {
		dev_dbg(&rank->dev, "invalid VPD structure version");
		return -EINVAL;
	}

	return 0;
}

int dpu_rank_mcu_get_vpd(struct dpu_rank_t *rank)
{
	int ret;

	dev_dbg(&rank->dev, "%s\n", __func__);

	ret = dpu_control_interface_flash_read(rank, &rank->vpd.vpd_header,
					       VPD_HEADER_OFF,
					       sizeof(struct dpu_vpd_header));
	if (ret == 0) {
		size_t size_entries;
		uint16_t repair_count;

		if (dpu_rank_mcu_validate_vpd_header(
			    rank, &rank->vpd.vpd_header) != 0) {
			dev_warn(&rank->dev, "invalid VPD header");
			/*
			 * The VPD segment might be corrupted. Pursue the rank
			 * initialization so that we might be able to flash it again. The
			 * SDK will prevent to use the rank if ignoreVpd is not specified.
			 */
			return -ENODATA;
		}

		repair_count = rank->vpd.vpd_header.repair_count;
		if (repair_count != (uint16_t)VPD_UNDEFINED_REPAIR_COUNT) {
			size_entries = repair_count *
				       sizeof(struct dpu_vpd_repair_entry);
			ret = dpu_control_interface_flash_read(
				rank, rank->vpd.repair_entries,
				VPD_HEADER_OFF +
					rank->vpd.vpd_header.struct_size,
				size_entries);
			if (ret < 0) {
				dev_warn(&rank->dev,
					 "failed to read repair entries\n");
				return -EAGAIN;
			}
		}
	} else {
		dev_warn(&rank->dev, "cannot request VPD from MCU\n");
		return -EAGAIN;
	}

	return 0;
}

int dpu_rank_mcu_check_ec_compatibility(struct dpu_rank_t *rank)
{
#define DPU_EC_MAJOR_VERSION "v2"
#define DPU_EC_MIN_MINOR 0
	char *p, *ec_major;
	unsigned int ec_minor;

	/*
	 * EC version string format is:
	 * [drdimm|bringup-dimm]_$ECMAJOR.$ECMINOR.#COMMITS-SHA1
	 */

	p = strchr(rank->mcu_version, '_');
	if (!p) {
		dev_warn(
			&rank->dev,
			"EC UPMEM firmware major version can't be extracted from mcu_version = \"%s\"\n",
			rank->mcu_version);
		return -EINVAL;
	}

	ec_major = ++p;
	if (strncmp(ec_major, DPU_EC_MAJOR_VERSION,
		    strlen(DPU_EC_MAJOR_VERSION))) {
		dev_warn(
			&rank->dev,
			"EC UPMEM firmware major version %.*s is not compatible with current driver\n",
			(int)strlen(DPU_EC_MAJOR_VERSION), ec_major);
		return -EINVAL;
	}

	p += strlen(DPU_EC_MAJOR_VERSION) + 1;
	if (sscanf(p, "%u", &ec_minor) != 1) {
		dev_warn(
			&rank->dev,
			"EC UPMEM firmware minor version can't be extracted from mcu_version \"%s\"\n",
			rank->mcu_version);
		return -EINVAL;
	}
	if (ec_minor < DPU_EC_MIN_MINOR) {
		dev_warn(
			&rank->dev,
			"EC UPMEM firmware minor version %u (< %u) is not compatible with current driver\n",
			ec_minor, DPU_EC_MIN_MINOR);
		return -EINVAL;
	}

	return 0;
}

int dpu_rank_mcu_probe(struct dpu_rank_t *rank)
{
	int ret;

	ret = dpu_rank_mcu_get_version(rank);
	if (ret)
		goto end;

	ret = dpu_rank_mcu_check_ec_compatibility(rank);
	if (ret)
		goto end;

	ret = dpu_rank_mcu_get_frequency_and_clock_div(rank);
	if (ret)
		goto end;

	ret = dpu_rank_mcu_set_rank_name(rank);
	if (ret)
		goto end;

	ret = dpu_rank_mcu_get_rank_index_count(rank);
	if (ret)
		goto end;

	ret = dpu_rank_mcu_get_vpd(rank);
	if (ret) {
		if (ret == -ENODATA)
			/* Forget about the error */
			ret = 0;
		goto end;
	}

end:
	return ret;
}
