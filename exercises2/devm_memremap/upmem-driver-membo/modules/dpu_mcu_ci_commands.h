/* SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause */
/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 *
 * Alternatively, this software may be distributed under the terms of the
 * GNU General Public License ("GPL") version 2 as published by the Free
 * Software Foundation.
 */

/* host communication command constants for the DIMM MCU */

#ifndef __DPU_MCU_CI_COMMAMDS_H__
#define __DPU_MCU_CI_COMMANDS_H__

#include <linux/version.h>

/* Common error codes and commands are already defined in the kernel */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 4, 0)
#include <linux/platform_data/cros_ec_commands.h>
#else
#include <linux/mfd/cros_ec_commands.h>
#endif

/* DIMM specific commands */

/* Set the clock oscillator frequency */
#define EC_CMD_OSC_FREQ 0x00C1

/* special value for the 'set_freq' parameter when reading only */
#define OSC_FREQ_DONT_SET 0

struct ec_params_osc {
	uint32_t set_fck_mhz;
} __packed;

struct ec_response_osc {
	uint16_t fck_mhz;
	uint16_t fck_min_mhz;
	uint16_t fck_max_mhz;
	uint8_t div_min;
	uint8_t div_max;
} __packed;

/* Set the 'signal' LED on the DIMM */
#define EC_CMD_DIMM_SIGNAL 0x00C2

struct ec_params_signal {
	uint8_t on_off;
};

/* Read the 1.2V VDD rail electrical parameters */
#define EC_CMD_DIMM_VDD 0x00C3

struct ec_response_vdd {
	uint16_t vdd_mv;
	uint16_t vdd_ma;
} __packed;

/* Information about the DIMM rank and it DPUs */
#define EC_CMD_RANK_INFO 0x00C4

struct ec_response_rank_info {
	/* Index of the addressed rank on this DIMM */
	uint8_t rank_index;
	/* total number of ranks on this DIMM */
	uint8_t rank_total;
	/* padding: reserved for future use */
	uint8_t padding[6];
	/* bitmap of disabled DPUs */
	uint64_t dpu_disabled;
	/* bitmap of WRAM requiring repairs (1 bit per DPU) */
	uint64_t wram_repair;
	/* bitmap of IRAM requiring repairs (1 bit per DPU) */
	uint64_t iram_repair;
} __packed;

/* Various identifiers of the DIMM */
#define EC_CMD_DIMM_ID 0x00C5

struct ec_response_dimm_id {
	char part_number[20]; /* e.g UPMEM-E19 */
	char serial_number[10];
	char dev_name[32]; /* /dev/dpu_region7/dpu_rank0 */
	char pretty_name[16]; /* e.g. B62 */
} __packed;

/* Set user-readable and system specific DIMM identifiers */
#define EC_CMD_DIMM_SET_ID 0x00C6

#define DIMM_ID_DEV_NAME 1
#define DIMM_ID_PRETTY_NAME 2

struct ec_params_dimm_id {
	uint8_t id_index; /* DIMM_ID_xxx */
	char id_string[32];
} __packed;

/* Initialize the SPD EEPROM with its default content and Serial number */
#define EC_CMD_DIMM_SPD_INIT 0x00C7

/* Set the VDDDPU rail voltage */
#define EC_CMD_DIMM_SET_VDD 0x00C8

struct ec_params_vdd {
	uint32_t vdd_mv;
} __packed;

#endif /* __DPU_MCU_CI_COMMANDS_H__ */
