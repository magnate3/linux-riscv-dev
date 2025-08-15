/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_TYPES_H
#define DPU_TYPES_H

#include <linux/types.h>
#include <dpu_chip_id.h>

struct dpu_rank_t;

/* Maximum number of DPUs in a CI */
#define DPU_MAX_NR_DPUS_PER_CI 8

/* Maximum number of Control Interfaces in a DPU rank */
#define DPU_MAX_NR_CIS 8

/* Maximum number of DPUs in a DPU rank */
#define DPU_MAX_NR_DPUS (DPU_MAX_NR_CIS * DPU_MAX_NR_DPUS_PER_CI)

/* Maximum number of groups in a Control Interface */
#define DPU_MAX_NR_GROUPS 8

/* Number of the DPU thread which is launched when booting the DPU */
#define DPU_BOOT_THREAD 0

/* ID of a DPU rank slice */
typedef uint8_t dpu_slice_id_t;

/* ID of a DPU rank slice member */
typedef uint8_t dpu_member_id_t;

/* ID of a DPU rank slice group */
typedef uint8_t dpu_group_id_t;

/* Size in IRAM */
typedef uint16_t iram_size_t;

/* Size in WRAM */
typedef uint32_t wram_size_t;

/* Size in MRAM */
typedef uint32_t mram_size_t;

/* DPU instruction */
typedef uint64_t dpuinstruction_t;

/* DPU word in WRAM */
typedef uint32_t dpuword_t;

/* Bitfield of CIs in a rank */
typedef uint8_t dpu_ci_bitfield_t;

/* Bitfield of DPUs in a CI. */
typedef uint8_t dpu_bitfield_t;

/* Bitfield of selected DPUs */
typedef uint32_t dpu_selected_mask_t;

typedef enum _dpu_clock_division_t {
	DPU_CLOCK_DIV8 = 0x0,
	DPU_CLOCK_DIV4 = 0x4,
	DPU_CLOCK_DIV3 = 0x3,
	DPU_CLOCK_DIV2 = 0x8,
} dpu_clock_division_t;

enum dpu_slice_target_type {
	DPU_SLICE_TARGET_NONE,
	DPU_SLICE_TARGET_DPU,
	DPU_SLICE_TARGET_ALL,
	DPU_SLICE_TARGET_GROUP,
	NR_OF_DPU_SLICE_TARGETS
};

struct dpu_slice_target {
	enum dpu_slice_target_type type;
	union {
		dpu_member_id_t dpu_id;
		dpu_group_id_t group_id;
	};
};

struct dpu_bit_config {
	uint16_t cpu2dpu;
	uint16_t dpu2cpu;
	uint8_t nibble_swap;
	uint8_t stutter;
};

enum dpu_rank_status {
	DPU_RANK_SUCCESS = 0,
	DPU_RANK_COMMUNICATION_ERROR,
	DPU_RANK_BACKEND_ERROR,
	DPU_RANK_SYSTEM_ERROR,
	DPU_RANK_INVALID_PROPERTY_ERROR,
	DPU_RANK_ENODEV,
};

struct dpu_rank_handler {
	u32 (*commit_commands)(struct dpu_rank_t *rank, u64 *buffer);
	u32 (*update_commands)(struct dpu_rank_t *rank, u64 *buffer);
};

struct dpu_carousel_config {
	uint8_t cmd_duration;
	uint8_t cmd_sampling;
	uint8_t res_duration;
	uint8_t res_sampling;
};

struct dpu_repair_config {
	uint8_t AB_msbs;
	uint8_t CD_msbs;
	uint8_t A_lsbs;
	uint8_t B_lsbs;
	uint8_t C_lsbs;
	uint8_t D_lsbs;
	uint8_t even_index;
	uint8_t odd_index;
};

enum dpu_temperature {
	DPU_TEMPERATURE_LESS_THAN_50 = 0,
	DPU_TEMPERATURE_BETWEEN_50_AND_60 = 1,
	DPU_TEMPERATURE_BETWEEN_60_AND_70 = 2,
	DPU_TEMPERATURE_BETWEEN_70_AND_80 = 3,
	DPU_TEMPERATURE_BETWEEN_80_AND_90 = 4,
	DPU_TEMPERATURE_BETWEEN_90_AND_100 = 5,
	DPU_TEMPERATURE_BETWEEN_100_AND_110 = 6,
	DPU_TEMPERATURE_GREATER_THAN_110 = 7,
};

enum dpu_pc_mode {
	DPU_PC_MODE_12 = 0,
	DPU_PC_MODE_13 = 1,
	DPU_PC_MODE_14 = 2,
	DPU_PC_MODE_15 = 3,
	DPU_PC_MODE_16 = 4,
};

enum dpu_error {
	DPU_OK,
	DPU_ERR_DRIVER,
	DPU_ERR_INTERNAL,
	DPU_ERR_TIMEOUT,
	DPU_ERR_NO_SUCH_FILE,
	DPU_ERR_INVALID_WRAM_ACCESS,
	DPU_ERR_INVALID_IRAM_ACCESS,
	DPU_ERR_INVALID_MRAM_ACCESS,
	DPU_ERR_MRAM_BUSY,
	DPU_ERR_DPU_ALREADY_RUNNING,
	DPU_ERROR_ELF_INVALID_FILE,
	DPU_ERR_DPU_DISABLED,
};

enum dpu_transfer_type {
	DPU_TRANSFER_FROM_MRAM,
	DPU_TRANSFER_TO_MRAM,
};

#define LOGV_PACKET(r, d, t)
#define LOG_CI(l, r, c, f, ...)
#define LOG_RANK(l, r, f, ...)
#define LOG_TEMPERATURE_ENABLED() 0
#define LOG_TEMPERATURE_TRIGGERED(r) 0
#define LOG_TEMPERATURE(r, d)
#define LOGD_LAST_COMMANDS(r)

#endif /*_DPU_TYPES_H_ */
