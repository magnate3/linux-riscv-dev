/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_RANK_MCU_H
#define DPU_RANK_MCU_H
#include <linux/types.h>

#include <dpu_region.h>
#include <dpu_rank.h>

int dpu_rank_mcu_set_vdd_dpu(struct dpu_rank_t *rank, uint32_t vdd_mv);
int dpu_rank_mcu_get_version(struct dpu_rank_t *rank);
int dpu_rank_mcu_get_frequency_and_clock_div(struct dpu_rank_t *rank);
int dpu_rank_mcu_set_rank_name(struct dpu_rank_t *rank);
int dpu_rank_mcu_get_rank_index_count(struct dpu_rank_t *rank);
int dpu_rank_mcu_get_vpd(struct dpu_rank_t *rank);
int dpu_rank_mcu_probe(struct dpu_rank_t *rank);

#endif /* DPU_RANK_MCU_H */
