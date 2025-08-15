/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_CONFIG_H
#define DPU_CONFIG_H

#include <linux/types.h>

#include <dpu_region.h>
#include <dpu_types.h>

uint32_t dpu_reset_rank(struct dpu_rank_t *rank);
uint32_t dpu_set_chip_id(struct dpu_rank_t *rank);
uint32_t dpu_soft_reset(struct dpu_rank_t *rank,
			dpu_clock_division_t clock_division);
uint32_t dpu_switch_mux_for_rank(struct dpu_rank_t *rank,
				 bool set_mux_for_host);
uint32_t dpu_switch_mux_for_dpu_line(struct dpu_rank_t *rank, uint8_t dpu_id,
				     uint8_t mask);
uint8_t dpu_get_host_mux_mram_state(struct dpu_rank_t *rank,
				    dpu_slice_id_t ci_id,
				    dpu_member_id_t dpu_id);
void dpu_set_host_mux_mram_state(struct dpu_rank_t *rank, dpu_slice_id_t ci_id,
				 dpu_member_id_t dpu_id, bool set);

#endif
