/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_POWER_MANAGEMENT_H
#define DPU_POWER_MANAGEMENT_H

#include <dpu_region.h>

void dpu_power_rank_enter_saving_mode(struct dpu_rank_t *rank);
void dpu_power_rank_exit_saving_mode(struct dpu_rank_t *rank);

/* 
 * Weird to pass struct dpu_rank_t as argument, but we don't have
 * a dimm structure yet.
 */
void dpu_power_dimm_enter_saving_mode(struct dpu_rank_t *rank);
void dpu_power_dimm_exit_saving_mode(struct dpu_rank_t *rank);

#endif /* DPU_POWER_MANAGEMENT_H */
