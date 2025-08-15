/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_RUNNER_H
#define DPU_RUNNER_H

#include <linux/types.h>

#include "dpu.h"

uint32_t dpu_reset_rank(struct dpu_rank_t *rank);
uint32_t dpu_load(struct dpu_rank_t *rank, const char *program);
uint32_t dpu_boot_rank(struct dpu_rank_t *rank);
uint32_t dpu_boot_dpu(struct dpu_t *dpu);
uint32_t dpu_poll_rank(struct dpu_rank_t *rank, uint8_t *nb_dpu_running);
uint32_t dpu_poll_dpu(struct dpu_t *dpu, bool *dpu_is_running,
		      bool *dpu_is_in_fault);

#endif /* DPU_RUNNER_H */