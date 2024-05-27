/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_MANAGEMENT_H
#define DPU_MANAGEMENT_H

#include <linux/types.h>

#include "dpu.h"

uint32_t dpu_rank_alloc(struct dpu_rank_t **rank);
uint32_t dpu_rank_free(struct dpu_rank_t *rank);
uint32_t dpu_get_number_of_available_ranks(void);
uint32_t dpu_get_number_of_dpus_for_rank(struct dpu_rank_t *rank);
struct dpu_t *dpu_get(struct dpu_rank_t *rank, uint8_t ci_id, uint8_t dpu_id);
uint8_t dpu_get_slice_id(struct dpu_t *dpu);
uint8_t dpu_get_member_id(struct dpu_t *dpu);

#endif /* DPU_MANAGEMENT_H */
