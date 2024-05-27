/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_CONTROL_INTERFACE_H
#define DPU_CONTROL_INTERFACE_H

#include "dpu_rank.h"

extern struct dpu_rank_handler rank_handler;

u32 dpu_control_interface_commit_command(struct dpu_rank_t *rank,
					 uint64_t *command);
u32 dpu_control_interface_update_command(struct dpu_rank_t *rank,
					 uint64_t *result);

#endif /* DPU_CONTROL_INTERFACE_H */