/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/types.h>

#include "dpu_control_interface.h"
#include "dpu_region.h"
#include "dpu_region_address_translation.h"
#include "dpu_types.h"

struct dpu_rank_handler rank_handler = {
	.commit_commands = dpu_control_interface_commit_command,
	.update_commands = dpu_control_interface_update_command,
};

u32 dpu_control_interface_commit_command(struct dpu_rank_t *rank,
					 uint64_t *command)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;

	tr->write_to_cis(tr, rank->region->base, rank->channel_id, command);

	return DPU_RANK_SUCCESS;
}

u32 dpu_control_interface_update_command(struct dpu_rank_t *rank,
					 uint64_t *result)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;

	tr->read_from_cis(tr, rank->region->base, rank->channel_id, result);

	return DPU_RANK_SUCCESS;
}
