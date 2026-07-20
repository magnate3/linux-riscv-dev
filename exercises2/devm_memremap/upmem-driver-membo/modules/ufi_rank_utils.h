/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#ifndef __UFI_RANK_UTILS_H__
#define __UFI_RANK_UTILS_H__

#include <linux/types.h>
#include "dpu_region.h"
#include "dpu_utils.h"
#include "dpu_control_interface.h"
#include "dpu_config.h"

#define __API_SYMBOL__
#define __builtin_popcount hweight32

#define GET_CMDS(r) ((r)->control_interface)
#define GET_DESC_HW(r) (&(r)->region->addr_translate.desc)
#define GET_CI_CONTEXT(r) (&(r)->runtime.control_interface)
#define GET_HANDLER(r) (&rank_handler)
#define GET_DEBUG(r) (NULL)

#define READ_DIR 'R'
#define WRITE_DIR 'W'

static inline u32 debug_record_last_cmd(struct dpu_rank_t *rank, char direction,
					u64 *commands)
{
	return DPU_OK;
}

#endif // __UFI_RANK_UTILS_H__
