/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_UTILS_INCLUDE_H
#define DPU_UTILS_INCLUDE_H

#ifdef __KERNEL__
#include <linux/version.h>

#include "dpu_types.h"

#define DPU_INDEX(rank, ci, dpu)                                               \
	(((rank)->region->addr_translate.desc.topology                         \
		  .nr_of_dpus_per_control_interface *                          \
	  (ci)) +                                                              \
	 (dpu))
#define DPU_GET_UNSAFE(rank, ci, dpu) ((rank)->dpus + DPU_INDEX(rank, ci, dpu))

#define _CONCAT_X(x, y) x##y
#define _CONCAT(x, y) _CONCAT_X(x, y)

#define FF(s)                                                                  \
	do {                                                                   \
		if ((status = (s)) != DPU_OK) {                                \
			goto end;                                              \
		}                                                              \
	} while (0)

#if LINUX_VERSION_CODE <= KERNEL_VERSION(3, 10, 0)
#define page_to_virt(page) phys_to_virt(page_to_phys(page))
#endif

#define for_each_dpu_in_rank(idx, ci, dpu, nb_cis, nb_dpus_per_ci)             \
	for (dpu = 0, idx = 0; dpu < nb_dpus_per_ci; ++dpu)                    \
		for (ci = 0; ci < nb_cis; ++ci, ++idx)

#endif /* __KERNEL__ */

#endif /* DPU_UTILS_INCLUDE_H */
