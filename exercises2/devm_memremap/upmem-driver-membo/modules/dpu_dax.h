/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_DAX_H
#define DPU_DAX_H

#include <linux/platform_device.h>

#include "dpu_region.h"

extern struct class *dpu_dax_class;
extern const struct attribute_group *dpu_dax_region_attrs_groups[];

int dpu_dax_init_device(struct platform_device *pdev,
			struct dpu_region *region);
void dpu_dax_release_device(struct dpu_region *region);

#endif /* DPU_DAX_H */
