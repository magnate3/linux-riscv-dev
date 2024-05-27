/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/platform_device.h>
#include "dpu_region_dev.h"

#define DPU_REGION_SIZE (16ULL * 1024 * 1024 * 1024)
#define DPU_PHYS_START_ADDR (0x100000000)

/* Hardcoded memory ranges for testing */
struct resource dpu_region_mem_resource[] = {
	//DEFINE_RES_MEM(DPU_PHYS_START_ADDR, DPU_REGION_SIZE),
	//DEFINE_RES_MEM((resource_size_t)DPU_PHYS_START_ADDR + DPU_REGION_SIZE,
	//	       DPU_REGION_SIZE),
};
#define NB_DPU_REGION (ARRAY_SIZE(dpu_region_mem_resource))

struct platform_device *dpu_region_device[NB_DPU_REGION];

int dpu_region_dev_probe(void)
{
	int res;
	int cnt = 0;

	for (res = 0; res < NB_DPU_REGION; res++) {
		u64 addr = dpu_region_mem_resource[res].start;
		u64 size = dpu_region_mem_resource[res].end -
			   dpu_region_mem_resource[res].start + 1;
		cnt += dpu_region_mem_add(addr, size, res);
	}

	pr_info("dpu_region: %d device(s) created successfully\n", cnt);
	return cnt ? 0 : -ENODEV;
}
