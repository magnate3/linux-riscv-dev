/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright 2019 - UPMEM
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/acpi.h>
#include <linux/device.h>
#include <linux/platform_device.h>

#include "dpu_rank.h"
#include "dpu_region_dev.h"

/* Flag for Processing-In-Memory regions (not standardized yet) */
#define ACPI_SRAT_MEM_PIM (1 << 3)

int dpu_region_srat_probe(void)
{
	acpi_status status;
	struct acpi_table_srat *srat;
	uintptr_t entry, end;
	int cnt = 0;

	status = acpi_get_table(ACPI_SIG_SRAT, 0,
				(struct acpi_table_header **)&srat);
	if (ACPI_FAILURE(status))
		return -ENODEV;

	end = (uintptr_t)srat + srat->header.length;
	entry = (uintptr_t)srat + sizeof(struct acpi_table_srat);
	while (entry < end) {
		struct acpi_srat_mem_affinity *ma = (void *)entry;
		if (ma->header.type == ACPI_SRAT_TYPE_MEMORY_AFFINITY &&
		    ma->flags & ACPI_SRAT_MEM_PIM) {
			u64 addr = ma->base_address;
			u64 size = ma->length;
			u64 chunk_len;
			while ((chunk_len = min(size, DPU_RANK_SIZE))) {
				cnt += dpu_region_mem_add(addr, chunk_len, cnt);
				size -= chunk_len;
				addr += chunk_len;
			}
		}
		entry += ma->header.length;
	}

	if (cnt == 0) {
		pr_warn("No PIM device in the SRAT entries.\n");
		return -ENODEV;
	}
	return 0;
}

/* Find the proximity domain associated to base_region_addr */
int dpu_region_srat_get_pxm(u64 base_region_addr)
{
	acpi_status status;
	struct acpi_table_srat *srat;
	uintptr_t entry, end;

	status = acpi_get_table(ACPI_SIG_SRAT, 0,
				(struct acpi_table_header **)&srat);
	if (ACPI_FAILURE(status))
		return -1;

	end = (uintptr_t)srat + srat->header.length;
	entry = (uintptr_t)srat + sizeof(struct acpi_table_srat);
	while (entry < end) {
		struct acpi_srat_mem_affinity *ma = (void *)entry;
		if (ma->header.type == ACPI_SRAT_TYPE_MEMORY_AFFINITY &&
		    ma->flags & ACPI_SRAT_MEM_PIM) {
			u64 addr = ma->base_address;
			u64 size = ma->length;
			u64 chunk_len;
			while ((chunk_len = min(size, DPU_RANK_SIZE))) {
				if (addr == base_region_addr)
					return ma->proximity_domain;
				size -= chunk_len;
				addr += chunk_len;
			}
		}
		entry += ma->header.length;
	}

	pr_warn("Failed to find proximity domain associated to address 0x%016llx\n",
		base_region_addr);
	return -1;
}