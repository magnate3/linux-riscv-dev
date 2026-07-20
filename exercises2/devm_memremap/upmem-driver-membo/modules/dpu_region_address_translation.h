/* SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause */
/* Copyright 2020 UPMEM. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 *
 * Alternatively, this software may be distributed under the terms of the
 * GNU General Public License ("GPL") version 2 as published by the Free
 * Software Foundation.
 */
#ifndef DPU_REGION_ADDRESS_TRANSLATION_H
#define DPU_REGION_ADDRESS_TRANSLATION_H

#include <dpu_types.h>
#include <dpu_hw_description.h>
#include <linux/pci.h>

#ifdef CONFIG_X86_64
extern struct dpu_region_address_translation xeon_sp_translate;
#endif
extern struct dpu_region_address_translation fpga_kc705_translate_1dpu;
extern struct dpu_region_address_translation fpga_kc705_translate_8dpu;
extern struct dpu_region_address_translation fpga_aws_translate;
extern struct dpu_region_address_translation fpga_bittware_translate;
#ifdef CONFIG_PPC64
extern struct dpu_region_address_translation power9_translate;
#endif

enum backend {
	DPU_BACKEND_XEON_SP = 0,
	DPU_BACKEND_FPGA_KC705,
	DPU_BACKEND_FPGA_AWS,
	DPU_BACKEND_POWER9,

	DPU_BACKEND_NUMBER
};

#define CAP_SAFE (1 << 0)
#define CAP_PERF (1 << 1)
#define CAP_HYBRID_CONTROL_INTERFACE (1 << 2)
#define CAP_HYBRID_MRAM (1 << 3)
#define CAP_HYBRID (CAP_HYBRID_MRAM | CAP_HYBRID_CONTROL_INTERFACE)

#ifndef MAX_NR_DPUS_PER_RANK
#define MAX_NR_DPUS_PER_RANK 64
struct dpu_transfer_mram {
	void *ptr[MAX_NR_DPUS_PER_RANK];
	uint32_t offset_in_mram;
	uint32_t size;
};
#endif

/* Backend description of the CPU/BIOS configuration address translation:
 * hw_description:	Describe the mapping configuration (chip_id, #dpus...).
 * init_rank:		Init data structures/threads for a single rank
 * destroy_rank:	Destroys data structures/threads for a single rank
 * write_to_cis:	Writes blocks of 64 bytes that targets all CIs. The
 *			backend MUST:
 *			- interleave
 *			- byte order
 *			- nopify and send MSB
 *			bit ordering must be done by upper software layer since
 *			only a few commands require it, which is unknown at this
 *			level.
 * read_from_cis:	Reads blocks of 64 bytes from all CIs, same comment as
 *			write_block_to_ci.
 * write_to_rank:	Writes to MRAMs using the matrix of descriptions of
 *			transfers for each dpu.
 * read_from_rank:	Reads from MRAMs using the matrix of descriptions of
 *		        transfers for each dpu.
 */
struct dpu_region_address_translation {
	/* Physical topology */
	struct dpu_hw_description_t desc;

	/* Id exposed through sysfs for userspace. */
	enum backend backend_id;

	/* PERF, SAFE, HYBRID & MRAM, HYBRID & CTL IF, ... */
	uint64_t capabilities;

	/* In hybrid mode, userspace needs to know the size it needs to mmap */
	uint64_t hybrid_mmap_size;

	/* Pointer to private data for each backend implementation */
	void *private;

	/* Returns -errno on error, 0 otherwise. */
	int (*init_rank)(struct dpu_region_address_translation *tr,
			 uint8_t channel_id);
	void (*destroy_rank)(struct dpu_region_address_translation *tr,
			     uint8_t channel_id);

	void (*write_to_rank)(struct dpu_region_address_translation *tr,
			      void *base_region_addr, uint8_t channel_id,
			      struct dpu_transfer_mram *transfer_matrix);
	void (*read_from_rank)(struct dpu_region_address_translation *tr,
			       void *base_region_addr, uint8_t channel_id,
			       struct dpu_transfer_mram *transfer_matrix);

	void (*write_to_cis)(struct dpu_region_address_translation *tr,
			     void *base_region_addr, uint8_t channel_id,
			     void *block_data);
	void (*read_from_cis)(struct dpu_region_address_translation *tr,
			      void *base_region_addr, uint8_t channel_id,
			      void *block_data);

	int (*mmap_hybrid)(struct dpu_region_address_translation *tr,
			   struct file *filp, struct vm_area_struct *vma);
};

enum backend dpu_get_translation_config(struct device *dev,
					unsigned int default_backend);
void dpu_region_set_address_translation(
	struct dpu_region_address_translation *address_translate,
	enum backend backend, const struct pci_dev *pci_dev);

#endif /* DPU_REGION_ADDRESS_TRANSLATION_H */
