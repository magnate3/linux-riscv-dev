/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/elf.h>
#include <linux/firmware.h>
#include <linux/types.h>
#include <linux/slab.h>

#include "dpu_config.h"
#include "dpu_memory.h"
#include "dpu_rank.h"
#include "dpu_region.h"
#include "dpu_runner.h"
#include "dpu_types.h"
#include "dpu_utils.h"
#include "ufi/ufi.h"

#define IRAM_MASK (0x80000000)
#define MRAM_MASK (0x08000000)
#define REGS_MASK (0xa0000000)

#define IRAM_ALIGN (3)
#define WRAM_ALIGN (2)

#define ALIGN_MASK(align) (~((1 << (align)) - 1))
#define IRAM_ALIGN_MASK ALIGN_MASK(IRAM_ALIGN)
#define WRAM_ALIGN_MASK ALIGN_MASK(WRAM_ALIGN)

static uint32_t load_iram(struct dpu_rank_t *rank, const void *content,
			  uint32_t addr, uint32_t size)
{
	uint32_t status;
	uint32_t iram_address;
	uint32_t iram_size;

	if (((addr & ~IRAM_ALIGN_MASK) != 0) ||
	    ((size & ~IRAM_ALIGN_MASK) != 0)) {
		pr_warn("Invalid IRAM access (addr = %x / size = %x)", addr,
			size);
		status = DPU_ERR_INVALID_IRAM_ACCESS;
		goto end;
	}

	iram_address = (addr & ~IRAM_MASK) >> IRAM_ALIGN;
	iram_size = size >> IRAM_ALIGN;

	FF(dpu_copy_to_iram_for_rank(rank, iram_address, content, iram_size));

end:
	return status;
}

static uint32_t load_wram(struct dpu_rank_t *rank, const void *content,
			  uint32_t addr, uint32_t size)
{
	uint32_t status;
	uint32_t wram_address;
	uint32_t wram_size;

	if (((addr & ~WRAM_ALIGN_MASK) != 0) ||
	    ((size & ~WRAM_ALIGN_MASK) != 0)) {
		pr_warn("Invalid WRAM access (addr = %x / size = %x)", addr,
			size);
		status = DPU_ERR_INVALID_WRAM_ACCESS;
		goto end;
	}

	wram_address = addr >> WRAM_ALIGN;
	wram_size = size >> WRAM_ALIGN;

	FF(dpu_copy_to_wram_for_rank(rank, wram_address, content, wram_size));

end:
	return status;
}

static uint32_t load_mram(struct dpu_rank_t *rank, const void *content,
			  uint32_t addr, uint32_t size)
{
	uint32_t status;
	uint32_t mram_address;
	struct dpu_transfer_mram transfer_matrix;
	dpu_transfer_matrix_clear_all(rank, &transfer_matrix);

	mram_address = addr & ~MRAM_MASK;

	if ((status = dpu_transfer_matrix_set_all(rank, &transfer_matrix,
						  content)) != DPU_OK) {
		goto end;
	}

	status = dpu_copy_to_mrams(rank, &transfer_matrix, size, mram_address);

end:
	return status;
}

uint32_t dpu_load(struct dpu_rank_t *rank, const char *program)
{
	uint32_t status;
	const struct firmware *firmware;
	Elf32_Ehdr ehdr;
	Elf32_Phdr *phdrs = NULL;
	int i;

	if (request_firmware_direct(&firmware, program, &rank->dev) < 0) {
		return DPU_ERR_NO_SUCH_FILE;
	}

	memcpy(&ehdr, firmware->data, sizeof(ehdr));
	if (ehdr.e_ident[EI_MAG0] != ELFMAG0 ||
	    ehdr.e_ident[EI_MAG1] != ELFMAG1 ||
	    ehdr.e_ident[EI_MAG2] != ELFMAG2 ||
	    ehdr.e_ident[EI_MAG3] != ELFMAG3) {
		pr_warn("DPU program is not a valid ELF file");
		status = DPU_ERROR_ELF_INVALID_FILE;
		goto end;
	}

	phdrs = kmalloc(sizeof(*phdrs) * ehdr.e_phnum, GFP_KERNEL);
	if (!phdrs) {
		pr_warn("Failed to allocate program headers");
		status = DPU_ERR_INTERNAL;
		goto end;
	}

	memcpy(phdrs, firmware->data + ehdr.e_phoff,
	       sizeof(*phdrs) * ehdr.e_phnum);

	for (i = 0; i < ehdr.e_phnum; i++) {
		Elf32_Phdr *phdr = &phdrs[i];

		switch (phdr->p_type) {
		case PT_LOAD:
			if (phdr->p_vaddr == REGS_MASK) {
				/* Not implemented */
			} else if ((phdr->p_vaddr & IRAM_MASK) == IRAM_MASK) {
				FF(load_iram(rank,
					     firmware->data + phdr->p_offset,
					     phdr->p_vaddr, phdr->p_memsz));
			} else if ((phdr->p_vaddr & MRAM_MASK) == MRAM_MASK) {
				FF(load_mram(rank,
					     firmware->data + phdr->p_offset,
					     phdr->p_vaddr, phdr->p_memsz));
			} else {
				FF(load_wram(rank,
					     firmware->data + phdr->p_offset,
					     phdr->p_vaddr, phdr->p_memsz));
			}
			break;
		default: /* Ignore other PT_* */
			break;
		}
	}

end:
	kfree(phdrs);
	release_firmware(firmware);
	return status;
}
EXPORT_SYMBOL(dpu_load);

uint32_t dpu_poll_rank(struct dpu_rank_t *rank, uint8_t *nb_dpu_running)
{
	uint32_t status;
	dpu_bitfield_t dpu_poll_running[DPU_MAX_NR_CIS];
	dpu_bitfield_t dpu_poll_in_fault[DPU_MAX_NR_CIS];
	struct dpu_run_context_t *run_context = &rank->runtime.run_context;
	uint8_t _nb_dpu_running;
	uint8_t each_ci, nr_cis;
	uint8_t mask = ALL_CIS;

	FF(ufi_select_all(rank, &mask));
	FF(ufi_read_dpu_run(rank, mask, dpu_poll_running));
	FF(ufi_read_dpu_fault(rank, mask, dpu_poll_in_fault));

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	_nb_dpu_running = 0;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		dpu_selected_mask_t mask_all =
			rank->runtime.control_interface.slice_info[each_ci]
				.enabled_dpus;

		run_context->dpu_in_fault[each_ci] =
			dpu_poll_in_fault[each_ci] & mask_all;
		run_context->dpu_running[each_ci] =
			(dpu_poll_running[each_ci] & mask_all) &
			(~run_context->dpu_in_fault[each_ci]);
		_nb_dpu_running += hweight8(run_context->dpu_running[each_ci]);
	}
	run_context->nb_dpu_running = _nb_dpu_running;

	if (nb_dpu_running) {
		*nb_dpu_running = _nb_dpu_running;
	}

end:
	return status;
}
EXPORT_SYMBOL(dpu_poll_rank);

uint32_t dpu_poll_dpu(struct dpu_t *dpu, bool *dpu_is_running,
		      bool *dpu_is_in_fault)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	struct dpu_run_context_t *run_context = &rank->runtime.run_context;
	dpu_slice_id_t ci_id = dpu->slice_id;
	dpu_member_id_t dpu_id = dpu->dpu_id;
	uint32_t mask_one = 1 << dpu_id;

	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}

	status = dpu_poll_rank(rank, NULL);

	*dpu_is_running = (run_context->dpu_running[ci_id] & mask_one) != 0;
	*dpu_is_in_fault = (run_context->dpu_in_fault[ci_id] & mask_one) != 0;

	return status;
}
EXPORT_SYMBOL(dpu_poll_dpu);

static uint32_t dpu_thread_boot_safe_for_rank(struct dpu_rank_t *rank,
					      uint8_t ci_mask, uint8_t thread,
					      uint8_t *previous)
{
	uint32_t status;
	uint8_t each_ci, nr_cis;
	uint8_t each_dpu, nr_dpus_per_ci;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	if (ci_mask == ALL_CIS) {
		FF(dpu_switch_mux_for_rank(rank, false));
	} else {
		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			uint8_t mux_mask = 0;

			for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
				if (!CI_MASK_ON(ci_mask, each_ci)) {
					/* Either the DPU of this slice is being booted, then you switch the mux at DPU-side, which
	                 * amounts to clearing the DPU's bit (which amounts to doing nothing here) or you take care
	                 * of not changing the mux direction.
	                 */
					mux_mask |=
						dpu_get_host_mux_mram_state(
							rank, each_ci, each_dpu)
						<< each_ci;
				}
			}
			FF(dpu_switch_mux_for_dpu_line(rank, each_dpu,
						       mux_mask));
		}
	}

	/* Don't forget to reselect all the dpus as switching mux may have changed the selection */
	FF(ufi_select_all(rank, &ci_mask));
	FF(ufi_thread_boot(rank, ci_mask, thread, previous));

end:
	return status;
}

uint32_t dpu_boot_rank(struct dpu_rank_t *rank)
{
	uint32_t status;
	struct dpu_run_context_t *run_context = &rank->runtime.run_context;
	uint32_t nb_dpu_running;
	uint8_t each_ci, nr_cis;
	uint8_t mask = ALL_CIS;

	if (run_context->nb_dpu_running != 0) {
		return DPU_ERR_DPU_ALREADY_RUNNING;
	}

	/* The implementation is copied from userspace;
	 * I'm not sure why we need to poll here.
	 */
	FF(dpu_poll_rank(rank, NULL));
	FF(dpu_thread_boot_safe_for_rank(rank, mask, DPU_BOOT_THREAD, NULL));

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nb_dpu_running = 0;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		dpu_selected_mask_t mask_all =
			rank->runtime.control_interface.slice_info[each_ci]
				.enabled_dpus;

		run_context->dpu_running[each_ci] = mask_all;
		nb_dpu_running += hweight32(mask_all);
	}
	run_context->nb_dpu_running = nb_dpu_running;

end:
	return status;
}
EXPORT_SYMBOL(dpu_boot_rank);

static uint32_t dpu_thread_boot_safe_for_dpu(struct dpu_t *dpu, uint8_t ci_mask,
					     uint8_t thread, uint8_t *previous)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	uint8_t each_ci, nr_cis;
	uint8_t mux_mask = 0;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (!CI_MASK_ON(ci_mask, each_ci)) {
			/* Either the DPU of this slice is being booted, then you switch the mux at DPU-side, which
	         * amounts to clearing the DPU's bit (which amounts to doing nothing here) or you take care
	         * of not changing the mux direction.
	         */
			mux_mask |= dpu_get_host_mux_mram_state(rank, each_ci,
								dpu->dpu_id)
				    << each_ci;
		}
	}

	FF(dpu_switch_mux_for_dpu_line(rank, dpu->dpu_id, mux_mask));

	/* Don't forget to reselect the dpu as switching mux may have changed the selection */
	FF(ufi_select_dpu(rank, &ci_mask, dpu->dpu_id));
	FF(ufi_thread_boot(rank, ci_mask, thread, previous));

end:
	return status;
}

uint32_t dpu_boot_dpu(struct dpu_t *dpu)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	struct dpu_run_context_t *run_context = &rank->runtime.run_context;
	dpu_slice_id_t ci_id = dpu->slice_id;
	dpu_member_id_t dpu_id = dpu->dpu_id;
	uint32_t mask_one = 1 << dpu_id;
	uint8_t ci_mask = CI_MASK_ONE(ci_id);
	bool dpu_is_running, dpu_is_in_fault;

	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}
	if ((run_context->dpu_running[ci_id] & mask_one) != 0) {
		return DPU_ERR_DPU_ALREADY_RUNNING;
	}

	FF(dpu_poll_dpu(dpu, &dpu_is_running, &dpu_is_in_fault));
	FF(dpu_thread_boot_safe_for_dpu(dpu, ci_mask, DPU_BOOT_THREAD, NULL));

	if (!dpu_is_running) {
		run_context->dpu_running[ci_id] |= mask_one;
		run_context->nb_dpu_running++;
	}

end:
	return status;
}
EXPORT_SYMBOL(dpu_boot_dpu);
