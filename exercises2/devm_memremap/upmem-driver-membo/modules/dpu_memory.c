/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/types.h>

#include "dpu_config.h"
#include "dpu_memory.h"
#include "dpu_region.h"
#include "dpu_region_address_translation.h"
#include "dpu_types.h"
#include "dpu_utils.h"
#include "ufi/ufi.h"

#define verify_wram_access(o, s, r)                                                       \
	do {                                                                              \
		if (!(s)) {                                                               \
			pr_warn("WARNING: wram access of size 0 at offset %d\n",          \
				o);                                                       \
			return DPU_OK;                                                    \
		}                                                                         \
		if (((o) >=                                                               \
		     (r)->region->addr_translate.desc.memories.wram_size) ||              \
		    (((o) + (s)) >                                                        \
		     (r)->region->addr_translate.desc.memories.wram_size)) {              \
			pr_warn("ERROR: invalid wram access ((%d >= %d) || (%d > %d))\n", \
				o,                                                        \
				(r)->region->addr_translate.desc.memories                 \
					.wram_size,                                       \
				(o) + (s),                                                \
				(r)->region->addr_translate.desc.memories                 \
					.wram_size);                                      \
			return DPU_ERR_INVALID_WRAM_ACCESS;                               \
		}                                                                         \
	} while (0)

#define verify_iram_access(o, s, r)                                                       \
	do {                                                                              \
		if (!(s)) {                                                               \
			pr_warn("WARNING: iram access of size 0 at offset %d\n",          \
				o);                                                       \
			return DPU_OK;                                                    \
		}                                                                         \
		if (((o) >=                                                               \
		     (r)->region->addr_translate.desc.memories.iram_size) ||              \
		    (((o) + (s)) >                                                        \
		     (r)->region->addr_translate.desc.memories.iram_size)) {              \
			pr_warn("ERROR: invalid iram access ((%d >= %d) || (%d > %d))\n", \
				o,                                                        \
				(r)->region->addr_translate.desc.memories                 \
					.iram_size,                                       \
				(o) + (s),                                                \
				(r)->region->addr_translate.desc.memories                 \
					.iram_size);                                      \
			return DPU_ERR_INVALID_IRAM_ACCESS;                               \
		}                                                                         \
	} while (0)

#define verify_mram_access_offset_and_size(o, s, r)                                                                  \
	do {                                                                                                         \
		if (((s) % 8 != 0) || ((o) % 8 != 0)) {                                                              \
			pr_warn("ERROR: invalid mram acess (size and offset need to be 8-byte aligned): (%d, %d)\n", \
				(o), (s));                                                                           \
			return DPU_ERR_INVALID_MRAM_ACCESS;                                                          \
		}                                                                                                    \
		if (((o) >=                                                                                          \
		     (r)->region->addr_translate.desc.memories.mram_size) ||                                         \
		    (((o) + (s)) >                                                                                   \
		     (r)->region->addr_translate.desc.memories.mram_size)) {                                         \
			pr_warn("ERROR: invalid mram access ((%d >= %d) || (%d > %d))\n",                            \
				o,                                                                                   \
				(r)->region->addr_translate.desc.memories                                            \
					.mram_size,                                                                  \
				(o) + (s),                                                                           \
				(r)->region->addr_translate.desc.memories                                            \
					.mram_size);                                                                 \
			return DPU_ERR_INVALID_MRAM_ACCESS;                                                          \
		}                                                                                                    \
	} while (0)

static inline uint32_t _transfer_matrix_index(struct dpu_t *dpu)
{
	return dpu->dpu_id * dpu->rank->region->addr_translate.desc.topology
				     .nr_of_control_interfaces +
	       dpu->slice_id;
}

static uint32_t do_mram_transfer(struct dpu_rank_t *rank,
				 enum dpu_transfer_type type,
				 struct dpu_transfer_mram *matrix)
{
	uint32_t status = DPU_OK;
	uint8_t each_dpu;
	uintptr_t host_ptr = 0;

	for (each_dpu = 0; each_dpu < MAX_NR_DPUS_PER_RANK; each_dpu++) {
		host_ptr |= (uintptr_t)matrix->ptr[each_dpu];
	}
	if (!host_ptr) {
		pr_debug(
			"transfer matrix does not contain any host ptr, do nothing\n");
		return status;
	}

	switch (type) {
	case DPU_TRANSFER_FROM_MRAM:
		if (dpu_rank_copy_from_rank(rank, matrix) < 0) {
			status = DPU_ERR_DRIVER;
		}
		break;
	case DPU_TRANSFER_TO_MRAM:
		if (dpu_rank_copy_to_rank(rank, matrix) < 0) {
			status = DPU_ERR_DRIVER;
		}
		break;

	default:
		status = DPU_ERR_INTERNAL;
		break;
	}

	return status;
}

static bool
duplicate_transfer_matrix(struct dpu_rank_t *rank,
			  const struct dpu_transfer_mram *transfer_matrix,
			  struct dpu_transfer_mram *even_transfer_matrix,
			  struct dpu_transfer_mram *odd_transfer_matrix)
{
	bool is_duplication_needed = false;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	if (nr_dpus_per_ci <= 1)
		return false;

	/* MRAM mux is set by pair of DPUs: DPU0-DPU1, DPU2-DPU3, DPU4-DPU5, DPU6-DPU7 have the same mux state.
     * In the case all DPUs are stopped and the transfer is authorized, we must take care not overriding MRAMs
     * whose transfer in the matrix is not defined. But with the pair of DPUs as explained above, we must
     * duplicate the transfer if one DPU of a pair has a defined transfer and not the other. Let's take an example
     * (where '1' means there is a defined transfer and '0' no transfer for this DPU should be done):
     *
     *       CI    0     1     2     3     4     5     6     7
     *          -------------------------------------------------
     *  DPU0    |  1  |  1  |  1  |  1  |  1  |  0  |  1  |  1  |
     *  DPU1    |  1  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
     *                              ....
     *
     *  In this case, we must not override the MRAM of CI5:DPU0, so we must switch the mux DPU-side. But doing so, we
     *  also switch the mux for CI5:DPU1. But CI5:DPU1 has a defined transfer, then we cannot do this at the
     *  same time and hence the duplication of the matrix.
     *  This applies only if it is the job of the API to do that and in that case matrix duplication is MANDATORY since
     *  we don't know how the backend goes through the matrix, so we must prepare all the muxes so that one transfer
     *  matrix is correct.
     *
     *  So the initial transfer matrix must be duplicated at most 2 times, one for even DPUs, one for odd DPUs:
     *
     *       CI    0     1     2     3     4     5     6     7
     *          -------------------------------------------------
     *  DPU0    |  1  |  1  |  1  |  1  |  1  |  0  |  1  |  1  |
     *  DPU1    |  0  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
     *
     *  For the matrix above, for transfers to be correct, we must duplicate it this way:
     *
     *       CI    0     1     2     3     4     5     6     7
     *          -------------------------------------------------
     *  DPU0    |  1  |  1  |  1  |  1  |  1  |  0  |  1  |  1  |
     *  DPU1    |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
     *
     *                              +
     *
     *       CI    0     1     2     3     4     5     6     7
     *          -------------------------------------------------
     *  DPU0    |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
     *  DPU1    |  0  |  1  |  1  |  1  |  1  |  1  |  1  |  1  |
     *
     *  Which amounts to, once such a duplication is detected, to split the initial transfer matrix into 2 matrix,
     *  one containing the odd line and the other the even.
     */

	/* Let's go through the matrix and search for pairs of DPUs whose one DPU has a defined transfer and the other one
        * has no transfer. If we find one such pair, let's duplicate the matrix.
        */
	for (each_dpu = 0; each_dpu < nr_dpus_per_ci; each_dpu += 2) {
		for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
			int first_dpu_idx, second_dpu_idx;
			struct dpu_t *first_dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu);
			struct dpu_t *second_dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu + 1);

			first_dpu_idx = _transfer_matrix_index(first_dpu);
			second_dpu_idx = _transfer_matrix_index(second_dpu);

			if ((!transfer_matrix->ptr[first_dpu_idx] ||
			     !transfer_matrix->ptr[second_dpu_idx]) &&
			    (transfer_matrix->ptr[first_dpu_idx] ||
			     transfer_matrix->ptr[second_dpu_idx])) {
				dpu_member_id_t dpu_id_notnull =
					transfer_matrix->ptr[first_dpu_idx] ?
						      each_dpu :
						      (dpu_member_id_t)(each_dpu + 1);

				pr_debug(
					"Duplicating transfer matrix since DPU %d of the pair (%d, %d) has a "
					"defined transfer whereas the other does not.",
					dpu_id_notnull, each_dpu, each_dpu + 1);

				is_duplication_needed = true;
				break;
			}
		}

		if (is_duplication_needed)
			break;
	}

	if (is_duplication_needed) {
		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; each_dpu += 2) {
			for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
				int first_dpu_idx, second_dpu_idx;
				struct dpu_t *first_dpu =
					DPU_GET_UNSAFE(rank, each_ci, each_dpu);
				struct dpu_t *second_dpu = DPU_GET_UNSAFE(
					rank, each_ci, each_dpu + 1);

				first_dpu_idx =
					_transfer_matrix_index(first_dpu);
				second_dpu_idx =
					_transfer_matrix_index(second_dpu);

				dpu_transfer_matrix_add_dpu(
					first_dpu, even_transfer_matrix,
					transfer_matrix->ptr[first_dpu_idx]);
				dpu_transfer_matrix_add_dpu(
					second_dpu, odd_transfer_matrix,
					transfer_matrix->ptr[second_dpu_idx]);
			}
		}
	}

	return is_duplication_needed;
}

static bool
is_transfer_matrix_full(struct dpu_rank_t *rank,
			const struct dpu_transfer_mram *transfer_matrix)
{
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		dpu_selected_mask_t enabled_dpus =
			rank->runtime.control_interface.slice_info[each_ci]
				.enabled_dpus;

		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			struct dpu_t *dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu);
			int dpu_idx = _transfer_matrix_index(dpu);

			if (!(enabled_dpus & (1 << each_dpu)))
				continue;

			if (!transfer_matrix->ptr[dpu_idx])
				return false;
		}
	}

	return true;
}

static uint32_t host_get_access_for_transfer_matrix(
	struct dpu_rank_t *rank,
	const struct dpu_transfer_mram *transfer_matrix)
{
	uint32_t status = DPU_OK;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	for (each_dpu = 0; each_dpu < nr_dpus_per_ci; each_dpu += 2) {
		uint8_t mask = 0;

		for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
			struct dpu_t *first_dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu);
			struct dpu_t *second_dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu + 1);

			int idx_first_dpu = _transfer_matrix_index(first_dpu);
			int idx_second_dpu = _transfer_matrix_index(second_dpu);

			uint8_t get_mux_first_dpu =
				(transfer_matrix->ptr[idx_first_dpu] != NULL);
			uint8_t get_mux_second_dpu =
				(transfer_matrix->ptr[idx_second_dpu] != NULL);

			if (first_dpu->enabled) {
				mask |= get_mux_first_dpu << each_ci;
			} else if (second_dpu->enabled) {
				mask |= get_mux_second_dpu << each_ci;
			}
		}

		if (mask) {
			FF(dpu_switch_mux_for_dpu_line(rank, each_dpu, mask));
			FF(dpu_switch_mux_for_dpu_line(rank, each_dpu + 1,
						       mask));
		}
	}

end:
	return status;
}

static uint32_t
host_get_access_for_transfer(struct dpu_rank_t *rank,
			     const struct dpu_transfer_mram *transfer_matrix)
{
	uint32_t status;
	bool is_full_matrix = is_transfer_matrix_full(rank, transfer_matrix);

	if (is_full_matrix)
		FF(dpu_switch_mux_for_rank(rank, true));
	else
		FF(host_get_access_for_transfer_matrix(rank, transfer_matrix));

end:
	return status;
}

static uint32_t
copy_mram_for_dpus(struct dpu_rank_t *rank, enum dpu_transfer_type type,
		   const struct dpu_transfer_mram *transfer_matrix)
{
	uint32_t status;
	struct dpu_transfer_mram even_transfer_matrix, odd_transfer_matrix;
	bool is_duplication_needed = false;

	dpu_transfer_matrix_clear_all(rank, &even_transfer_matrix);
	dpu_transfer_matrix_clear_all(rank, &odd_transfer_matrix);

	is_duplication_needed = duplicate_transfer_matrix(rank, transfer_matrix,
							  &even_transfer_matrix,
							  &odd_transfer_matrix);

	if (!is_duplication_needed) {
		FF(host_get_access_for_transfer(rank, transfer_matrix));

		FF(do_mram_transfer(
			rank, type,
			(struct dpu_transfer_mram *)transfer_matrix));
	} else {
		even_transfer_matrix.size = transfer_matrix->size;
		even_transfer_matrix.offset_in_mram =
			transfer_matrix->offset_in_mram;
		FF(host_get_access_for_transfer_matrix(rank,
						       &even_transfer_matrix));
		FF(do_mram_transfer(rank, type, &even_transfer_matrix));

		odd_transfer_matrix.size = transfer_matrix->size;
		odd_transfer_matrix.offset_in_mram =
			transfer_matrix->offset_in_mram;
		FF(host_get_access_for_transfer_matrix(rank,
						       &odd_transfer_matrix));
		FF(do_mram_transfer(rank, type, &odd_transfer_matrix));
	}
end:
	return status;
}

void dpu_transfer_matrix_clear_all(struct dpu_rank_t *rank,
				   struct dpu_transfer_mram *transfer_matrix)
{
	memset(transfer_matrix, 0, sizeof(struct dpu_transfer_mram));
}
EXPORT_SYMBOL(dpu_transfer_matrix_clear_all);

void *dpu_transfer_matrix_get_ptr(struct dpu_t *dpu,
				  struct dpu_transfer_mram *transfer_matrix)
{
	uint32_t dpu_index;

	if (!dpu->enabled) {
		return NULL;
	}

	dpu_index = _transfer_matrix_index(dpu);
	return transfer_matrix->ptr[dpu_index];
}
EXPORT_SYMBOL(dpu_transfer_matrix_get_ptr);

uint32_t dpu_transfer_matrix_add_dpu(struct dpu_t *dpu,
				     struct dpu_transfer_mram *transfer_matrix,
				     const void *buffer)
{
	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}

	transfer_matrix->ptr[_transfer_matrix_index(dpu)] = (void *)buffer;

	return DPU_OK;
}
EXPORT_SYMBOL(dpu_transfer_matrix_add_dpu);

uint32_t dpu_transfer_matrix_set_all(struct dpu_rank_t *rank,
				     struct dpu_transfer_mram *transfer_matrix,
				     const void *buffer)
{
	uint32_t status;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			struct dpu_t *dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu);

			if (dpu->enabled) {
				if ((status = dpu_transfer_matrix_add_dpu(
					     dpu, transfer_matrix, buffer)) !=
				    DPU_OK) {
					return status;
				}
			}
		}
	}

	return DPU_OK;
}
EXPORT_SYMBOL(dpu_transfer_matrix_set_all);

uint32_t dpu_copy_to_mram(struct dpu_t *dpu, uint32_t mram_byte_offset,
			  const uint8_t *source, uint32_t nb_of_bytes)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	struct dpu_transfer_mram transfer_matrix;

	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}

	if (!nb_of_bytes) {
		return DPU_OK;
	}

	dpu_transfer_matrix_clear_all(rank, &transfer_matrix);

	if ((status = dpu_transfer_matrix_add_dpu(dpu, &transfer_matrix,
						  (void *)source)) != DPU_OK) {
		goto end;
	}

	status = dpu_copy_to_mrams(rank, &transfer_matrix, nb_of_bytes,
				   mram_byte_offset);

end:
	return status;
}
EXPORT_SYMBOL(dpu_copy_to_mram);

uint32_t dpu_copy_from_mram(struct dpu_t *dpu, uint8_t *destination,
			    uint32_t mram_byte_offset, uint32_t nb_of_bytes)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	struct dpu_transfer_mram transfer_matrix;

	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}

	if (!nb_of_bytes) {
		return DPU_OK;
	}

	dpu_transfer_matrix_clear_all(rank, &transfer_matrix);

	if ((status = dpu_transfer_matrix_add_dpu(
		     dpu, &transfer_matrix, (void *)destination)) != DPU_OK) {
		goto end;
	}

	status = dpu_copy_from_mrams(rank, &transfer_matrix, nb_of_bytes,
				     mram_byte_offset);

end:
	return status;
}
EXPORT_SYMBOL(dpu_copy_from_mram);

uint32_t dpu_copy_to_mrams(struct dpu_rank_t *rank,
			   struct dpu_transfer_mram *transfer_matrix,
			   uint32_t size, uint32_t offset)
{
	uint32_t status;

	transfer_matrix->size = size;
	transfer_matrix->offset_in_mram = offset;

	verify_mram_access_offset_and_size(offset, size, rank);

	if (rank->runtime.run_context.nb_dpu_running > 0) {
		pr_warn("Host does not have access to the MRAM because %u DPU%s running.",
			rank->runtime.run_context.nb_dpu_running,
			rank->runtime.run_context.nb_dpu_running > 1 ? "s are" :
									     " is");
		return DPU_ERR_MRAM_BUSY;
	}
	status =
		copy_mram_for_dpus(rank, DPU_TRANSFER_TO_MRAM, transfer_matrix);

	return status;
}
EXPORT_SYMBOL(dpu_copy_to_mrams);

uint32_t dpu_copy_from_mrams(struct dpu_rank_t *rank,
			     struct dpu_transfer_mram *transfer_matrix,
			     uint32_t size, uint32_t offset)
{
	uint32_t status;

	transfer_matrix->size = size;
	transfer_matrix->offset_in_mram = offset;

	verify_mram_access_offset_and_size(offset, size, rank);

	if (rank->runtime.run_context.nb_dpu_running > 0) {
		pr_warn("Host does not have access to the MRAM because %u DPU%s running.",
			rank->runtime.run_context.nb_dpu_running,
			rank->runtime.run_context.nb_dpu_running > 1 ? "s are" :
									     " is");
		return DPU_ERR_MRAM_BUSY;
	}

	status = copy_mram_for_dpus(rank, DPU_TRANSFER_FROM_MRAM,
				    transfer_matrix);

	return status;
}
EXPORT_SYMBOL(dpu_copy_from_mrams);

uint32_t dpu_copy_to_iram_for_rank(struct dpu_rank_t *rank,
				   uint16_t iram_instruction_index,
				   const uint64_t *source,
				   uint16_t nb_of_instructions)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;
	dpuinstruction_t *iram_array[DPU_MAX_NR_CIS] = {
		[0 ... DPU_MAX_NR_CIS - 1] = (dpuinstruction_t *)source
	};

	verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

	FF(ufi_select_all(rank, &mask));
	FF(ufi_iram_write(rank, mask, iram_array, iram_instruction_index,
			  nb_of_instructions));

end:
	return status;
}
EXPORT_SYMBOL(dpu_copy_to_iram_for_rank);

uint32_t dpu_copy_to_iram_for_dpu(struct dpu_t *dpu,
				  uint16_t iram_instruction_index,
				  const uint64_t *source,
				  uint16_t nb_of_instructions)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	dpu_slice_id_t slice_id = dpu->slice_id;
	uint8_t mask = CI_MASK_ONE(slice_id);
	dpuinstruction_t *iram_array[DPU_MAX_NR_CIS];
	iram_array[slice_id] = (dpuinstruction_t *)source;

	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}

	verify_iram_access(iram_instruction_index, nb_of_instructions, rank);

	FF(ufi_select_dpu(rank, &mask, dpu->dpu_id));
	FF(ufi_iram_write(rank, mask, iram_array, iram_instruction_index,
			  nb_of_instructions));

end:
	return status;
}
EXPORT_SYMBOL(dpu_copy_to_iram_for_dpu);

uint32_t dpu_copy_to_wram_for_rank(struct dpu_rank_t *rank,
				   uint32_t wram_word_offset,
				   const uint32_t *source, uint32_t nb_of_words)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;
	dpuword_t *wram_array[DPU_MAX_NR_CIS] = { [0 ... DPU_MAX_NR_CIS - 1] =
							  (dpuword_t *)source };

	verify_wram_access(wram_word_offset, nb_of_words, rank);

	FF(ufi_select_all(rank, &mask));
	FF(ufi_wram_write(rank, mask, wram_array, wram_word_offset,
			  nb_of_words));

end:
	return status;
}
EXPORT_SYMBOL(dpu_copy_to_wram_for_rank);

uint32_t dpu_copy_to_wram_for_dpu(struct dpu_t *dpu, uint32_t wram_word_offset,
				  const uint32_t *source, uint32_t nb_of_words)
{
	uint32_t status;
	struct dpu_rank_t *rank = dpu->rank;
	dpu_slice_id_t slice_id = dpu->slice_id;
	uint8_t mask = CI_MASK_ONE(slice_id);
	dpuword_t *wram_array[DPU_MAX_NR_CIS];
	wram_array[slice_id] = (dpuword_t *)source;

	if (!dpu->enabled) {
		return DPU_ERR_DPU_DISABLED;
	}

	verify_wram_access(wram_word_offset, nb_of_words, rank);

	FF(ufi_select_dpu(rank, &mask, dpu->dpu_id));
	FF(ufi_wram_write(rank, mask, wram_array, wram_word_offset,
			  nb_of_words));

end:
	return status;
}
EXPORT_SYMBOL(dpu_copy_to_wram_for_dpu);
