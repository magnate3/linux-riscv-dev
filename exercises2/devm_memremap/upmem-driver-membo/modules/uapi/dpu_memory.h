/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_MEMORY_H
#define DPU_MEMORY_H

#include <linux/types.h>

#include "dpu.h"

#ifndef MAX_NR_DPUS_PER_RANK
#define MAX_NR_DPUS_PER_RANK 64
struct dpu_transfer_mram {
	void *ptr[MAX_NR_DPUS_PER_RANK];
	uint32_t offset_in_mram;
	uint32_t size;
};
#endif

void dpu_transfer_matrix_clear_all(struct dpu_rank_t *rank,
				   struct dpu_transfer_mram *transfer_matrix);
void *dpu_transfer_matrix_get_ptr(struct dpu_t *dpu,
				  struct dpu_transfer_mram *transfer_matrix);
uint32_t dpu_transfer_matrix_add_dpu(struct dpu_t *dpu,
				     struct dpu_transfer_mram *transfer_matrix,
				     const void *buffer);
uint32_t dpu_transfer_matrix_set_all(struct dpu_rank_t *rank,
				     struct dpu_transfer_mram *transfer_matrix,
				     const void *buffer);
uint32_t dpu_copy_to_mrams(struct dpu_rank_t *rank,
			   struct dpu_transfer_mram *transfer_matrix,
			   uint32_t size, uint32_t offset);
uint32_t dpu_copy_from_mram(struct dpu_t *dpu, uint8_t *destination,
			    uint32_t mram_byte_offset, uint32_t nb_of_bytes);
uint32_t dpu_copy_to_mram(struct dpu_t *dpu, uint32_t mram_byte_offset,
			  const uint8_t *source, uint32_t nb_of_bytes);
uint32_t dpu_copy_from_mrams(struct dpu_rank_t *rank,
			     struct dpu_transfer_mram *transfer_matrix,
			     uint32_t size, uint32_t offset);

uint32_t dpu_copy_to_iram_for_rank(struct dpu_rank_t *rank,
				   uint16_t iram_instruction_index,
				   const uint64_t *source,
				   uint16_t nb_of_instructions);
uint32_t dpu_copy_to_iram_for_dpu(struct dpu_t *dpu,
				  uint16_t iram_instruction_index,
				  const uint64_t *source,
				  uint16_t nb_of_instructions);

uint32_t dpu_copy_to_wram_for_rank(struct dpu_rank_t *rank,
				   uint32_t wram_word_offset,
				   const uint32_t *source,
				   uint32_t nb_of_words);
uint32_t dpu_copy_to_wram_for_dpu(struct dpu_t *dpu, uint32_t wram_word_offset,
				  const uint32_t *source, uint32_t nb_of_words);

#endif /* DPU_MEMORY_H */
