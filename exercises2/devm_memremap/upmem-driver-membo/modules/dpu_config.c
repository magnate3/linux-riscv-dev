/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/firmware.h>
#include <linux/mutex.h>
#include <linux/printk.h>
#include <linux/types.h>

#include "dpu_config.h"
#include "dpu_region.h"
#include "dpu_types.h"
#include "dpu_utils.h"
#include "ufi/ufi.h"
#include "ufi/ufi_bit_config.h"
#include "ufi/ufi_dma_wavegen_config.h"

#define BYTE_ORDER_EXPECTED 0x000103FF0F8FCFEFULL
// #define BIT_ORDER_EXPECTED 0x0F884422

#define NR_OF_WRAM_BANKS 4

#define REFRESH_MODE_VALUE 4

/* Bit set when the DPU has the control of the bank */
#define MUX_DPU_BANK_CTRL (1 << 0)
/* Bit set when the DPU can write to the bank */
#define MUX_DPU_WRITE_CTRL (1 << 1)
/* Bit set when the host or the DPU wrote to the bank without permission */
#define MUX_COLLISION_ERR (1 << 7)

#define WAVEGEN_MUX_HOST_EXPECTED 0x00
#define WAVEGEN_MUX_DPU_EXPECTED (MUX_DPU_BANK_CTRL | MUX_DPU_WRITE_CTRL)

static bool byte_order_values_are_compatible(uint64_t reference, uint64_t found)
{
	uint8_t each_byte;

	for (each_byte = 0; each_byte < sizeof(uint64_t); ++each_byte) {
		if (hweight8(
			    (uint8_t)((reference >> (8 * each_byte)) & 0xFF)) !=
		    hweight8((uint8_t)((found >> (8 * each_byte)) & 0xFF))) {
			return false;
		}
	}

	return true;
}

static uint32_t dpu_byte_order(struct dpu_rank_t *rank)
{
	uint32_t status;
	uint64_t *byte_order_results = rank->data;
	uint8_t mask = ALL_CIS;
	uint8_t nr_cis, each_ci;
	bool execute_byte_order = false;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;

	FF(ufi_select_cis(rank, &mask));
	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (CI_MASK_ON(mask, each_ci) &&
		    !rank->runtime.control_interface.slice_info[each_ci]
			     .byte_order) {
			execute_byte_order = true;
			break;
		}
	}

	if (!execute_byte_order) {
		for (each_ci = 0; each_ci < nr_cis; ++each_ci)
			byte_order_results[each_ci] =
				rank->runtime.control_interface
					.slice_info[each_ci]
					.byte_order;
	} else {
		FF(ufi_byte_order(rank, mask, byte_order_results));
		for (each_ci = 0; each_ci < nr_cis; ++each_ci)
			rank->runtime.control_interface.slice_info[each_ci]
				.byte_order = byte_order_results[each_ci];
	}

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (!CI_MASK_ON(mask, each_ci))
			continue;

		if (!byte_order_values_are_compatible(
			    BYTE_ORDER_EXPECTED, byte_order_results[each_ci])) {
			pr_warn("invalid byte order (reference: 0x%016llx; found: 0x%016llx)\n",
				BYTE_ORDER_EXPECTED,
				byte_order_results[each_ci]);
			status = DPU_ERR_INTERNAL;
		}
	}

end:
	return status;
}

static uint32_t dpu_bit_config(struct dpu_rank_t *rank,
			       struct dpu_bit_config *config)
{
	uint32_t status;
	uint32_t bit_config_results[DPU_MAX_NR_CIS];
	uint32_t bit_config_result;
	uint8_t mask = ALL_CIS;
	uint8_t nr_cis, each_ci;

	FF(ufi_select_cis(rank, &mask));
	FF(ufi_bit_config(rank, mask, NULL, bit_config_results));

	bit_config_result = bit_config_results[__builtin_ctz(mask)];
	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;

	/* Let's verify that all CIs have the same bit config result as the first CI. */
	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (!CI_MASK_ON(mask, each_ci))
			continue;

		if (bit_config_results[each_ci] != bit_config_result) {
			pr_warn("inconsistent bit configuration between the different CIs (0x%08x != 0x%08x)",
				bit_config_results[each_ci], bit_config_result);
			status = DPU_ERR_INTERNAL;
			goto end;
		}
	}

	dpu_bit_config_compute(bit_config_result, config);

	config->stutter = 0;

	pr_debug(
		"bit_order: 0x%08x nibble_swap: 0x%02x cpu_to_dpu: 0x%04x dpu_to_cpu: 0x%04x",
		bit_config_result, config->nibble_swap, config->cpu2dpu,
		config->dpu2cpu);

end:
	return status;
}

static uint32_t dpu_ci_shuffling_box_config(struct dpu_rank_t *rank,
					    const struct dpu_bit_config *config)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;

	FF(ufi_select_cis(rank, &mask));
	FF(ufi_bit_config(rank, mask, config, NULL));

end:
	return status;
}

static uint32_t dpu_identity(struct dpu_rank_t *rank)
{
	uint32_t status;
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint32_t identity_results[DPU_MAX_NR_CIS];
	uint32_t identity_result;
	uint8_t mask = ALL_CIS;
	uint8_t nr_cis, each_ci;

	FF(ufi_select_cis(rank, &mask));
	FF(ufi_identity(rank, mask, identity_results));

	identity_result = identity_results[__builtin_ctz(mask)];
	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;

	/* Let's verify that all CIs have the same identity as the first CI. */
	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (!CI_MASK_ON(mask, each_ci))
			continue;

		if (identity_results[each_ci] != identity_result) {
			pr_warn("inconsistent identity between the different CIs (0x%08x != 0x%08x)",
				identity_results[each_ci], identity_result);
			status = DPU_ERR_INTERNAL;
			goto end;
		}
	}

	tr->desc.signature.chip_id = identity_result;

	pr_debug("chip ID: 0x%08x", identity_result);

end:
	return status;
}

static enum dpu_temperature from_celsius_to_dpu_enum(uint8_t temperature)
{
	if (temperature < 50)
		return DPU_TEMPERATURE_LESS_THAN_50;
	if (temperature < 60)
		return DPU_TEMPERATURE_BETWEEN_50_AND_60;
	if (temperature < 70)
		return DPU_TEMPERATURE_BETWEEN_60_AND_70;
	if (temperature < 80)
		return DPU_TEMPERATURE_BETWEEN_70_AND_80;
	if (temperature < 90)
		return DPU_TEMPERATURE_BETWEEN_80_AND_90;
	if (temperature < 100)
		return DPU_TEMPERATURE_BETWEEN_90_AND_100;
	if (temperature < 110)
		return DPU_TEMPERATURE_BETWEEN_100_AND_110;

	return DPU_TEMPERATURE_GREATER_THAN_110;
}

static uint32_t dpu_thermal_config(struct dpu_rank_t *rank,
				   uint8_t thermal_config)
{
	uint32_t status;
	enum dpu_temperature temperature =
		from_celsius_to_dpu_enum(thermal_config);
	uint8_t mask = ALL_CIS;

	pr_debug("%d°C (value: 0x%04x)", thermal_config, temperature);

	FF(ufi_select_cis(rank, &mask));
	ufi_thermal_config(rank, mask, temperature);

end:
	return status;
}

static uint32_t dpu_carousel_config(struct dpu_rank_t *rank,
				    const struct dpu_carousel_config *config)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;

	pr_debug(
		"cmd_duration: %d cmd_sampling: %d res_duration: %d res_sampling: %d",
		config->cmd_duration, config->cmd_sampling,
		config->res_duration, config->res_sampling);

	FF(ufi_select_all(rank, &mask));
	FF(ufi_carousel_config(rank, mask, config));

end:
	return status;
}

static uint32_t dpu_iram_repair_config(struct dpu_rank_t *rank)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;

	/* IRAM repair disabled */
	struct dpu_repair_config config = { 0, 0, 0, 0, 0, 0, 0, 1 };
	struct dpu_repair_config *config_array[DPU_MAX_NR_CIS] = {
		[0 ... DPU_MAX_NR_CIS - 1] = &config
	};

	FF(ufi_select_all(rank, &mask));
	FF(ufi_iram_repair_config(rank, mask, config_array));

end:
	return status;
}

static uint32_t dpu_wram_repair_config(struct dpu_rank_t *rank)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;
	uint8_t each_wram_bank;

	/* WRAM repair disabled */
	struct dpu_repair_config config = { 0, 0, 0, 0, 0, 0, 0, 1 };
	struct dpu_repair_config *config_array[DPU_MAX_NR_CIS] = {
		[0 ... DPU_MAX_NR_CIS - 1] = &config
	};

	FF(ufi_select_all(rank, &mask));
	for (each_wram_bank = 0; each_wram_bank < NR_OF_WRAM_BANKS;
	     ++each_wram_bank) {
		FF(ufi_wram_repair_config(rank, mask, each_wram_bank,
					  config_array));
	}

end:
	return status;
}

static uint32_t dpu_clear_debug(struct dpu_rank_t *rank)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;

	FF(ufi_select_all(rank, &mask));

	FF(ufi_clear_debug_replace(rank, mask));
	FF(ufi_clear_fault_poison(rank, mask));
	FF(ufi_clear_fault_bkp(rank, mask));
	FF(ufi_clear_fault_dma(rank, mask));
	FF(ufi_clear_fault_mem(rank, mask));
	FF(ufi_clear_fault_dpu(rank, mask));
	FF(ufi_clear_fault_intercept(rank, mask));

end:
	return status;
}

static uint32_t dpu_clear_run_bits(struct dpu_rank_t *rank)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;
	uint32_t nr_run_bits, each_bit;

	FF(ufi_select_all(rank, &mask));

	nr_run_bits = rank->region->addr_translate.desc.dpu.nr_of_threads +
		      rank->region->addr_translate.desc.dpu.nr_of_notify_bits;

	for (each_bit = 0; each_bit < nr_run_bits; ++each_bit) {
		FF(ufi_clear_run_bit(rank, mask, each_bit, NULL));
	}

end:
	return status;
}

static uint32_t dpu_set_pc_mode(struct dpu_rank_t *rank,
				enum dpu_pc_mode pc_mode)
{
	uint32_t status;
	enum dpu_pc_mode pc_modes[DPU_MAX_NR_CIS] = { [0 ... DPU_MAX_NR_CIS -
						       1] = pc_mode };
	uint8_t mask = ALL_CIS;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;

	FF(ufi_select_all(rank, &mask));
	FF(ufi_set_pc_mode(rank, mask, pc_modes));

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
		mask = ALL_CIS;
		FF(ufi_select_dpu(rank, &mask, each_dpu));
		FF(ufi_get_pc_mode(rank, mask, pc_modes));

		for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
			struct dpu_t *dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu);
			if (dpu->enabled) {
				if (pc_modes[each_dpu] != pc_mode) {
					pr_warn("invalid PC mode (expected: %d, found: %d)",
						pc_mode, pc_modes[each_dpu]);
					status = DPU_ERR_INTERNAL;
					break;
				}
			}
		}
	}

end:
	return status;
}

static uint32_t dpu_set_stack_direction(struct dpu_rank_t *rank,
					bool stack_is_up)
{
	uint32_t status;
	uint8_t previous_directions[DPU_MAX_NR_CIS];
	bool stack_directions[DPU_MAX_NR_CIS] = { [0 ... DPU_MAX_NR_CIS - 1] =
							  stack_is_up };
	uint8_t mask = ALL_CIS;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;

	FF(ufi_select_all(rank, &mask));
	FF(ufi_set_stack_direction(rank, mask, stack_directions, NULL));
	FF(ufi_set_stack_direction(rank, mask, stack_directions,
				   previous_directions));

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			struct dpu_t *dpu =
				DPU_GET_UNSAFE(rank, each_ci, each_dpu);
			if (dpu->enabled) {
				bool stack_if_effectively_up =
					previous_directions[each_ci] &
					(1 << each_dpu);
				if (stack_if_effectively_up != stack_is_up) {
					pr_warn("invalid stack mode (expected: %d, found: %d)",
						stack_is_up,
						stack_if_effectively_up);
					status = DPU_ERR_INTERNAL;
				}
			}
		}
	}

end:
	return status;
}

static uint32_t dpu_reset_internal_state(struct dpu_rank_t *rank)
{
	uint32_t status;
	dpuinstruction_t *iram_array[DPU_MAX_NR_CIS];
	dpuinstruction_t *internal_state_reset;
	const struct firmware *reset_program;
	iram_size_t internal_state_reset_size;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci;
	uint8_t nr_threads, each_thread;
	uint8_t mask = ALL_CIS, mask_all;
	uint8_t state[DPU_MAX_NR_CIS];
	bool running;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;
	nr_threads = rank->region->addr_translate.desc.dpu.nr_of_threads;

	/* DPU program should be located at /lib/firmware */
	if (request_firmware_direct(&reset_program, "internalStateReset.bin",
				    &rank->dev) < 0) {
		return DPU_ERR_NO_SUCH_FILE;
	}

	internal_state_reset = (dpuinstruction_t *)reset_program->data;
	internal_state_reset_size =
		reset_program->size / sizeof(dpuinstruction_t);

	for (each_ci = 0; each_ci < DPU_MAX_NR_CIS; ++each_ci) {
		iram_array[each_ci] = internal_state_reset;
	}

	FF(ufi_select_all(rank, &mask));
	FF(ufi_iram_write(rank, mask, iram_array, 0,
			  internal_state_reset_size));

	for (each_thread = 0; each_thread < nr_threads; ++each_thread) {
		/* Do not use dpu_thread_boot_safe_for_rank functions here as it would
         * increase reset duration for nothing.
         */
		FF(ufi_thread_boot(rank, mask, each_thread, NULL));
	}

	mask_all = (1 << nr_dpus_per_ci) - 1;
	do {
		FF(ufi_read_dpu_run(rank, mask, state));

		running = false;
		for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
			if (!CI_MASK_ON(mask, each_ci))
				continue;
			running = running || ((state[each_ci] & mask_all) != 0);
		}
	} while (running);

end:
	release_firmware(reset_program);
	return status;
}

static uint32_t dpu_init_groups(struct dpu_rank_t *rank,
				const bool *all_dpus_are_enabled_save,
				const dpu_selected_mask_t *enabled_dpus_save)
{
	uint32_t status = DPU_OK;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;
	uint8_t ci_mask = 0;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (all_dpus_are_enabled_save[each_ci]) {
			ci_mask |= CI_MASK_ONE(each_ci);
		} else if (enabled_dpus_save[each_ci]) {
			for (each_dpu = 0; each_dpu < nr_dpus_per_ci;
			     ++each_dpu) {
				uint8_t single_ci_mask = CI_MASK_ONE(each_ci);
				uint8_t group = ((enabled_dpus_save[each_ci] &
						  (1 << each_dpu)) != 0) ?
							      DPU_ENABLED_GROUP :
							      DPU_DISABLED_GROUP;
				FF(ufi_select_dpu(rank, &single_ci_mask,
						  each_dpu));
				FF(ufi_write_group(rank, single_ci_mask,
						   group));
			}
		}
	}

	if (ci_mask != 0) {
		FF(ufi_select_all(rank, &ci_mask));
		FF(ufi_write_group(rank, ci_mask, DPU_ENABLED_GROUP));
	}

	/* Set the rank context with the saved context */
	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		struct dpu_configuration_slice_info_t *ci_info =
			&rank->runtime.control_interface.slice_info[each_ci];

		ci_info->all_dpus_are_enabled =
			all_dpus_are_enabled_save[each_ci];
		ci_info->enabled_dpus = enabled_dpus_save[each_ci];

		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			DPU_GET_UNSAFE(rank, each_ci, each_dpu)->enabled =
				(ci_info->enabled_dpus & (1 << each_dpu)) != 0;
		}
	}

end:
	return status;
}

static void save_enabled_dpus(struct dpu_rank_t *rank,
			      bool *all_dpus_are_enabled_save,
			      dpu_selected_mask_t *enabled_dpus_save,
			      bool update_save)
{
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;
	dpu_selected_mask_t all_dpus_mask;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	all_dpus_mask = (1 << nr_dpus_per_ci) - 1;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		struct dpu_configuration_slice_info_t *ci_info =
			&rank->runtime.control_interface.slice_info[each_ci];

		all_dpus_are_enabled_save[each_ci] =
			!update_save ? ci_info->all_dpus_are_enabled :
					     all_dpus_are_enabled_save[each_ci] &
					       ci_info->all_dpus_are_enabled;
		enabled_dpus_save[each_ci] =
			!update_save ? ci_info->enabled_dpus :
					     enabled_dpus_save[each_ci] &
					       ci_info->enabled_dpus;

		/* Do not even talk to CIs where ALL dpus are disabled. */
		if (!ci_info->enabled_dpus)
			continue;

		ci_info->all_dpus_are_enabled = true;
		ci_info->enabled_dpus = all_dpus_mask;

		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			DPU_GET_UNSAFE(rank, each_ci, each_dpu)->enabled = true;
		}
	}
}

#define TIMEOUT_MUX_STATUS 100
#define CMD_GET_MUX_CTRL 0x02
static uint32_t dpu_check_wavegen_mux_status_for_dpu(struct dpu_rank_t *rank,
						     uint8_t dpu_id,
						     uint8_t *expected)
{
	uint32_t status;
	uint8_t dpu_dma_ctrl;
	uint8_t result_array[DPU_MAX_NR_CIS];
	uint32_t timeout = TIMEOUT_MUX_STATUS;
	uint8_t nr_cis, each_ci;
	uint8_t ci_mask = ALL_CIS;
	bool should_retry;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;

	FF(ufi_select_dpu_even_disabled(rank, &ci_mask, dpu_id));

	// Check Mux control through dma_rdat_ctrl of fetch1
	// 1 - Select WaveGen Read register @0xFF and set it @0x02  (mux and collision ctrl)
	// 2 - Flush readop2 (Pipeline to DMA cfg data path)
	// 3 - Read dpu_dma_ctrl
	FF(ufi_write_dma_ctrl(rank, ci_mask, 0xFF, CMD_GET_MUX_CTRL));
	FF(ufi_clear_dma_ctrl(rank, ci_mask));

	do {
		should_retry = false;

		FF(ufi_read_dma_ctrl(rank, ci_mask, result_array));
		for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
			if (!CI_MASK_ON(ci_mask, each_ci))
				continue;

			dpu_dma_ctrl = result_array[each_ci];

			// Expected 0x3 for DPU4 since it is the only one to be refreshed
			// Expected 0x0 for others DPU since no refresh has been issued
			pr_debug(
				"[DPU %hhu:%hhu] XMA Init = 0x%02x (expected = 0x%02x)",
				each_ci, dpu_id, dpu_dma_ctrl,
				expected[each_ci]);

			if ((dpu_dma_ctrl & 0x7F) != expected[each_ci]) {
				should_retry = true;
				break;
			}
		}

		timeout--;
	} while (timeout && should_retry); // Do not check Collision Error bit

	if (!timeout) {
		pr_warn("Timeout waiting for result to be correct");
		return DPU_ERR_TIMEOUT;
	}

end:
	return status;
}

static uint32_t dpu_check_wavegen_mux_status_for_rank(struct dpu_rank_t *rank,
						      uint8_t expected)
{
	uint32_t status;
	uint8_t dpu_dma_ctrl;
	uint8_t result_array[DPU_MAX_NR_CIS];
	uint32_t timeout;
	uint8_t ci_mask = ALL_CIS, mask;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;
	bool should_retry;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	/* ci_mask retains the real disabled CIs, whereas mask does not take
     * care of disabled dpus (and then CIs) since it should switch mux of
     * disabled dpus: but not in the case a CI is completely deactivated.
     */

	// Check Mux control through dma_rdat_ctrl of fetch1
	// 1 - Select WaveGen Read register @0xFF and set it @0x02  (mux and collision ctrl)
	// 2 - Flush readop2 (Pipeline to DMA cfg data path)
	// 3 - Read dpu_dma_ctrl
	ci_mask = ALL_CIS;
	FF(ufi_select_all(rank, &ci_mask));
	FF(ufi_write_dma_ctrl(rank, ci_mask, 0xFF, CMD_GET_MUX_CTRL));
	FF(ufi_clear_dma_ctrl(rank, ci_mask));

	for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
		timeout = TIMEOUT_MUX_STATUS;

		do {
			should_retry = false;

			mask = ALL_CIS;
			FF(ufi_select_dpu_even_disabled(rank, &mask, each_dpu));
			FF(ufi_read_dma_ctrl(rank, mask, result_array));

			for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
				if (!CI_MASK_ON(ci_mask, each_ci))
					continue;

				dpu_dma_ctrl = result_array[each_ci];

				if ((dpu_dma_ctrl & 0x7F) != expected) {
					pr_debug("DPU (%d, %d) failed", each_ci,
						 each_dpu);
					should_retry = true;
				}
			}

			timeout--;
		} while (timeout &&
			 should_retry); // Do not check Collision Error bit

		if (!timeout) {
			pr_warn("Timeout waiting for result to be correct");
			return DPU_ERR_TIMEOUT;
		}
	}

end:
	return status;
}

static uint32_t host_handle_access_for_dpu(struct dpu_rank_t *rank,
					   uint8_t dpu_id,
					   dpu_ci_bitfield_t ci_mux_pos)
{
	uint32_t status;
	uint8_t nr_cis, each_ci;
	uint8_t ci_mask = ALL_CIS, expected[DPU_MAX_NR_CIS];

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;

	FF(ufi_select_dpu_even_disabled(rank, &ci_mask, dpu_id));
	FF(ufi_set_mram_mux(rank, ci_mask, ci_mux_pos));

	for (each_ci = 0; each_ci < nr_cis; ++each_ci)
		expected[each_ci] = (ci_mux_pos & (1 << each_ci)) ?
						  WAVEGEN_MUX_HOST_EXPECTED :
						  WAVEGEN_MUX_DPU_EXPECTED;

	FF(dpu_check_wavegen_mux_status_for_dpu(rank, dpu_id, expected));

end:
	return status;
}

static uint32_t host_handle_access_for_rank(struct dpu_rank_t *rank,
					    bool set_mux_for_host)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;

	FF(ufi_select_all_even_disabled(rank, &mask));
	FF(ufi_set_mram_mux(rank, mask, set_mux_for_host ? 0xFF : 0x0));

	FF(dpu_check_wavegen_mux_status_for_rank(
		rank, set_mux_for_host ? WAVEGEN_MUX_HOST_EXPECTED :
					       WAVEGEN_MUX_DPU_EXPECTED));

end:
	return status;
}

uint32_t dpu_soft_reset(struct dpu_rank_t *rank,
			dpu_clock_division_t clock_division)
{
	uint32_t status;
	uint8_t mask = ALL_CIS;

	FF(ufi_select_cis(rank, &mask));
	ufi_soft_reset(rank, mask, clock_division, 0);

end:
	return status;
}

uint8_t dpu_get_host_mux_mram_state(struct dpu_rank_t *rank,
				    dpu_slice_id_t ci_id,
				    dpu_member_id_t dpu_id)
{
	return (rank->runtime.control_interface.slice_info[ci_id]
			.host_mux_mram_state >>
		dpu_id) &
	       1;
}

void dpu_set_host_mux_mram_state(struct dpu_rank_t *rank, dpu_slice_id_t ci_id,
				 dpu_member_id_t dpu_id, bool set)
{
	if (set)
		rank->runtime.control_interface.slice_info[ci_id]
			.host_mux_mram_state |= (1 << dpu_id);
	else
		rank->runtime.control_interface.slice_info[ci_id]
			.host_mux_mram_state &= ~(1 << dpu_id);
}

/*
 * Do care about pairs of dpus, simply switch mux according to mask for both:
 * the concept of "pair of dpus" must only be known here.
 */
uint32_t dpu_switch_mux_for_dpu_line(struct dpu_rank_t *rank, uint8_t dpu_id,
				     uint8_t mask)
{
	uint32_t status = DPU_OK;
	dpu_member_id_t dpu_pair_base_id = (dpu_member_id_t)(dpu_id & ~1);
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci;
	uint8_t mux_mram_mask;
	bool switch_base_line;
	bool switch_friend_line;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_ci = rank->region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;

	if (rank->runtime.run_context.nb_dpu_running > 0) {
		pr_warn("Host can not get access to the MRAM because %u DPU%s running.\n",
			rank->runtime.run_context.nb_dpu_running,
			rank->runtime.run_context.nb_dpu_running > 1 ? "s are" :
									     " is");
		status = DPU_ERR_MRAM_BUSY;
		goto end;
	}

	/* Check if switching mux is needed */
	switch_base_line = false;
	switch_friend_line = false;

	mux_mram_mask = 0;
	for (each_ci = 0; each_ci < nr_cis; ++each_ci)
		mux_mram_mask |= dpu_get_host_mux_mram_state(rank, each_ci,
							     dpu_pair_base_id)
				 << each_ci;

	switch_base_line = !(mux_mram_mask == mask);

	if ((dpu_pair_base_id + 1) < nr_dpus_per_ci) {
		mux_mram_mask = 0;
		for (each_ci = 0; each_ci < nr_cis; ++each_ci)
			mux_mram_mask |=
				dpu_get_host_mux_mram_state(
					rank, each_ci, dpu_pair_base_id + 1)
				<< each_ci;

		switch_friend_line = !(mux_mram_mask == mask);
	}

	if (!switch_base_line && !switch_friend_line) {
		pr_debug("Mux is in the right direction, nothing to do.\n");
		goto end;
	}

	/* Update mux state:
     * We must switch the mux at host side for the current dpu but switch the mux at dpu side
     * for other dpus of the same line.
     * We record the state before actually placing the mux in this state. If we did record the
     * state after setting the mux, we could be interrupted between the setting of the mux and
     * the recording of the state, and then the debugger would miss a mux state.
     */
	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if (switch_base_line)
			dpu_set_host_mux_mram_state(rank, each_ci,
						    dpu_pair_base_id,
						    mask & (1 << each_ci));
		if (switch_friend_line)
			dpu_set_host_mux_mram_state(rank, each_ci,
						    dpu_pair_base_id + 1,
						    mask & (1 << each_ci));
	}

	if (switch_base_line)
		FF(host_handle_access_for_dpu(rank, dpu_pair_base_id, mask));
	if (switch_friend_line)
		FF(host_handle_access_for_dpu(rank, dpu_pair_base_id + 1,
					      mask));

end:
	return status;
}

uint32_t dpu_switch_mux_for_rank(struct dpu_rank_t *rank, bool set_mux_for_host)
{
	uint32_t status = DPU_OK;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_cis;
	bool switch_mux;

	nr_cis = rank->region->addr_translate.desc.topology
			 .nr_of_control_interfaces;
	nr_dpus_per_cis = rank->region->addr_translate.desc.topology
				  .nr_of_dpus_per_control_interface;

	if (rank->runtime.run_context.nb_dpu_running > 0) {
		pr_warn("Host can not get access to the MRAM because %u DPU%s running.\n",
			rank->runtime.run_context.nb_dpu_running,
			rank->runtime.run_context.nb_dpu_running > 1 ? "s are" :
									     " is");
		status = DPU_ERR_MRAM_BUSY;
		goto end;
	}

	switch_mux = false;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		if ((set_mux_for_host &&
		     hweight8(
			     rank->runtime.control_interface.slice_info[each_ci]
				     .host_mux_mram_state) < nr_dpus_per_cis) ||
		    (!set_mux_for_host &&
		     rank->runtime.control_interface.slice_info[each_ci]
			     .host_mux_mram_state)) {
			pr_debug(
				"At least CI %d has mux in the wrong direction (0x%x), must switch rank.\n",
				each_ci,
				rank->runtime.control_interface
					.slice_info[each_ci]
					.host_mux_mram_state);
			switch_mux = true;
			break;
		}
	}

	if (!switch_mux) {
		pr_debug("Mux is in the right direction, nothing to do.");
		goto end;
	}

	/* We record the state before actually placing the mux in this state. If we
     * did record the state after setting the mux, we could be interrupted between
     * the setting of the mux and the recording of the state, and then the debugger
     * would miss a mux state.
     */
	for (each_ci = 0; each_ci < nr_cis; ++each_ci)
		rank->runtime.control_interface.slice_info[each_ci]
			.host_mux_mram_state =
			set_mux_for_host ? (1 << nr_dpus_per_cis) - 1 : 0x0;

	FF(host_handle_access_for_rank(rank, set_mux_for_host));

end:
	return status;
}

uint32_t dpu_reset_rank(struct dpu_rank_t *rank)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	bool all_dpus_are_enabled_save[DPU_MAX_NR_CIS];
	dpu_selected_mask_t enabled_dpus_save[DPU_MAX_NR_CIS];
	struct dpu_dma_config dma_config;
	struct dpu_wavegen_config wavegen_config;
	struct dpu_bit_config *bit_config;
	uint32_t status;

	bit_config = &tr->desc.dpu.pcb_transformation;

	// TODO: Here there is a problem, fck_frquency should be asked to the MCU
	// so that should happen after comm with MCU
	fetch_dma_and_wavegen_configs(tr->desc.timings.fck_frequency_in_mhz,
				      tr->desc.timings.clock_division,
				      REFRESH_MODE_VALUE, true, &dma_config,
				      &wavegen_config);

	/* All DPUs are enabled during the reset */
	save_enabled_dpus(rank, all_dpus_are_enabled_save, enabled_dpus_save,
			  false);

	FF(dpu_byte_order(rank));
	FF(dpu_soft_reset(rank, DPU_CLOCK_DIV4));
	FF(dpu_bit_config(rank, bit_config));
	FF(dpu_ci_shuffling_box_config(rank, bit_config));
	FF(dpu_soft_reset(rank, DPU_CLOCK_DIV4));
	FF(dpu_ci_shuffling_box_config(rank, bit_config));
	FF(dpu_identity(rank));
	FF(dpu_thermal_config(rank, tr->desc.timings.std_temperature));
	FF(dpu_carousel_config(rank, &tr->desc.timings.carousel));
	// TODO IRAM/WRAM repair.
	FF(dpu_iram_repair_config(rank));
	FF(dpu_wram_repair_config(rank));
	FF(dpu_dma_config(rank, &dma_config));
	FF(dpu_dma_shuffling_box_config(rank, bit_config));
	FF(dpu_wavegen_config(rank, &wavegen_config));
	FF(dpu_clear_debug(rank));
	FF(dpu_clear_run_bits(rank));
	FF(dpu_set_pc_mode(rank, DPU_PC_MODE_16));
	FF(dpu_set_stack_direction(rank, true));
	FF(dpu_reset_internal_state(rank));
	FF(dpu_switch_mux_for_rank(rank, true));
	FF(dpu_init_groups(rank, all_dpus_are_enabled_save, enabled_dpus_save));

end:
	return status;
}
EXPORT_SYMBOL(dpu_reset_rank);

uint32_t dpu_set_chip_id(struct dpu_rank_t *rank)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	struct dpu_bit_config *bit_config;
	uint32_t status;

	bit_config = &tr->desc.dpu.pcb_transformation;

	FF(dpu_byte_order(rank));
	FF(dpu_soft_reset(rank, DPU_CLOCK_DIV4));
	FF(dpu_bit_config(rank, bit_config));
	FF(dpu_ci_shuffling_box_config(rank, bit_config));
	FF(dpu_soft_reset(rank, DPU_CLOCK_DIV4));
	FF(dpu_identity(rank));

end:
	return status;
}
