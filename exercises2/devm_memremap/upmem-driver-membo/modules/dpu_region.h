/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#ifndef DPU_REGION_INCLUDE_H
#define DPU_REGION_INCLUDE_H

#include <linux/cdev.h>
#include <linux/list.h>
#include <linux/memremap.h>
#include <linux/mutex.h>
#include <linux/idr.h>
#include <linux/version.h>
#include <linux/device.h>

#include <dpu_rank.h>
#include <dpu_region_address_translation.h>
#include <dpu_region_constants.h>
#include <dpu_vpd_structures.h>

#include <dpu_fpga_kc705_device.h>

#define DPU_REGION_NAME "dpu_region"
#define DPU_REGION_PATH DPU_REGION_NAME "%d"

struct dpu_dax_device {
	struct percpu_ref ref;
	struct dev_pagemap pgmap;
	struct completion cmp;
};

struct dpu_region {
	struct dpu_region_address_translation addr_translate;
	uint8_t mode;

	struct mutex lock;

	/* Memory driver */
	struct dpu_dax_device dpu_dax_dev;
	struct cdev cdev_dax;
	struct device dev_dax;
	dev_t devt_dax;
	void *base; /* linear address corresponding to the region resource */
	uint64_t size;
	void *mc_flush_address;

	/* Pci fpga kc705 driver */
	struct pci_device_fpga dpu_fpga_kc705_dev;
	uint8_t activate_ila;
	uint8_t activate_filtering_ila;
	uint8_t activate_mram_bypass;
	uint8_t spi_mode_enabled;
	uint32_t mram_refresh_emulation_period;
	struct dentry *iladump;
	struct dentry *dpu_debugfs;

	/* Pci fpga aws driver */
	struct xdma_dev *dpu_fpga_aws_dev;

	struct dpu_rank_t {
		struct list_head list;

		struct dpu_region *region;

		struct cdev cdev;
		struct device dev;

		struct dpu_rank_owner owner;
		struct dpu_runtime_state_t runtime;
		struct dpu_bit_config bit_config;

		struct dpu_t *dpus;

		uint8_t id;
		uint8_t channel_id;
        int nid;
        atomic_t nr_ltb_sections;
        bool is_reserved;
		uint8_t slot_index;

		uint8_t debug_mode;

		uint64_t control_interface[DPU_MAX_NR_CIS];
		uint64_t data[DPU_MAX_NR_CIS];

		/* Preallocates a huge array of struct page *
		* pointers for get_user_pages and xferp.
		*/
		struct page **xfer_dpu_page_array;
		struct xfer_page xfer_pg[DPU_MAX_NR_DPUS];

		/* Information requested from the MCU */
		uint8_t rank_index;
		uint8_t rank_count;
		char mcu_version[DPU_RANK_MCU_VERSION_LEN];
		char part_number[DPU_DIMM_PART_NUMBER_LEN]; /* e.g UPMEM-E19 */
		char serial_number[DPU_DIMM_SERIAL_NUMBER_LEN];
		struct dpu_vpd vpd;
	} rank;
};

extern struct ida dpu_region_ida;

void dpu_region_lock(struct dpu_region *region);
void dpu_region_unlock(struct dpu_region *region);

int dpu_region_dev_probe(void);
int dpu_region_srat_probe(void);
int dpu_region_srat_get_pxm(u64 base_region_addr);

int dpu_region_mem_add(u64 addr, u64 size, int index);

#endif /* DPU_REGION_INCLUDE_H */
