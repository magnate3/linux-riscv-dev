/* SPDX-License-Identifier: GPL-2.0 */
/*
 * Copyright 2020 - UPMEM
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/dmi.h>
#include <linux/sizes.h>
#include <linux/slab.h>

#include "dpu_region.h"

/* Memory Device - Type 17 of SMBIOS spec */
struct memdev_dmi_entry {
	u8 type;
	u8 length;
	u16 handle;
	u16 phys_mem_array_handle;
	u16 mem_err_info_handle;
	u16 total_width;
	u16 data_width;
	u16 size;
	u8 form_factor;
	u8 device_set;
	u8 device_locator;
	u8 bank_locator;
	u8 memory_type;
	u16 type_detail;
	u16 speed;
	u8 manufacturer;
	u8 serial_number;
	u8 asset_tag;
	u8 part_number;
	u8 attributes;
	u32 extended_size;
	u16 conf_mem_clk_speed;
} __attribute__((__packed__));

struct dimm_phys_pos {
	struct list_head list;
	u32 serial_number;
	unsigned cpu;
	unsigned index;
	unsigned channel;
	char slot;
};
static LIST_HEAD(dimm_positions);

static const char *dmi_string(const struct dmi_header *dh, u8 idx)
{
	const u8 *ptr = ((u8 *)dh) + dh->length;

	if (!idx)
		return "";

	while (--idx > 0 && *ptr)
		ptr += strlen(ptr) + 1;

	if (*ptr != '\0')
		return ptr;

	return "";
}

static void dmi_entry_detect_pim(const struct dmi_header *dh, void *priv)
{
	struct memdev_dmi_entry *md = (struct memdev_dmi_entry *)dh;
	const char *loc, *sn;
	u32 serial;
	struct dimm_phys_pos *pos;

	switch (dh->type) {
	case DMI_ENTRY_MEM_DEVICE:
		if (md->memory_type != 0x1A /* DDR4 */)
			break;
		if (strcmp(dmi_string(dh, md->manufacturer), "UPMEM"))
			break;
		sn = dmi_string(dh, md->serial_number);
		if (sscanf(sn, "%08x", &serial) != 1) {
			pr_warn("Unknown DIMM S/N format: %s\n", sn);
			break;
		}
		loc = dmi_string(dh, md->device_locator);

		pos = kzalloc(sizeof(*pos), GFP_KERNEL);
		if (!pos)
			break;
		pos->serial_number = serial;

		if (sscanf(loc, "CPU%u_DIMM_%c%u", &pos->cpu, &pos->slot,
			   &pos->index) == 3)
			pos->channel = pos->slot - 'A';
		else
			pr_warn("Unknown slot format: %s\n", loc);

		list_add(&pos->list, &dimm_positions);
		break;
	}
}

void dpu_rank_dmi_find_channel(struct dpu_rank_t *rank)
{
	const char *sn = rank->serial_number;
	const char *pn = rank->part_number;
	u32 serial;
	struct list_head *pos;

	if (sscanf(sn, "%08x", &serial) != 1) {
		dev_warn(&rank->dev, "Unknown S/N format: %s\n", sn);
		return;
	}
	list_for_each (pos, &dimm_positions) {
		struct dimm_phys_pos *dimm =
			list_entry(pos, struct dimm_phys_pos, list);
		if (serial != dimm->serial_number)
			continue;

		rank->channel_id = dimm->channel;
		rank->slot_index = dimm->index;
		dev_info(&rank->dev, "DIMM P/N %.6s slot %c%d channel %d\n", pn,
			 dimm->slot, dimm->index, rank->channel_id);
	}
}

void dpu_rank_dmi_init(void)
{
	dmi_walk(dmi_entry_detect_pim, NULL);
}

void dpu_rank_dmi_exit(void)
{
	struct list_head *pos, *n;
	list_for_each_safe (pos, n, &dimm_positions) {
		struct dimm_phys_pos *dimm =
			list_entry(pos, struct dimm_phys_pos, list);
		list_del(pos);
		kfree(dimm);
	}
}
