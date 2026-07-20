/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/slab.h>

#include <dpu_region.h>
#include <dpu_control_interface.h>
#include <dpu_mcu_ci_commands.h>
#include <dpu_mcu_ci_protocol.h>
#include <dpu_vpd_structures.h>
#include <dpu_rank_mcu.h>

/* dpu_rank attributes */
static ssize_t is_owned_show(struct device *dev, struct device_attribute *attr,
			     char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%d\n", rank->owner.is_owned);
}

static ssize_t usage_count_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%d\n", rank->owner.usage_count);
}

static ssize_t mcu_version_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%s\n", rank->mcu_version);
}

static ssize_t fck_frequency_show(struct device *dev,
				  struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;

	return sprintf(buf, "%u\n", tr->desc.timings.fck_frequency_in_mhz);
}

static ssize_t clock_division_show(struct device *dev,
				   struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;

	return sprintf(buf, "%u\n", tr->desc.timings.clock_division);
}

static ssize_t rank_index_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%u\n", rank->rank_index);
}

static ssize_t rank_count_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%u\n", rank->rank_count);
}

static ssize_t part_number_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%s\n", rank->part_number);
}

static ssize_t serial_number_show(struct device *dev,
				  struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%s\n", rank->serial_number);
}

static ssize_t signal_led_store(struct device *dev,
				struct device_attribute *attr, const char *buf,
				size_t len)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct ec_params_signal signal;
	int ret;

	ret = kstrtou8(buf, 10, &signal.on_off);
	if (ret)
		return ret;

	ret = dpu_control_interface_mcu_command(
		rank, EC_CMD_DIMM_SIGNAL, 0, &signal, sizeof(signal), NULL, 0);
	if (ret < 0) {
		dev_warn(&rank->dev, "fail to send signal command to MCU\n");
		return ret;
	}

	return len;
}

static ssize_t channel_id_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%d\n", rank->channel_id);
}

static ssize_t nb_ci_show(struct device *dev, struct device_attribute *attr,
			  char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint8_t nb_ci;

	nb_ci = tr->desc.topology.nr_of_control_interfaces;

	return sprintf(buf, "%d\n", nb_ci);
}

static ssize_t nb_dpus_per_ci_show(struct device *dev,
				   struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint8_t nb_dpus_per_ci;

	nb_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;

	return sprintf(buf, "%d\n", nb_dpus_per_ci);
}

static ssize_t mram_size_show(struct device *dev, struct device_attribute *attr,
			      char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint32_t mram_size;

	mram_size = tr->desc.memories.mram_size;

	return sprintf(buf, "%d\n", mram_size);
}

static ssize_t dpu_chip_id_show(struct device *dev,
				struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;

	return sprintf(buf, "%d\n", tr->desc.signature.chip_id);
}

static ssize_t backend_id_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint8_t backend_id;

	backend_id = (uint8_t)tr->backend_id;

	return sprintf(buf, "%hhu\n", backend_id);
}

static ssize_t mode_show(struct device *dev, struct device_attribute *attr,
			 char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%d\n", rank->region->mode);
}

static ssize_t mode_store(struct device *dev, struct device_attribute *attr,
			  const char *buf, size_t len)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region *region = rank->region;
	int ret;
	uint8_t tmp;

	ret = kstrtou8(buf, 10, &tmp);
	if (ret)
		return ret;

	if (tmp != DPU_REGION_MODE_PERF && tmp != DPU_REGION_MODE_SAFE) {
		dev_err(dev, "mode: value %u is undefined\n", tmp);
		return -EINVAL;
	}

	dpu_region_lock(region);

	/* In perf mode, one can access the rank through the dpu_rank device too:
	 * switch from safe to perf is possible iff no rank is allocated.
	 * switch from perf to safe is possible iff see below
	 */
	if (region->mode == DPU_REGION_MODE_SAFE &&
	    tmp == DPU_REGION_MODE_PERF) {
		if (rank->owner.is_owned) {
			dev_err(dev, "dpu_rank is allocated in safe "
				     "mode, can't switch to perf mode.\n");
			dpu_region_unlock(region);
			return -EBUSY;
		}
	} else if (region->mode == DPU_REGION_MODE_PERF &&
		   tmp == DPU_REGION_MODE_SAFE) {
		// TODO
		// 1/ find a way to get a refcount on dax device
		// 2/ find a way to deny access to dax mmap...
	}

	region->mode = tmp;

	dpu_region_unlock(region);

	return len;
}

static ssize_t debug_mode_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%hhu\n", rank->debug_mode);
}

static ssize_t debug_mode_store(struct device *dev,
				struct device_attribute *attr, const char *buf,
				size_t len)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region *region = rank->region;
	int ret;
	uint8_t tmp;

	if (!capable(CAP_SYS_ADMIN))
		return -EPERM;

	ret = kstrtou8(buf, 10, &tmp);
	if (ret)
		return ret;

	dpu_region_lock(region);

	rank->debug_mode = tmp;

	dpu_region_unlock(region);

	return len;
}

static ssize_t rank_id_show(struct device *dev, struct device_attribute *attr,
			    char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);

	return sprintf(buf, "%d\n", rank->id);
}

static ssize_t capabilities_show(struct device *dev,
				 struct device_attribute *attr, char *buf)
{
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint64_t capabilities;

	capabilities = tr->capabilities;

	return sprintf(buf, "%#llx\n", capabilities);
}

static ssize_t byte_order_show(struct device *dev,
			       struct device_attribute *attr, char *buf)
{
#define BYTE_ORDER_STR_LEN strlen("0xFFFFFFFFFFFFFFFF ")
	struct dpu_rank_t *rank = dev_get_drvdata(dev);
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
	uint8_t nr_cis, each_ci;
	ssize_t ret_size = 0;

	nr_cis = tr->desc.topology.nr_of_control_interfaces;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		/* The trailing space is used for userspace simplicity */
		ret_size += sprintf(
			buf + each_ci * BYTE_ORDER_STR_LEN, "0x%016llx ",
			rank->runtime.control_interface.slice_info[each_ci]
				.byte_order);
	}

	ret_size++;
	strcat(buf, "\n");

	return ret_size;
}

static inline struct dpu_rank_t *dev_to_rank(struct device *dev)
{
	return container_of(dev, struct dpu_rank_t, dev);
}

static inline struct dpu_rank_t *kobj_to_rank(struct kobject *kobj)
{
	return dev_to_rank(container_of(kobj, struct device, kobj));
}

/* Note that dimm_vpd_read returns the VPD of the DIMM,
 * not the rank.
 */
static ssize_t dimm_vpd_read(struct file *filp, struct kobject *kobj,
			     struct bin_attribute *bin_attr, char *buf,
			     loff_t off, size_t count)
{
	struct dpu_rank_t *rank = kobj_to_rank(kobj);
	uint16_t repair_count;
	size_t space_left;
	size_t size_entries;
	int ret;

	if (count != 1 || off != 0)
		return -EFAULT;

	/* Update VPD value from MCU */
	ret = dpu_rank_mcu_get_vpd(rank);

	memcpy(buf, &rank->vpd.vpd_header, sizeof(struct dpu_vpd_header));

	/*
	 * The VPD segment might contain garbage or we might have failed to read the
	 * MCU flash. If so, do not expose more than the VPD header.
	 */
	repair_count = rank->vpd.vpd_header.repair_count;
	if ((repair_count != (uint16_t)VPD_UNDEFINED_REPAIR_COUNT) &&
	    (ret == 0)) {
		size_entries =
			repair_count * sizeof(struct dpu_vpd_repair_entry);
		/* Total size must not exceed PAGE_SIZE */
		space_left = PAGE_SIZE - sizeof(struct dpu_vpd_header);
		size_entries =
			(size_entries > space_left) ? space_left : size_entries;
		memcpy(buf + sizeof(struct dpu_vpd_header),
		       rank->vpd.repair_entries, size_entries);

		return (ssize_t)(sizeof(struct dpu_vpd_header) + size_entries);
	}

	return (ssize_t)(sizeof(struct dpu_vpd_header));
}

static DEVICE_ATTR_RO(is_owned);
static DEVICE_ATTR_RO(usage_count);
static DEVICE_ATTR_RO(mcu_version);
static DEVICE_ATTR_RO(fck_frequency);
static DEVICE_ATTR_RO(clock_division);
static DEVICE_ATTR_RO(rank_index);
static DEVICE_ATTR_RO(rank_count);
static DEVICE_ATTR_RO(part_number);
static DEVICE_ATTR_RO(serial_number);
static DEVICE_ATTR_WO(signal_led);
static DEVICE_ATTR_RO(channel_id);
static DEVICE_ATTR_RO(nb_ci);
static DEVICE_ATTR_RO(nb_dpus_per_ci);
static DEVICE_ATTR_RO(mram_size);
static DEVICE_ATTR_RO(dpu_chip_id);
static DEVICE_ATTR_RO(backend_id);
static DEVICE_ATTR_RW(mode);
static DEVICE_ATTR_RW(debug_mode);
static DEVICE_ATTR_RO(rank_id);
static DEVICE_ATTR_RO(capabilities);
static DEVICE_ATTR_RO(byte_order);

static BIN_ATTR_RO(dimm_vpd, 1);

static struct attribute *dpu_rank_attrs[] = {
	&dev_attr_is_owned.attr,       &dev_attr_usage_count.attr,
	&dev_attr_mcu_version.attr,    &dev_attr_fck_frequency.attr,
	&dev_attr_clock_division.attr, &dev_attr_rank_index.attr,
	&dev_attr_rank_count.attr,     &dev_attr_part_number.attr,
	&dev_attr_serial_number.attr,  &dev_attr_signal_led.attr,
	&dev_attr_channel_id.attr,     &dev_attr_nb_ci.attr,
	&dev_attr_nb_dpus_per_ci.attr, &dev_attr_mram_size.attr,
	&dev_attr_dpu_chip_id.attr,    &dev_attr_backend_id.attr,
	&dev_attr_mode.attr,	       &dev_attr_debug_mode.attr,
	&dev_attr_rank_id.attr,	       &dev_attr_capabilities.attr,
	&dev_attr_byte_order.attr,     NULL,
};

static struct bin_attribute *dpu_rank_bin_attrs[] = {
	&bin_attr_dimm_vpd,
	NULL,
};

static struct attribute_group dpu_rank_attrs_group = {
	.attrs = dpu_rank_attrs,
	.bin_attrs = dpu_rank_bin_attrs,
};

const struct attribute_group *dpu_rank_attrs_groups[] = { &dpu_rank_attrs_group,
							  NULL };
