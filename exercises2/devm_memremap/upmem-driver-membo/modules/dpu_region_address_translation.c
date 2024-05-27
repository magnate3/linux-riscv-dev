/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/dmi.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/pci.h>
#include <linux/version.h>

#include "dpu_pci_ids.h"
#include "dpu_region_address_translation.h"

#ifdef CONFIG_DMI
/* Translation backends detected using the DMI system attributes */
static const struct dmi_system_id pim_platform_dmi_table[] = {
	{
		.ident = "Intel Xeon Scalable Platform",
		.matches =
			{
				DMI_MATCH(DMI_SYS_VENDOR, "UPMEM"),
#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 12, 0)
				DMI_MATCH(DMI_PRODUCT_FAMILY, "Xeon_Scalable"),
#endif /* LINUX_VERSION_CODE >= KERNEL_VERSION(4, 12, 0) */
			},
		.driver_data = (void *)DPU_BACKEND_XEON_SP,
	},
};
#endif /* CONFIG_DMI */
/* Translation backends detected using the device-tree */
static const struct of_device_id pim_platform_of_table[] = {
	{
		.compatible = "ibm,powernv",
		.data = (void *)DPU_BACKEND_POWER9,
	},
};

enum backend dpu_get_translation_config(struct device *dev,
					unsigned int default_backend)
{
	struct device_node *np;
	const struct of_device_id *of_id;
#ifdef CONFIG_DMI
	const struct dmi_system_id *dmi_id;
	dmi_id = dmi_first_match(pim_platform_dmi_table);
	if (dmi_id) {
		unsigned int backend = (uintptr_t)dmi_id->driver_data;
		dev_info(dev, "Translation backend: DMI matched '%s' (%d)\n",
			 dmi_id->ident, backend);
		return (enum backend)backend;
	}
#endif /* CONFIG_DMI */
	np = of_find_matching_node_and_match(NULL, pim_platform_of_table,
					     &of_id);
	if (np) {
		unsigned int backend = (uintptr_t)of_id->data;
		dev_info(dev, "Translation backend: OF matched '%s' (%d)\n",
			 of_id->compatible, backend);
		return (enum backend)backend;
	}

	return (enum backend)default_backend;
}

void dpu_region_set_address_translation(
	struct dpu_region_address_translation *addr_translate,
	enum backend backend, const struct pci_dev *pci_dev)
{
	struct dpu_region_address_translation *tr;

	switch (backend) {
	case DPU_BACKEND_FPGA_KC705:
		if (pci_dev->subsystem_vendor == KC705_SUBVENDOR_ID_1DPU) {
			pr_info("dpu_region: Using fpga kc705_1dpu config\n");
			tr = &fpga_kc705_translate_1dpu;
		} else if (pci_dev->subsystem_vendor ==
			   KC705_SUBVENDOR_ID_8DPU) {
			pr_info("dpu_region: Using fpga kc705_8dpu config\n");
			tr = &fpga_kc705_translate_8dpu;
		} else {
			pr_warn("Unknown PCI subsystem vendor ID\n");
			return;
		}

		break;
	case DPU_BACKEND_FPGA_AWS:
		if (pci_dev->device == BITTWARE_FPGA_PCIE3_DEVICE_ID) {
			pr_info("dpu_region: Using fpga bittware config\n");
			tr = &fpga_bittware_translate;
		} else if (pci_dev->device == AWS_DPU_DEVICE_ID) {
			pr_info("dpu_region: Using fpga aws config\n");
			tr = &fpga_aws_translate;
		} else {
			pr_warn("Unknown PCI Device ID\n");
			return;
		}

		break;
#ifdef CONFIG_X86_64
	case DPU_BACKEND_XEON_SP:
		pr_info("dpu_region: Using xeon sp config\n");
		tr = &xeon_sp_translate;

		break;
#endif
#ifdef CONFIG_PPC64
	case DPU_BACKEND_POWER9:
		pr_info("dpu_region: Using power9 config\n");
		tr = &power9_translate;

		break;
#endif
	default:
		pr_err("dpu_region: Unknown backend\n");
		return;
	}

	memcpy(addr_translate, tr,
	       sizeof(struct dpu_region_address_translation));
}

MODULE_ALIAS("dmi:*:svnUPMEM:*");
