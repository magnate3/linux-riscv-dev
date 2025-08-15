/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/pci.h>
#include <linux/platform_device.h>
#include <linux/cdev.h>
#include <linux/fs.h>
#include <linux/debugfs.h>
#include <linux/ioctl.h>
#include <linux/mm.h>
#include <linux/mm_types.h>
#include <linux/mutex.h>
#include <linux/slab.h>
#include <linux/version.h>

#include <dpu_fpga_kc705_device.h>
#include <dpu_fpga_kc705_dma_op.h>
#include <dpu_fpga_kc705_fs.h>

#include <dpu_fpga_aws_libxdma_api.h>
#include <dpu_fpga_aws_libxdma.h>

#include <dpu_region.h>
#include <dpu_region_address_translation.h>
#include <dpu_rank.h>
#include <dpu_rank_mcu.h>
#include <dpu_dax.h>
#include <dpu_config.h>
#include <dpu_pci_ids.h>
#include <dpu_utils.h>
#include <dpu_membo.h>

static unsigned int default_backend = 0;

DEFINE_IDA(dpu_region_ida);
extern struct class *dpu_membo_class;
extern bool membo_initialized;

void dpu_region_lock(struct dpu_region *region)
{
	mutex_lock(&region->lock);
}

void dpu_region_unlock(struct dpu_region *region)
{
	mutex_unlock(&region->lock);
}

struct dpu_region_pdev {
	struct list_head list;
	struct platform_device *pdev;
};
static LIST_HEAD(region_pdev);

int dpu_region_mem_add(u64 addr, u64 size, int index)
{
	struct platform_device *pdev;
	struct dpu_region_pdev *reg;
	struct resource res = DEFINE_RES_MEM(addr, size);

	pr_info("MEM DPU region%d: %016llx->%016llx %lld GB\n", index, addr,
		addr + size, size / SZ_1G);
	pdev = platform_device_register_simple("dpu_region_mem", index, &res,
					       1);
	if (IS_ERR(pdev)) {
		pr_warn("Cannot register region%d (%016llx->%016llx)\n", index,
			addr, addr + size);
		return 0;
	}
	reg = kzalloc(sizeof(*reg), GFP_KERNEL);
	if (ZERO_OR_NULL_PTR(reg)) {
		platform_device_unregister(pdev);
		return 0;
	}
	reg->pdev = pdev;
	list_add(&reg->list, &region_pdev);
	return 1;
}

void dpu_region_mem_exit(void)
{
	struct list_head *pos, *n;
	list_for_each_safe (pos, n, &region_pdev) {
		struct dpu_region_pdev *reg =
			list_entry(pos, struct dpu_region_pdev, list);
		/* remove from the list BEFORE the 'devm' structure is freed */
		list_del(pos);
		platform_device_unregister(reg->pdev);
		kfree(reg);
	}
}

static int dpu_region_mem_probe(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct dpu_region *region;
	int ret;

	dev_dbg(dev, "device probed\n");

	region = devm_kzalloc(dev, sizeof(struct dpu_region), GFP_KERNEL);
	if (!region) {
		ret = -ENOMEM;
		goto err;
	}

	/* 1/ Init dax device */
	ret = dpu_dax_init_device(pdev, region);
	if (ret)
		goto err;

	/* 2/ Init dpu_rank device associated to that dax device */
	dpu_region_set_address_translation(
		&region->addr_translate,
		dpu_get_translation_config(dev, default_backend), NULL);

	mutex_init(&region->lock);
	region->addr_translate.private = region;
	dev_set_drvdata(&pdev->dev, region);

	ret = dpu_rank_init_device(dev, region, 0);
	if (ret) {
		dev_dbg(dev, "cannot create dpu rank device\n");
		goto free_dev_dax;
	}

	ret = dpu_set_chip_id(&region->rank);
	if (ret) {
		dev_err(dev, "cannot get region chip id\n");
		ret = -1;
		goto destroy_rank_device;
	}

	/* MCU should be requested after doing the byte/bit ordering */
	ret = dpu_rank_mcu_probe(&region->rank);
	if (ret) {
		dev_err(dev, "cannot probe the mcu\n");
		ret = -1;
		goto destroy_rank_device;
	}

	/*
	 * try to find the memory controller channel associated with the rank,
	 * by correlating the DMI table information with the DIMM serial number.
	 * If there is no DMI support, it will silently do nothing.
	 */
	dpu_rank_dmi_find_channel(&region->rank);

	dev_dbg(dev, "device loaded.\n");

	return 0;

destroy_rank_device:
	dpu_rank_release_device(region);
free_dev_dax:
	dpu_dax_release_device(region);
err:
	return ret;
}

static int dpu_region_mem_remove(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct dpu_region *region;

	dev_dbg(dev, "removing dpu region device (and its rank)\n");

	region = dev_get_drvdata(dev);
	if (!region)
		return -EINVAL;

	/* Release DAX device */
	dpu_dax_release_device(region);

	/* Release dpu_rank device */
	dpu_rank_release_device(region);

	return 0;
}

/* Memory driver */
static struct platform_driver dpu_region_mem_driver = {
	.driver = { .name = DPU_REGION_NAME "_mem", .owner = THIS_MODULE },
	.probe = dpu_region_mem_probe,
	.remove = dpu_region_mem_remove,
};

/* Default value for bank_map structure */
static const struct bank_map init_banks[] = {
	{ "dma", 0, 0x0, 0, NULL, 0 },
	{ "bypass", 2, 0x0, 0, NULL, 0 },
};

static int dpu_region_init_fpga_kc705(struct pci_dev *pci_dev,
				      struct dpu_region *region)
{
	struct pci_device_fpga *pdev;
	resource_size_t bar_start;
	resource_size_t bar_len;
	int bar;
	int err, i;

	/* Init pci device */

	// Set Bus Master Enable (BME) bit
	pci_set_master(pci_dev);

	err = pci_enable_device(pci_dev);
	if (err) {
		pr_err("[PCI-FPGA] Could not enable PCI device: %d\n", err);
		goto error;
	}

	// Set DMA Mask
	err = pci_set_dma_mask(pci_dev, 0x7FFFFFFFFFFFFFFF);
	if (err) {
		pr_err("[PCI-FPGA] Init: DMA not supported\n");
		goto error;
	}
	pci_set_consistent_dma_mask(pci_dev, 0x7FFFFFFFFFFFFFFF);

	err = pci_request_regions(pci_dev, "dpu_region");
	if (err) {
		pr_err("[PCI-FPGA] Failed to setup DPU-FPGA device: %d\n", err);
		goto error_request_regions_failed;
	}

	/* Add the new device to the list of pci device fpga */
	pdev = &region->dpu_fpga_kc705_dev;

	for (i = 0; i < BANKS_NUM; ++i) {
		memcpy(&pdev->banks[i], &init_banks[i],
		       sizeof(struct bank_map));

		bar = pdev->banks[i].bar;
		bar_start = pci_resource_start(pci_dev, bar);
		bar_len = pci_resource_len(pci_dev, bar);

		pdev->banks[i].phys = bar_start;
		pdev->banks[i].len = bar_len;
		pdev->banks[i].addr = pci_iomap(pci_dev, bar, bar_len);
	}

	region->rank.id = ida_simple_get(&dpu_region_ida, 0, 0, GFP_KERNEL);

	dev_set_drvdata(&pci_dev->dev, region);
	pdev->dev = pci_dev;

	region->activate_ila = 0;
	region->activate_filtering_ila = 0;
	region->activate_mram_bypass = 0;
	region->mram_refresh_emulation_period = 0;
	region->spi_mode_enabled = 0;

	err = xpdma_init(pdev);
	if (err)
		goto error_pdev;

	return 0;

error_pdev:
	pci_release_regions(pci_dev);
error_request_regions_failed:
	pci_disable_device(pci_dev);
error:
	return err;
}

static int dpu_region_fpga_kc705_probe(struct pci_dev *pci_dev,
				       const struct pci_device_id *id)
{
	struct device *dev = &pci_dev->dev;
	struct dpu_region *region;
	int ret;

	dev_dbg(dev, "fpga kc705 device probed\n");

	region = kzalloc(sizeof(struct dpu_region), GFP_KERNEL);
	if (!region) {
		ret = -ENOMEM;
		goto err;
	}

	/* 1/ Init fpga kc705 device */
	ret = dpu_region_init_fpga_kc705(pci_dev, region);
	if (ret)
		goto free_region;

	/* 2/ Init dpu_rank device associated to that kc705 device */
	dpu_region_set_address_translation(&region->addr_translate,
					   DPU_BACKEND_FPGA_KC705, pci_dev);

	mutex_init(&region->lock);
	/* Finally, store pdev also into dpu_region_address_translation
	 * private member since the current implementation requires this
	 * structure to work, kind of a hack yes.
	 */
	region->addr_translate.private = &region->dpu_fpga_kc705_dev;
	dev_set_drvdata(dev, region);

	ret = dpu_rank_init_device(dev, region, false);
	if (ret) {
		dev_err(dev, "cannot create dpu rank device\n");
		goto free_fpga;
	}

	ret = dpu_set_chip_id(&region->rank);
	if (ret) {
		dev_err(dev, "cannot get region chip id\n");
		ret = -1;
		goto destroy_rank_device;
	}

	/* Create debugfs and sysfs entries */
	ret = dpu_debugfs_sysfs_fpga_kc705_create(&region->dpu_fpga_kc705_dev,
						  region);
	if (ret)
		goto destroy_rank_device;

	dev_dbg(dev, "device loaded.\n");

	return 0;

destroy_rank_device:
	dpu_rank_release_device(region);
free_fpga:
	pci_release_regions(pci_dev);
	pci_disable_device(pci_dev);
	xpdma_exit(&region->dpu_fpga_kc705_dev);
free_region:
	kfree(region);
err:
	return ret;
}

static void dpu_region_fpga_kc705_remove(struct pci_dev *pci_dev)
{
	struct device *dev = &pci_dev->dev;
	struct dpu_region *region;

	dev_dbg(dev, "removing dpu region device (and its rank)\n");

	pci_release_regions(pci_dev);
	pci_disable_device(pci_dev);

	region = dev_get_drvdata(dev);
	if (!region)
		return;

	iounmap(region->dpu_fpga_kc705_dev.banks[0].addr);

	xpdma_exit(region->addr_translate.private);

	dpu_debugfs_sysfs_fpga_kc705_destroy(&region->dpu_fpga_kc705_dev,
					     region);

	/* Release dpu_rank device */
	dpu_rank_release_device(region);
	kfree(region);
}

static struct pci_device_id dpu_region_fpga_kc705_ids[] = {
	{
		PCI_DEVICE(KC705_VENDOR_ID, KC705_DEVICE_ID),
	},
	{
		0,
	}
};

MODULE_DEVICE_TABLE(pci, dpu_region_fpga_kc705_ids);

static struct pci_driver dpu_region_fpga_kc705_driver = {
	.name = DPU_REGION_NAME "_fpga_kc705",
	.id_table = dpu_region_fpga_kc705_ids,
	.probe = dpu_region_fpga_kc705_probe,
	.remove = dpu_region_fpga_kc705_remove,
};

static int dpu_region_init_fpga_aws(struct pci_dev *pci_dev,
				    kernel_ulong_t mem_bar_index,
				    struct dpu_region *region)
{
	struct xdma_dev *xdev;
	char *xdev_name;
	int user_max, c2h_channel_max, h2c_channel_max;
	int err;

	/* Init pci device */
	user_max = MAX_USER_IRQ;
	h2c_channel_max = XDMA_CHANNEL_NUM_MAX;
	c2h_channel_max = XDMA_CHANNEL_NUM_MAX;

	xdev_name = devm_kzalloc(&pci_dev->dev, 32, GFP_KERNEL);
	if (!xdev_name)
		return -ENOMEM;

	region->rank.id = ida_simple_get(&dpu_region_ida, 0, 0, GFP_KERNEL);
	sprintf(xdev_name, "dpu_rank_xdev%d", region->rank.id);

	xdev = xdma_device_open(xdev_name, pci_dev, &user_max, &h2c_channel_max,
				&c2h_channel_max);
	if (!xdev) {
		err = -EINVAL;
		goto error;
	}

	if ((user_max > MAX_USER_IRQ) ||
	    (h2c_channel_max > XDMA_CHANNEL_NUM_MAX) ||
	    (c2h_channel_max > XDMA_CHANNEL_NUM_MAX)) {
		err = -EINVAL;
		goto error;
	}

	if (!h2c_channel_max && !c2h_channel_max)
		pr_warn("NO engine found!\n");

	if (user_max) {
		u32 mask = (1 << (user_max + 1)) - 1;

		err = xdma_user_isr_enable(xdev, mask);
		if (err)
			goto error;
	}

	region->base = xdev->bar[mem_bar_index];
	region->dpu_fpga_aws_dev = xdev;

	return 0;
error:
	kfree(xdev_name);
	ida_simple_remove(&dpu_region_ida, region->rank.id);

	return err;
}

static int dpu_region_fpga_aws_probe(struct pci_dev *pci_dev,
				     const struct pci_device_id *id)
{
	struct device *dev = &pci_dev->dev;
	struct dpu_region *region;
	int ret;

	dev_dbg(dev, "fpga aws device probed\n");

	region = kzalloc(sizeof(struct dpu_region), GFP_KERNEL);
	if (!region) {
		ret = -ENOMEM;
		goto err;
	}

	/* 1/ Init fpga aws device */
	ret = dpu_region_init_fpga_aws(pci_dev, id->driver_data, region);
	if (ret)
		goto free_region;

	/* 2/ Init dpu_rank device associated to that aws device */
	dpu_region_set_address_translation(&region->addr_translate,
					   DPU_BACKEND_FPGA_AWS, pci_dev);

	region->addr_translate.private = region;
	mutex_init(&region->lock);
	dev_set_drvdata(dev, region);

	ret = dpu_rank_init_device(dev, region, 1);
	if (ret) {
		dev_err(dev, "cannot create dpu rank device\n");
		goto free_fpga;
	}

	ret = dpu_set_chip_id(&region->rank);
	if (ret) {
		dev_err(dev, "cannot get region chip id\n");
		ret = -1;
		goto destroy_rank_device;
	}

	dev_dbg(dev, "device loaded.\n");

	return 0;

destroy_rank_device:
	dpu_rank_release_device(region);
free_fpga:
	xdma_device_close(pci_dev, region->dpu_fpga_aws_dev);
free_region:
	kfree(region);
err:
	return ret;
}

static void dpu_region_fpga_aws_remove(struct pci_dev *pci_dev)
{
	struct device *dev = &pci_dev->dev;
	struct dpu_region *region;

	dev_dbg(dev, "removing dpu region device (and its rank)\n");

	region = dev_get_drvdata(dev);
	if (!region)
		return;

	xdma_device_close(pci_dev, region->dpu_fpga_aws_dev);

	/* Release dpu_rank device */
	dpu_rank_release_device(region);
	kfree(region);
}

#define BAR_NUMBER(x) .driver_data = (kernel_ulong_t)(x)
static struct pci_device_id dpu_region_fpga_aws_ids[] = {
	{ PCI_DEVICE(CL_DRAM_DMA_VENDOR_ID, CL_DRAM_DMA_DEVICE_ID),
	  BAR_NUMBER(4) },
	{ PCI_DEVICE(AWS_DPU_VENDOR_ID, AWS_DPU_DEVICE_ID), BAR_NUMBER(4) },
	{ PCI_DEVICE(BITTWARE_VENDOR_ID, BITTWARE_FPGA_PCIE3_DEVICE_ID),
	  BAR_NUMBER(2) },
	{
		0,
	}
};

MODULE_DEVICE_TABLE(pci, dpu_region_fpga_aws_ids);

static struct pci_driver dpu_region_fpga_aws_driver = {
	.name = DPU_REGION_NAME "_fpga_aws",
	.id_table = dpu_region_fpga_aws_ids,
	.probe = dpu_region_fpga_aws_probe,
	.remove = dpu_region_fpga_aws_remove,
};

static int __init dpu_region_init(void)
{
	int ret;
    int node;

	dpu_rank_class = class_create(THIS_MODULE, DPU_RANK_NAME);
	if (IS_ERR(dpu_rank_class)) {
		ret = PTR_ERR(dpu_rank_class);
		return ret;
	}
	dpu_rank_class->dev_groups = dpu_rank_attrs_groups;

	dpu_dax_class = class_create(THIS_MODULE, "dpu_dax");
	if (IS_ERR(dpu_dax_class)) {
		class_destroy(dpu_rank_class);
		ret = PTR_ERR(dpu_dax_class);
		return ret;
	}
	dpu_dax_class->dev_groups = dpu_dax_region_attrs_groups;

    membo_initialized = true;
    dpu_membo_class = class_create(THIS_MODULE, DPU_MEMBO_NAME);
    if (IS_ERR(dpu_membo_class))
        membo_initialized = false;
    if (membo_initialized)
        dpu_membo_class->dev_uevent = dpu_membo_dev_uevent;

	pr_debug("dpu: get rank information from DMI\n");
	dpu_rank_dmi_init();

    /* Init MemBo context for each node */
    if (membo_initialized)
        for_each_online_node(node) {
            if (init_membo_context(node) == -ENOMEM) {
                for_each_online_node(node)
                    destroy_membo_context(node);
                membo_initialized = false;
                pr_debug("membo: MemBo initialization failed\n");
                break;
            }
        }

	pr_debug("dpu: initializing memory driver\n");
	ret = platform_driver_register(&dpu_region_mem_driver);
	if (ret)
		goto mem_error;

	pr_debug("dpu: creating memory devices if available\n");
	ret = dpu_region_srat_probe();
	if (ret)
		ret = dpu_region_dev_probe();
	if (ret)
		pr_info("dpu: memory devices unavailable\n");

	pr_debug("dpu: initializing fpga kc705 driver\n");
	ret = pci_register_driver(&dpu_region_fpga_kc705_driver);
	if (ret)
		goto kc705_error;

	pr_debug("dpu: initializing fpga aws driver\n");
	ret = pci_register_driver(&dpu_region_fpga_aws_driver);
	if (ret)
		goto aws_error;

    /* Activate MemBo service */
    if (membo_initialized)
        if (dpu_membo_create_device())
            membo_initialized = false;

    if (membo_initialized) {
        membo_init();
    }

	return 0;

aws_error:
	pci_unregister_driver(&dpu_region_fpga_kc705_driver);
kc705_error:
	dpu_region_mem_exit();
	platform_driver_unregister(&dpu_region_mem_driver);
mem_error:
	dpu_rank_dmi_exit();
	class_destroy(dpu_dax_class);
	class_destroy(dpu_rank_class);
	class_destroy(dpu_membo_class);
	return ret;
}

static void __exit dpu_region_exit(void)
{
    int node;
	pr_debug("dpu_region: unloading driver\n");

	pci_unregister_driver(&dpu_region_fpga_aws_driver);
	pci_unregister_driver(&dpu_region_fpga_kc705_driver);
	dpu_region_mem_exit();
	platform_driver_unregister(&dpu_region_mem_driver);
	dpu_rank_dmi_exit();
	class_destroy(dpu_dax_class);
	class_destroy(dpu_rank_class);
	ida_destroy(&dpu_region_ida);

    dpu_membo_release_device();
	class_destroy(dpu_membo_class);
    for_each_online_node(node)
        destroy_membo_context(node);
}

module_init(dpu_region_init);
module_exit(dpu_region_exit);

module_param(default_backend, uint, 0);
MODULE_PARM_DESC(default_backend, "0: xeon_sp (default)");

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Alexandre Ghiti - UPMEM");
MODULE_VERSION("6.4");
