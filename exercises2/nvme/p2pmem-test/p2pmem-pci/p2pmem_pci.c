/* * P2PMEM PCI EP Device Driver
 * Copyright (c) 2017, Eideticom
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 * more details.
 *
 * Copyright (C) 2017 Eideitcom
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/pci-p2pdma.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/pfn_t.h>

#include <linux/mmap_lock.h>
#include <linux/mm.h>

#define PCI_VENDOR_EIDETICOM 0x1de5
#define PCI_VENDOR_MICROSEMI 0x11f8
#define PCI_MTRAMON_DEV_ID   0xf117
#define PCI_VENDOR_QEMU 0x1b36
#define PCI_DEV_QEMU 0x0010

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Stephen Bates <stephen@eideticom.com");
MODULE_DESCRIPTION("A P2PMEM driver for simple PCIe End Points (EPs)");

static int max_devices = 16;
module_param(max_devices, int, 0444);
MODULE_PARM_DESC(max_devices, "Maximum number of char devices");

#define MTRAMON_BAR 4
#define QEMU_BAR 2
#define MTRAMON_INIT 0

static struct class *p2pmem_class;
static DEFINE_IDA(p2pmem_ida);
static dev_t p2pmem_devt;

static struct pci_device_id p2pmem_pci_id_table[] = {
        { PCI_DEVICE(PCI_VENDOR_EIDETICOM, 0x1000), .driver_data = 0 },
	{ PCI_DEVICE(PCI_VENDOR_QEMU, PCI_DEV_QEMU), .driver_data = QEMU_BAR},
	{ PCI_DEVICE(PCI_VENDOR_MICROSEMI,
		     PCI_MTRAMON_DEV_ID), .driver_data = MTRAMON_BAR },
	{ 0, }
};
MODULE_DEVICE_TABLE(pci, p2pmem_pci_id_table);

struct p2pmem_dev {
	struct device dev;
	struct pci_dev *pdev;
	int id;
	struct cdev cdev;
	bool created_by_hack;
};

static struct p2pmem_dev *to_p2pmem(struct device *dev)
{
	return container_of(dev, struct p2pmem_dev, dev);
}

struct p2pmem_vma {
	struct p2pmem_dev *p2pmem_dev;
	atomic_t mmap_count;
	size_t nr_pages;

	/* Protects the used_pages array */
	struct mutex mutex;
	struct page *used_pages[];
};

static void p2pmem_vma_open(struct vm_area_struct *vma)
{
	struct p2pmem_vma *pv = vma->vm_private_data;

	atomic_inc(&pv->mmap_count);
}

static void p2pmem_vma_free_pages(struct vm_area_struct *vma)
{
	int i;
	struct p2pmem_vma *pv = vma->vm_private_data;

	mutex_lock(&pv->mutex);

	for (i = 0; i < pv->nr_pages; i++) {
		if (pv->used_pages[i]) {
			pci_free_p2pmem(pv->p2pmem_dev->pdev,
					page_to_virt(pv->used_pages[i]),
					PAGE_SIZE);
			pv->used_pages[i] = NULL;
		}
	}

	mutex_unlock(&pv->mutex);
}

static void p2pmem_vma_close(struct vm_area_struct *vma)
{
	struct p2pmem_vma *pv = vma->vm_private_data;

	if (!atomic_dec_and_test(&pv->mmap_count))
		return;

	p2pmem_vma_free_pages(vma);

	dev_dbg(&pv->p2pmem_dev->dev, "vma close");
	kfree(pv);
}

static vm_fault_t p2pmem_vma_fault(struct vm_fault *vmf)
{
	struct p2pmem_vma *pv = vmf->vma->vm_private_data;
	unsigned int pg_idx;
	struct page *pg;
	pfn_t pfn;
	vm_fault_t rc;

	pg_idx = (vmf->address - vmf->vma->vm_start) / PAGE_SIZE;

	mutex_lock(&pv->mutex);

	if (pv->used_pages[pg_idx])
		pg = pv->used_pages[pg_idx];
	else
		pg = virt_to_page(pci_alloc_p2pmem(pv->p2pmem_dev->pdev,
						   PAGE_SIZE));

	if (!pg) {
		mutex_unlock(&pv->mutex);
		return VM_FAULT_OOM;
	}

	pv->used_pages[pg_idx] = pg;

	pfn = phys_to_pfn_t(page_to_phys(pg), PFN_DEV | PFN_MAP);
	rc = vmf_insert_mixed(vmf->vma, vmf->address, pfn);

	mutex_unlock(&pv->mutex);

	return rc;
}

const struct vm_operations_struct p2pmem_vmops = {
	.open = p2pmem_vma_open,
	.close = p2pmem_vma_close,
	.fault = p2pmem_vma_fault,
};

static int p2pmem_open(struct inode *inode, struct file *filp)
{
	struct p2pmem_dev *p;

	p = container_of(inode->i_cdev, struct p2pmem_dev, cdev);
	filp->private_data = p;

	return 0;
}

static int p2pmem_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct p2pmem_dev *p = filp->private_data;
	struct p2pmem_vma *pv;
	size_t nr_pages = (vma->vm_end - vma->vm_start) / PAGE_SIZE;

	if ((vma->vm_flags & VM_MAYSHARE) != VM_MAYSHARE) {
		dev_warn(&p->dev, "mmap failed: can't create private mapping\n");
		return -EINVAL;
	}

	dev_dbg(&p->dev, "Allocating mmap with %zd pages.\n", nr_pages);

	pv = kzalloc(sizeof(*pv) + sizeof(pv->used_pages[0]) * nr_pages,
		     GFP_KERNEL);
	if (!pv)
		return -ENOMEM;

	mutex_init(&pv->mutex);
	pv->nr_pages = nr_pages;
	pv->p2pmem_dev = p;
	atomic_set(&pv->mmap_count, 1);

	vma->vm_private_data = pv;
	vma->vm_ops = &p2pmem_vmops;
#if 0
	vma->vm_flags |= VM_MIXEDMAP;
#else
	vm_flags_set(vma, VM_MIXEDMAP);
#endif
	return 0;
}

static const struct file_operations p2pmem_fops = {
	.owner = THIS_MODULE,
	.open = p2pmem_open,
	.mmap = p2pmem_mmap,
};

static int p2pmem_test_page_mappings(struct p2pmem_dev *p)
{
	void *addr;
	int err = 0;
	struct page *page;
	struct pci_bus_region bus_region;
	struct resource res;
	phys_addr_t pa;

	addr = pci_alloc_p2pmem(p->pdev, PAGE_SIZE);
	if (!addr)
		return -ENOMEM;

	page = virt_to_page(addr);
	if (!is_zone_device_page(page)) {
		dev_err(&p->dev,
			"ERROR: kernel virt_to_page does not point to a ZONE_DEVICE page!");
		err = -EFAULT;
		goto out;
	}

	bus_region.start = pci_p2pmem_virt_to_bus(p->pdev, addr);
	bus_region.end = bus_region.start + PAGE_SIZE;

	pcibios_bus_to_resource(p->pdev->bus, &res, &bus_region);

	pa = page_to_phys(page);
	if (pa != res.start) {
		dev_err(&p->dev,
			"ERROR: page_to_phys does not map to the BAR address!"
			"  %pa[p] != %pa[p]", &pa, &res.start);
		err = -EFAULT;
		goto out;
	}

	pa = virt_to_phys(addr);
	if (pa != res.start) {
		dev_err(&p->dev,
			"ERROR: virt_to_phys does not map to the BAR address!"
			"  %pa[p] != %pa[p]", &pa, &res.start);
		err = -EFAULT;
		goto out;
	}

	if (page_to_virt(page) != addr) {
		dev_err(&p->dev,
			"ERROR: page_to_virt does not map to the correct address!");
		err = -EFAULT;
		goto out;
	}

out:
	if (err == 0)
		dev_info(&p->dev, "kernel page mappings seem sane.");

	pci_free_p2pmem(p->pdev, addr, PAGE_SIZE);
	return err;
}

static int p2pmem_test_p2p_access(struct p2pmem_dev *p)
{
	u32 *addr;
	const u32 test_value = 0x11223344;
	int err = 0;

	addr = pci_alloc_p2pmem(p->pdev, PAGE_SIZE);
	if (!addr)
		return -ENOMEM;

	WRITE_ONCE(addr[0], 0);
	if (READ_ONCE(addr[0]) != 0) {
		err = -EFAULT;
		goto out;
	}

	WRITE_ONCE(addr[0], test_value);
	if (READ_ONCE(addr[0]) != test_value) {
		err = -EFAULT;
		goto out;
	}

out:
	if (err == 0)
		dev_info(&p->dev, "kernel can access p2p memory.");
	else
		dev_err(&p->dev, "ERROR: kernel can't access p2p memory!");

	pci_free_p2pmem(p->pdev, addr, PAGE_SIZE);
	return err;
}

static int p2pmem_test(struct p2pmem_dev *p)
{
	int err;

	err = p2pmem_test_page_mappings(p);
	if (err)
		return err;

	return p2pmem_test_p2p_access(p);
}

static void p2pmem_release(struct device *dev)
{
	struct p2pmem_dev *p = to_p2pmem(dev);

	kfree(p);
}

static struct p2pmem_dev *p2pmem_create(struct pci_dev *pdev)
{
	struct p2pmem_dev *p;
	int err;

	p = kzalloc(sizeof(*p), GFP_KERNEL);
	if (!p)
		return ERR_PTR(-ENOMEM);

	p->pdev = pdev;

	device_initialize(&p->dev);
	p->dev.class = p2pmem_class;
	p->dev.parent = &pdev->dev;
	p->dev.release = p2pmem_release;

	p->id = ida_simple_get(&p2pmem_ida, 0, 0, GFP_KERNEL);
	if (p->id < 0) {
		err = p->id;
		goto out_free;
	}

	dev_set_name(&p->dev, "p2pmem%d", p->id);
	p->dev.devt = MKDEV(MAJOR(p2pmem_devt), p->id);

	cdev_init(&p->cdev, &p2pmem_fops);
	p->cdev.owner = THIS_MODULE;

	err = cdev_device_add(&p->cdev, &p->dev);
	if (err)
		goto out_ida;

	dev_info(&p->dev, "registered");

	p2pmem_test(p);

	return p;

out_ida:
	ida_simple_remove(&p2pmem_ida, p->id);
out_free:
	kfree(p);
	return ERR_PTR(err);
}

void p2pmem_destroy(struct p2pmem_dev *p)
{
	dev_info(&p->dev, "unregistered");
	cdev_device_del(&p->cdev, &p->dev);
	ida_simple_remove(&p2pmem_ida, p->id);
	put_device(&p->dev);
}
static int
print_bars(struct pci_dev *dev)
{
	int i, iom, iop;
	unsigned long flags;
	static const char *bar_names[PCI_STD_RESOURCE_END + 1]  = {
		"BAR0",
		"BAR1",
		"BAR2",
		"BAR3",
		"BAR4",
		"BAR5",
	};

	iom = 0;
	iop = 0;

	for (i = 0; i < ARRAY_SIZE(bar_names); i++) {
		if (pci_resource_len(dev, i) != 0 &&
				pci_resource_start(dev, i) != 0) {
			flags = pci_resource_flags(dev, i);
			if (flags & IORESOURCE_MEM) {
				pr_info("****mem  %s \n", bar_names[i]);
				iom++;
			} else if (flags & IORESOURCE_IO) {
				pr_info("****io %s \n", bar_names[i]);
				iop++;
			}
		}
	}
        return 0;

}
static int pci_get_res(int num ,struct pci_dev *pdev, resource_size_t *res_size)
{
   struct resource *res;
   res = &pdev->resource[2];
   *res_size = resource_size(res);
   return 0;
}
static int p2pmem_pci_probe(struct pci_dev *pdev,
			    const struct pci_device_id *id)
{
	struct p2pmem_dev *p;
	int err = 0;
	resource_size_t res_size;
	if (pci_enable_device_mem(pdev) < 0) {
		dev_err(&pdev->dev, "unable to enable device!\n");
		goto out;
	}

	print_bars(pdev);
	pci_get_res(QEMU_BAR,pdev,&res_size);
	err = pci_p2pdma_add_resource(pdev, QEMU_BAR, res_size, 0);
	if (err) {
		dev_err(&pdev->dev, "unable to add p2p resource");
		goto out_disable_device;
	}

	pci_p2pmem_publish(pdev, true);

	p = p2pmem_create(pdev);
	if (IS_ERR(p))
		goto out_disable_device;

	pci_set_drvdata(pdev, p);

	pr_info("%s succ \n", __func__);
	return 0;

out_disable_device:
	pci_disable_device(pdev);
out:
	return err;
}

static void p2pmem_pci_remove(struct pci_dev *pdev)
{
	struct p2pmem_dev *p = pci_get_drvdata(pdev);

	p2pmem_destroy(p);
}

static struct pci_driver p2pmem_pci_driver = {
	.name = "p2pmem_pci",
	.id_table = p2pmem_pci_id_table,
	.probe = p2pmem_pci_probe,
	.remove = p2pmem_pci_remove,
};

#if MTRAMON_INIT
static void ugly_mtramon_hack_init(void)
{
	struct pci_dev *pdev = NULL;
	struct p2pmem_dev *p;
	int err;

	while ((pdev = pci_get_device(PCI_VENDOR_MICROSEMI,
				      PCI_MTRAMON_DEV_ID,
				      pdev))) {
		// If there's no driver it can be handled by the regular
		//  pci driver case
		if (!pdev->driver)
			continue;

		// The NVME driver already handled it
		if (pdev->p2pdma)
			continue;

		if (!pdev->p2pdma) {
			err = pci_p2pdma_add_resource(pdev, MTRAMON_BAR, 0, 0);
			if (err) {
				dev_err(&pdev->dev,
					"unable to add p2p resource");
				continue;
			}
		}

		p = p2pmem_create(pdev);
		if (!p)
			continue;

		p->created_by_hack = true;
	}
}
#endif
static void ugly_hack_to_create_p2pmem_devs_for_other_devices(void)
{
	struct pci_dev *pdev = NULL;
	struct p2pmem_dev *p;

	while ((pdev = pci_get_device(PCI_ANY_ID, PCI_ANY_ID, pdev))) {
		if (!pdev->p2pdma)
			continue;

		p = p2pmem_create(pdev);
		if (!p)
			continue;

		p->created_by_hack = true;
	}
}

static void ugly_hack_deinit(void)
{
	struct class_dev_iter iter;
	struct device *dev;
	struct p2pmem_dev *p;

	class_dev_iter_init(&iter, p2pmem_class, NULL, NULL);
	while ((dev = class_dev_iter_next(&iter))) {
		p = to_p2pmem(dev);
		if (p->created_by_hack)
			p2pmem_destroy(p);
	}
	class_dev_iter_exit(&iter);
}

static int __init p2pmem_pci_init(void)
{
	int rc;

	p2pmem_class = class_create(THIS_MODULE, "p2pmem_device");
	if (IS_ERR(p2pmem_class))
		return PTR_ERR(p2pmem_class);

	rc = alloc_chrdev_region(&p2pmem_devt, 0, max_devices, "p2pmem");
	if (rc)
		goto err_class;

	ugly_hack_to_create_p2pmem_devs_for_other_devices();
#if MTRAMON_INIT
	ugly_mtramon_hack_init();
#endif
	rc = pci_register_driver(&p2pmem_pci_driver);
	if (rc)
		goto err_chdev;

	pr_info(KBUILD_MODNAME ": module loaded\n");

	return 0;
err_chdev:
	unregister_chrdev_region(p2pmem_devt, max_devices);
err_class:
	class_destroy(p2pmem_class);
	return rc;
}

static void __exit p2pmem_pci_cleanup(void)
{
	pci_unregister_driver(&p2pmem_pci_driver);
	ugly_hack_deinit();
	unregister_chrdev_region(p2pmem_devt, max_devices);
	class_destroy(p2pmem_class);
	pr_info(KBUILD_MODNAME ": module unloaded\n");
}

late_initcall(p2pmem_pci_init);
module_exit(p2pmem_pci_cleanup);
