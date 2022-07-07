#include <linux/device.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/uio_driver.h>
#include <linux/io.h>
#include <linux/irq.h>
#include <linux/msi.h>
#include <linux/version.h>
#include <linux/slab.h>

#include "compat.h"

#define HBA_VENDOR_ID 0x10ee
#define HBA_DEVICE_ID 0X7014

struct hba_uio_pci_dev {
	struct uio_info info;
	struct pci_dev *pdev;
	atomic_t refcnt;
};

/**
 * This is the irqcontrol callback to be registered to uio_info.
 * It can be used to disable/enable interrupt from user space processes.
 *
 * @param info
 *  pointer to uio_info.
 * @param irq_state
 *  state value. 1 to enable interrupt, 0 to disable interrupt.
 *
 * @return
 *  - On success, 0.
 *  - On failure, a negative value.
 */
static int hbauio_pci_irqcontrol(struct uio_info *info, s32 irq_state)
{
	struct hba_uio_pci_dev *udev = info->priv;
	struct pci_dev *dev = udev->pdev;

	pci_cfg_access_lock(dev);

	pci_intx(dev, irq_state);

	pci_cfg_access_unlock(dev);

	return 0;
}

/**
 * This is interrupt handler which will check if the interrupt is for the right device.
 * If yes, disable it here and will be enable later.
 */
static irqreturn_t hbauio_pci_irqhandler(int irq, void *dev_id)
{
	struct hba_uio_pci_dev *udev = (struct hba_uio_pci_dev *)dev_id;
	struct uio_info *info = &udev->info;

	if (!pci_check_and_mask_intx(udev->pdev))
		return IRQ_NONE;

	uio_event_notify(info);

	/* Message signal mode, no share IRQ and automasked */
	return IRQ_HANDLED;
}

static int hbauio_pci_enable_interrupts(struct hba_uio_pci_dev *udev)
{
	int err = 0;

	if (pci_intx_mask_supported(udev->pdev)) {
		dev_dbg(&udev->pdev->dev, "using INTX\n");
		udev->info.irq_flags = IRQF_SHARED | IRQF_NO_THREAD;
		udev->info.irq = udev->pdev->irq;
	} else {
		dev_notice(&udev->pdev->dev, "PCI INTX mask not supported\n");
	}

	err = request_irq(udev->info.irq, hbauio_pci_irqhandler, udev->info.irq_flags, udev->info.name, udev);
	dev_info(&udev->pdev->dev, "hba uio device register with irq %ld\n", udev->info.irq);

	return err;
}

static void hbauio_pci_disable_interrupts(struct hba_uio_pci_dev *udev)
{
	if (udev->info.irq) {
		free_irq(udev->info.irq, udev);
		udev->info.irq = 0;
	}
}

static int hbauio_pci_open(struct uio_info *info, struct inode *inode)
{
	struct hba_uio_pci_dev *udev = info->priv;
	struct pci_dev *dev = udev->pdev;
	int err;

	if (atomic_inc_return(&udev->refcnt) != 1)
		return 0;

	/* set bus master, which was cleared by the reset function */
	pci_set_master(dev);

	/* enable interrupts */
	err = hbauio_pci_enable_interrupts(udev);
	if (err) {
		atomic_dec(&udev->refcnt);
		dev_err(&dev->dev, "Enable interrupt fails\n");
	}
	return err;
}

static int hbauio_pci_release(struct uio_info *info, struct inode *inode)
{
	struct hba_uio_pci_dev *udev = info->priv;
	struct pci_dev *dev = udev->pdev;

	if (atomic_dec_and_test(&udev->refcnt)) {
		hbauio_pci_disable_interrupts(udev);

		/* stop the device from further DMA */
		pci_clear_master(dev);
	}

	return 0;
}

/* Remap pci resources described by bar #pci_bar in uio resource n. */
static int hbauio_pci_setup_iomem(struct pci_dev *dev, struct uio_info *info, int n, int pci_bar, const char *name)
{
	unsigned long addr, len;
	void *internal_addr;

	if (n >= ARRAY_SIZE(info->mem))
		return -EINVAL;

	addr = pci_resource_start(dev, pci_bar);
	len = pci_resource_len(dev, pci_bar);
	if (addr == 0 || len == 0)
		return -1;

	internal_addr = ioremap(addr, len);
	if (internal_addr == NULL)
		return -1;
	info->mem[n].name = name;
	info->mem[n].addr = addr;
	info->mem[n].internal_addr = internal_addr;
	info->mem[n].size = len;
	info->mem[n].memtype = UIO_MEM_PHYS;
	return 0;
}

/* Get pci port io resources described by bar #pci_bar in uio resource n. */
static int hbauio_pci_setup_ioport(struct pci_dev *dev, struct uio_info *info, int n, int pci_bar, const char *name)
{
	unsigned long addr, len;

	if (n >= ARRAY_SIZE(info->port))
		return -EINVAL;

	addr = pci_resource_start(dev, pci_bar);
	len = pci_resource_len(dev, pci_bar);
	if (addr == 0 || len == 0)
		return -EINVAL;

	info->port[n].name = name;
	info->port[n].start = addr;
	info->port[n].size = len;
	info->port[n].porttype = UIO_PORT_X86;

	return 0;
}

/* Unmap previously ioremap'd resources */
static void hbauio_pci_release_iomem(struct uio_info *info)
{
	int i;

	for (i = 0; i < MAX_UIO_MAPS; i++) {
		if (info->mem[i].internal_addr)
			iounmap(info->mem[i].internal_addr);
	}
}

static int hbauio_setup_bars(struct pci_dev *dev, struct uio_info *info)
{
	int i, iom, iop, ret;
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
		if (pci_resource_len(dev, i) != 0 && pci_resource_start(dev, i) != 0) {
			flags = pci_resource_flags(dev, i);
			if (flags & IORESOURCE_MEM) {
				ret = hbauio_pci_setup_iomem(dev, info, iom, i, bar_names[i]);
				if (ret != 0)
					return ret;
				iom++;
			} else if (flags & IORESOURCE_IO) {
				ret = hbauio_pci_setup_ioport(dev, info, iop, i, bar_names[i]);
				if (ret != 0)
					return ret;
				iop++;
			}
		}
	}

	return (iom != 0 || iop != 0) ? ret : -ENOENT;
}

static int hbauio_pci_probe(struct pci_dev *dev, const struct pci_device_id *id)
{
	struct hba_uio_pci_dev *udev;
	//dma_addr_t map_dma_addr;
	//void *map_addr;
	int err;

#ifdef HAVE_PCI_IS_BRIDGE_API
	if (pci_is_bridge(dev)) {
		dev_warn(&dev->dev, "Ignoring PCI bridge device\n");
		return -ENODEV;
	}
#endif

	udev = kzalloc(sizeof(struct hba_uio_pci_dev), GFP_KERNEL);
	if (udev == NULL)
		return -ENOMEM;

	err = pci_enable_device(dev);
	if (err != 0) {
		dev_err(&dev->dev, "Cannot enalbe PCI device\n");
		goto fail_free;
	}

	pci_set_master(dev);
	pci_try_set_mwi(dev);

	/* remap IO memory */
	err = hbauio_setup_bars(dev, &udev->info);
	if (err != 0)
		goto fail_release_iomem;

	if (pci_set_dma_mask(dev, DMA_BIT_MASK(64)) || pci_set_consistent_dma_mask(dev, DMA_BIT_MASK(64))) {
		dev_info(&dev->dev, "Cannot set DMA mask 64 and consistent DMA mask 64\n");
		if (pci_set_dma_mask(dev, DMA_BIT_MASK(32)) || pci_set_consistent_dma_mask(dev, DMA_BIT_MASK(32))) {
			dev_err(&dev->dev, "Cannot set DMA mask 32 and consistent DMA mask 32\n");
			goto fail_release_iomem;
		}
	}

	/* fill uio infos */
	udev->info.name = "hba_uio";
	udev->info.version = "0.1";
	udev->info.irqcontrol = hbauio_pci_irqcontrol;
	udev->info.open = hbauio_pci_open;
	udev->info.release = hbauio_pci_release;
	udev->info.priv = udev;
	udev->pdev = dev;
	atomic_set(&udev->refcnt, 0);

	err = uio_register_device(&dev->dev, &udev->info);
	if (err != 0)
		goto fail_release_iomem;

	pci_set_drvdata(dev, udev);

	return 0;

fail_release_iomem:
	hbauio_pci_release_iomem(&udev->info);
	pci_disable_device(dev);
fail_free:
	kfree(udev);

	return 0;
}



static void hbauio_pci_remove(struct pci_dev *dev)
{
	struct hba_uio_pci_dev *udev = pci_get_drvdata(dev);

	hbauio_pci_release(&udev->info, NULL);

	uio_unregister_device(&udev->info);
	hbauio_pci_release_iomem(&udev->info);
	pci_disable_device(dev);
	pci_set_drvdata(dev, NULL);
	kfree(udev);
}


static struct pci_device_id hbauio_pci_table[] = {
	{
		HBA_VENDOR_ID, HBA_DEVICE_ID, PCI_ANY_ID, PCI_ANY_ID, 0, 0, 0
	},
	{}
};

static struct pci_driver igbuio_pci_driver = {
	.name = "hba_uio",
	.id_table = hbauio_pci_table,
	.probe = hbauio_pci_probe,
	.remove = hbauio_pci_remove,
};

static int __init hbauio_pci_init_module(void)
{
	if (hbauio_kernel_is_locked_down()) {
		pr_err("Not able to use module, kernel lock down is enabled\n");
		return -EINVAL;
	}

	return pci_register_driver(&igbuio_pci_driver);
}

static void __exit hbauio_pci_exit_module(void)
{
	pci_unregister_driver(&igbuio_pci_driver);
}

module_init(hbauio_pci_init_module);
module_exit(hbauio_pci_exit_module);

MODULE_DESCRIPTION("UIO driver for HBA cards");
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Yanli.Qian");
