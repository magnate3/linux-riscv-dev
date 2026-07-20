/*
 * Copyright 2016 Ecole Polytechnique Federale Lausanne (EPFL)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of version 2 of the GNU General Public License as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St., Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <linux/module.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/pci.h>
#include <linux/uaccess.h>

#include "pcidma.h"

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("control bus master (DMA) for PCI devices");

static DEFINE_SPINLOCK(devices_lock);

#define MAX_DEVICES 8

static struct pci_dev *enabled_device[MAX_DEVICES];
static int nr_devices;

static int pcidma_enable(struct args_enable *args)
{
	struct pci_loc *loc;
	unsigned int devfn;
	struct pci_dev *dev;

	if (nr_devices >= MAX_DEVICES)
		return -EBUSY;

	loc = &args->pci_loc;

	devfn = PCI_DEVFN(loc->slot, loc->func);
	dev = pci_get_domain_bus_and_slot(loc->domain, loc->bus, devfn);
	if (!dev)
		return -EINVAL;

	pci_set_master(dev);

	spin_lock(&devices_lock);
	enabled_device[nr_devices++] = dev;
	spin_unlock(&devices_lock);

	return 0;
}

static int pcidma_release(struct inode *inode, struct file *file)
{
	int i;

	spin_lock(&devices_lock);
	for (i = 0; i < nr_devices; i++)
		pci_clear_master(enabled_device[i]);
	nr_devices = 0;
	spin_unlock(&devices_lock);

	return 0;
}

static long pcidma_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct args_enable args_enable;
	void __user *argp = (void __user *)arg;

	switch (cmd) {
	case PCIDMA_ENABLE:
		if (copy_from_user(&args_enable, argp, sizeof(args_enable)))
			return -EFAULT;
		return pcidma_enable(&args_enable);
	default:
		return -ENOTTY;
	}
}

static const struct file_operations pcidma_fops = {
	.owner		= THIS_MODULE,
	.release	= pcidma_release,
	.unlocked_ioctl	= pcidma_ioctl,
};

static struct miscdevice pcidma_dev = {
	MISC_DYNAMIC_MINOR,
	"pcidma",
	&pcidma_fops,
};

static int __init pcidma_init(void)
{
	return misc_register(&pcidma_dev);
}

static void __exit pcidma_exit(void)
{
	misc_deregister(&pcidma_dev);
}

module_init(pcidma_init);
module_exit(pcidma_exit);
