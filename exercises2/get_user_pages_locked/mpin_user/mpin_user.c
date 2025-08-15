// SPDX-License-Identifier: GPL-2.0
/*
 * mpin_user.c - Memory pinning kernel module
 *
 * Copyright (C) 2023 Weka.io ltd.
 *
 * Author: Zhou Wang <wangzhou1-AT-hisilicon.com>
 * Author: Boaz Harrosh <boaz@weka.io>
 * ~~~~~
 * Original code is from a posted kernel patch:
 * Subject: uacce: Add uacce_ctrl misc device
 * From: Zhou Wang <wangzhou1-AT-hisilicon.com>
 * https://lwn.net/Articles/843432/
 *
 */
/*
 * This module provides an ioctl to pin user memory pages.
 * This is necessary to prevent physical addresses from changing,
 * as user, like dpdk/spdk use the physical addresses in user-mode apps,
 * And also to overcome limitations when using DPDK in systems without the
 * IOMMU enabled.
 */

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/device.h>
#include <linux/module.h>
#include <linux/version.h>
#include <linux/slab.h>
#include <linux/miscdevice.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/fs.h> /* lets us pick up on xarray.h */

#include "mpin_user.h"

/*
 * We assume that there is no need to do pinning at all on kernel versions
 * older than where xarray.h is included, based on our observations. At
 * any rate, our implementation depends on xarray.
 *
 * We will report an error on device open if memory pinning is not possible.
 */
#ifndef XA_FLAGS_ALLOC
#	undef ENABLE_MPIN
#	define MPIN_ENABLED 0
#else
#	define ENABLE_MPIN
#	define MPIN_ENABLED 1
#endif

#ifdef ENABLE_MPIN

#ifndef FOLL_PIN
static inline
long pin_user_pages(unsigned long start, unsigned long nr_pages,
		unsigned int gup_flags, struct page **pages,
		struct vm_area_struct **vmas)
{
	return get_user_pages(start, nr_pages, gup_flags, pages, vmas);
}

void unpin_user_pages(struct page **pages, unsigned long npages)
{
	uint i;

	for (i = 0; i < npages; ++i)
		put_page(pages[i]);
}

#ifndef	FOLL_LONGTERM
#define	FOLL_LONGTERM 0
#endif

#endif /* missing FOLL_PIN, introduced in Linux 5.6 */

struct mpin_user_container {
	struct xarray array;
};

struct pin_pages {
	unsigned long first;
	unsigned long nr_pages;
	struct page **pages;
};

static int mpin_user_pin_page(struct mpin_user_container *priv, struct mpin_user_address *addr)
{
	unsigned int flags = FOLL_FORCE | FOLL_WRITE | FOLL_LONGTERM;
	unsigned long first, last, nr_pages;
	struct page **pages;
	struct pin_pages *p;
	int ret;

	if (!(addr->addr && addr->size)) {
		pr_err("mpin_user: %s: called-by(%s:%d) addr=0x%llx size=0x%llx\n",
			__func__, current->comm, current->pid, addr->addr, addr->size);
		return 0; /* nothing to pin */
	}

	first = (addr->addr & PAGE_MASK) >> PAGE_SHIFT;
	last = ((addr->addr + addr->size - 1) & PAGE_MASK) >> PAGE_SHIFT;
	nr_pages = last - first + 1;

	pr_debug("mpin_user: %s: called-by(%s:%d) addr=0x%llx size=0x%llx first=0x%lx last=0x%lx nr_pages=0x%lx",
		__func__, current->comm, current->pid, addr->addr, addr->size, first, last,
		nr_pages);

	pages = vmalloc(nr_pages * sizeof(struct page *));
	if (pages == NULL) {
		pr_err("mpin_user: %s called-by(%s:%d) addr=0x%llx size=0x%llx first=0x%lx last=0x%lx nr_pages=0x%lx",
			__func__, current->comm, current->pid, addr->addr, addr->size, first, last,
			nr_pages);
		return -ENOMEM;
	}

	p = kzalloc(sizeof(*p), GFP_KERNEL);
	if (p == NULL) {
		ret = -ENOMEM;
		goto free;
	}

	ret = pin_user_pages(addr->addr & PAGE_MASK, nr_pages, flags, pages, NULL);
	if (ret != nr_pages) {
		pr_err("uacce: Failed to pin page\n");
		goto free_p;
	}

	p->first = first;
	p->nr_pages = nr_pages;
	p->pages = pages;

	ret = xa_err(xa_store(&priv->array, p->first, p, GFP_KERNEL));
	if (ret != 0)
		goto unpin_pages;
	return 0;

unpin_pages:
	unpin_user_pages(pages, nr_pages);
free_p:
	kfree(p);
free:
	vfree(pages);
	return ret;
}

static int mpin_user_unpin_page(struct mpin_user_container *priv,
				struct mpin_user_address *addr)
{
	unsigned long first, last, nr_pages;
	struct pin_pages *p;

	first = (addr->addr & PAGE_MASK) >> PAGE_SHIFT;
	last = ((addr->addr + addr->size - 1) & PAGE_MASK) >> PAGE_SHIFT;
	nr_pages = last - first + 1;

	/* find pin_pages */
	p = xa_load(&priv->array, first);
	if (p == NULL)
		return -ENODEV;
	if (p->nr_pages != nr_pages)
		return -EINVAL;

	/* unpin */
	unpin_user_pages(p->pages, p->nr_pages);

	/* release resource */
	xa_erase(&priv->array, first);
	vfree(p->pages);
	kfree(p);

	return 0;
}

#endif /* ENABLE_MPIN */

static int mpin_open(struct inode *inode, struct file *file)
{
#ifdef ENABLE_MPIN
	struct mpin_user_container *p;

	p = kzalloc(sizeof(*p), GFP_KERNEL);
	if (p == NULL)
		return -ENOMEM;

	file->private_data = p;
	xa_init(&p->array);

	return 0;
#else
    return -ENOTSUPP;
#endif /* ENABLE_MPIN */
}

static int mpin_release(struct inode *inode, struct file *file)
{
#ifdef ENABLE_MPIN
	struct mpin_user_container *priv = file->private_data;
	struct pin_pages *p;
	unsigned long idx;

	xa_for_each(&priv->array, idx, p) {
		unpin_user_pages(p->pages, p->nr_pages);
		xa_erase(&priv->array, p->first);
		vfree(p->pages);
		kfree(p);
	}

	xa_destroy(&priv->array);
	kfree(priv);
#endif /* ENABLE_MPIN */

	return 0;
}

static long mpin_unl_ioctl(struct file *filep, unsigned int cmd,
				unsigned long arg)
{
#ifdef ENABLE_MPIN
	struct mpin_user_container *p = filep->private_data;
	struct mpin_user_address addr;
	int (*func)(struct mpin_user_container *priv, struct mpin_user_address *addr);

	switch (cmd) {
	case MPIN_CMD_PIN:
		func = mpin_user_pin_page;
		break;

	case MPIN_CMD_UNPIN:
		func = mpin_user_unpin_page;
		break;

	default:
		return -EINVAL;
	}

	if (copy_from_user(&addr, (void __user *)arg, sizeof(struct mpin_user_address)))
		return -EFAULT;

	return (*func)(p, &addr);

#else /* ! ENABLE_MPIN */
	switch (cmd) {
	case MPIN_CMD_PIN:
		return 0;
	case MPIN_CMD_UNPIN:
		return 0;
	default:
		return -EINVAL;
	}
#endif /* ! ENABLE_MPIN */
}

static ssize_t mpin_read(struct file *file, char __user *buf, size_t nbytes, loff_t *ppos)
{
	const char s[] = __stringify(MPIN_ENABLED) "\n";
	ssize_t l = sizeof(s);

	if (*ppos > 0)
		return 0;

	if (copy_to_user(buf, s, l))
		return -EFAULT;

	*ppos += l;
	return l - 1;
}

/* ~~~ Register/unregister the mpin_user interface ~~~ */

#include <linux/proc_fs.h>
/* Kernel 5.6 changed the proc_fs API */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 6, 0)
#	define OWNER_THIS_MODULE .owner = THIS_MODULE,
#	define proc_ops file_operations
#	define proc_open	open
#	define proc_read	read
#	define proc_write	write
#	define proc_release	release
#	define proc_mmap	mmap
#	define proc_poll	poll
#	define proc_lseek	llseek
#	define proc_ioctl	unlocked_ioctl
#else
#	define OWNER_THIS_MODULE
#endif

static const struct proc_ops proc_mpin_user_ops = {
	OWNER_THIS_MODULE
	.proc_open = mpin_open,
	.proc_release = mpin_release,
	.proc_ioctl = mpin_unl_ioctl,
	.proc_read = mpin_read,
};

static int __init mpin_misc_init(void)
{
	proc_create(MPIN_USER_N, 0666, NULL, &proc_mpin_user_ops);
	pr_info("%s loaded. Pinning is %s\n", MPIN_USER_N, MPIN_ENABLED ? "enabled" : "disabled");
	return 0;
}

static void __exit mpin_misc_exit(void)
{
	pr_info("%s unloaded\n", MPIN_USER_N);
	remove_proc_entry(MPIN_USER_N, NULL);
}

module_init(mpin_misc_init)
module_exit(mpin_misc_exit)

MODULE_LICENSE("GPL");
MODULE_VERSION(__stringify(MPIN_USER_VERSION));
