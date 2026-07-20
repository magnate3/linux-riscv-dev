/* SPDX-License-Identifier: GPL-2.0-or-later */

/* DMA Buffer Exporter Kernel Mode Driver. 
 * Copyright (C) 2021 Intel Corporation.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
 
#include <linux/device.h>
#include <linux/dma-buf.h>
#include <linux/highmem.h>
#include <linux/module.h>
#include <linux/slab.h>
#include <linux/miscdevice.h>
#include "./dma_buf_exporter_kmd.h"

MODULE_IMPORT_NS(DMA_BUF);

#define DRIVER_VERSION  "1.0.0-0"
#define DRIVER_AUTHOR   "Intel Corporation"
#define DRIVER_DESC     "DMA Buffer Exporter Driver"

struct dma_buf_exporter_data {
	int num_pages;
	struct page *pages[];
};

static struct dma_buf *dma_buf_exporter_alloc(size_t size);
static void dma_buf_exporter_free(struct dma_buf *dma_buf);

static int dma_buf_exporter_attach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment)
{
	pr_info("dma_buf_exporter: allattaching  dma_buf \n");
	return 0;
}

static void dma_buf_exporter_detach(struct dma_buf *dmabuf, struct dma_buf_attachment *attachment)
{
	pr_info("dma_buf_exporter: detaching  dma_buf \n");
	return;
}

static struct sg_table *dma_buf_exporter_map_dma_buf(struct dma_buf_attachment *attachment,
					 enum dma_data_direction dir)
{
	struct dma_buf_exporter_data *data = attachment->dmabuf->priv;
	struct sg_table *table;
	struct scatterlist *sg;
	int i;

	pr_info("dma_buf_exporter: mapping dma_buf \n");
	table = kmalloc(sizeof(*table), GFP_KERNEL);
	if (!table)
		return ERR_PTR(-ENOMEM);

	if (sg_alloc_table(table, data->num_pages, GFP_KERNEL)) {
		kfree(table);
		return ERR_PTR(-ENOMEM);
	}

	sg = table->sgl;
	for (i = 0; i < data->num_pages; i++) {
		sg_set_page(sg, data->pages[i], PAGE_SIZE, 0);
		sg = sg_next(sg);
	}

	if (!dma_map_sg(NULL, table->sgl, table->nents, dir)) {
		sg_free_table(table);
		kfree(table);
		return ERR_PTR(-ENOMEM);
	}

	return table;
}

static void dma_buf_exporter_unmap_dma_buf(struct dma_buf_attachment *attachment,
			       struct sg_table *table,
			       enum dma_data_direction dir)
{
	pr_info("dma_buf_exporter: unmapping dma_buf \n");
	dma_unmap_sg(NULL, table->sgl, table->nents, dir);
	sg_free_table(table);
	kfree(table);
	return;
}

static void dma_buf_exporter_release_dma_buf(struct dma_buf *dma_buf)
{
	pr_info("dma_buf_exporter: releasing dma_buf \n");
	return;
}

static const struct dma_buf_ops dma_buf_exporter_ops = {
	.attach = dma_buf_exporter_attach,
	.detach = dma_buf_exporter_detach,
	.map_dma_buf = dma_buf_exporter_map_dma_buf,
	.unmap_dma_buf = dma_buf_exporter_unmap_dma_buf,
	.release = dma_buf_exporter_release_dma_buf,
};

static struct dma_buf *dma_buf_exporter_alloc(size_t size)
{
	DEFINE_DMA_BUF_EXPORT_INFO(dma_buf_exporter_info);
	struct dma_buf *dma_buf;
	struct dma_buf_exporter_data *data;
	int i, npages;

	pr_info("dma_buf_exporter: allocating dma_buf \n");
	npages = PAGE_ALIGN(size) / PAGE_SIZE;
	if (!npages) {
		printk("Invalid npages ... \n");
		return ERR_PTR(-EINVAL);
	}
		

	printk("allocating private data ... npages = %d \n", npages);
	data = kmalloc(sizeof(*data) + npages * sizeof(struct page *),
		       GFP_KERNEL);
	if (!data) {
		printk("Failed to allocate private data \n");
		return ERR_PTR(-ENOMEM);
	}

	for (i = 0; i < npages; i++) {
		data->pages[i] = alloc_page(GFP_KERNEL);
		if (!data->pages[i]){
			printk("Unable to allocate page ...\n");
			goto err;
		}
			
	}
	data->num_pages = npages;
#if 0
struct dma_buf_export_info {
  const char * exp_name;
  struct module * owner;
  const struct dma_buf_ops * ops;
  size_t size;
  int flags;
  struct reservation_object * resv;
  void * priv;
};
#endif
	dma_buf_exporter_info.exp_name = KBUILD_MODNAME;
	dma_buf_exporter_info.owner = THIS_MODULE;
	dma_buf_exporter_info.ops = &dma_buf_exporter_ops;
	dma_buf_exporter_info.size = npages * PAGE_SIZE;
	dma_buf_exporter_info.flags = O_CLOEXEC;
	dma_buf_exporter_info.resv = NULL;
	dma_buf_exporter_info.priv = data;

	printk("Exporting dma_buf ... Exporting dma_buf ... \n");
	dma_buf = dma_buf_export(&dma_buf_exporter_info);
	
	if (IS_ERR(dma_buf)) {
		printk("Failed to export dma_buf ... \n");
		goto err;
	}
		
	printk("dma_buf export completed ... \n");
	return dma_buf;

err:
	printk("Error handling path ... \n");
	i = data->num_pages;
	while (i--)
		put_page(data->pages[i]);
	kfree(data);
	return ERR_PTR(-ENOMEM);
}

static void dma_buf_exporter_free(struct dma_buf *dma_buf)
{
	struct dma_buf_exporter_data *data = dma_buf->priv;
	int i;

	pr_info("dma_buf_exporter: freeing dma_buf \n");
	for (i = 0; i < data->num_pages; i++)
		put_page(data->pages[i]);

	kfree(data);
}

static long dma_buf_exporter_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	struct dma_exporter_buf_alloc_data data;
	struct dma_buf *dma_buf;

	pr_info("dma_buf_exporter: dma_buf_exporter_ioctl %u \n", cmd);
	if (copy_from_user(&data, (void __user *)arg, sizeof(data))) {
		pr_info("dma_buf_exporter: failed to copy user data. ");
		return -EFAULT;
	}

	switch (cmd) {
		case DMA_BUF_EXPORTER_ALLOC: {
			
			pr_info("dma_buf_exporter: allocating dma_buf of size %llu ", data.size);
			dma_buf = dma_buf_exporter_alloc(data.size);
			if (!dma_buf) {
				pr_err("dma_buf_exporter: ERROR exporter alloc page failed\n");
				return -ENOMEM;
			}
			
			data.fd = dma_buf_fd(dma_buf, O_CLOEXEC);

			if (copy_to_user((void __user *)arg, &data, sizeof(data)))
				return -EFAULT;

			return 0;
		}

		case DMA_BUF_EXPORTER_FREE: {
				pr_info("dma_buf_exporter: freeing dma_buf of size %llu ", data.size);
				dma_buf = dma_buf_get(data.fd);
				
				if (IS_ERR(dma_buf))
					return PTR_ERR(dma_buf);

				dma_buf_exporter_free(dma_buf);
				return 0;
		}

		default: {
			pr_info("dma_buf_exporter: invalid IOCTL code %u ", cmd);
			return -EINVAL;
		}

	}

	return 0;
}

/* This is called whenever a process attempts to open the device file */ 
static int dma_buf_open(struct inode *inode, struct file *file)
{
	pr_info("dma_buf_exporter: dma_buf device open (%p) \n", file);

	__module_get(THIS_MODULE);
	return 0;
}

static int dma_buf_release(struct inode *inode, struct file *file) 
{ 
    pr_info("dma_buf_exporter(%p,%p)\n", inode, file); 
 
    module_put(THIS_MODULE); 
    return 0; 
} 

static struct file_operations dma_buf_exporter_fops = {
	.owner   		= THIS_MODULE,
	.open    		= dma_buf_open,
	.unlocked_ioctl = dma_buf_exporter_ioctl,
	.release 		= dma_buf_release, /* a.k.a. close */ 
};

static struct miscdevice dma_buf_exporter_dev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = DMA_BUF_EXPORTER_DEV_NAME,
	.fops = &dma_buf_exporter_fops,
};
 
static int __init dma_buf_exporter_init(void)
{
	pr_info("dma_buf_exporter: Loading dma_buf_exporter_kmd ...");
	return misc_register(&dma_buf_exporter_dev);
}

static void __exit dma_buf_exporter_exit(void)
{
	pr_info("dma_buf_exporter: Unloading dma_buf_exporter_kmd ...");
	misc_deregister(&dma_buf_exporter_dev);
}


module_init(dma_buf_exporter_init);
module_exit(dma_buf_exporter_exit);

MODULE_VERSION(DRIVER_VERSION);
MODULE_AUTHOR(DRIVER_AUTHOR);
MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_LICENSE("GPL v2");
