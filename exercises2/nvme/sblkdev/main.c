// SPDX-License-Identifier: GPL-2.0

#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include <linux/module.h>
#include "device.h"

#ifdef CONFIG_SBLKDEV_REQUESTS_BASED
#pragma message("Request-based scheme selected.")
#else
#pragma message("Bio-based scheme selected.")
#endif

#ifdef CONFIG_SBLKDEV_BLOCK_SIZE
#pragma message("Specific block size selected.")
#endif

#ifdef HAVE_BI_BDEV
#pragma message("The struct bio have pointer to struct block_device.")
#endif
#ifdef HAVE_BI_BDISK
#pragma message("The struct bio have pointer to struct gendisk.")
#endif
#ifdef HAVE_BLK_MQ_ALLOC_DISK
#pragma message("The blk_mq_alloc_disk() function was found.")
#endif
#ifdef HAVE_ADD_DISK_RESULT
#pragma message("The function add_disk() has a return code.")
#endif
#ifdef HAVE_BDEV_BIO_ALLOC
#pragma message("The function bio_alloc_bioset() has a parameter bdev.")
#endif
#ifdef HAVE_BLK_CLEANUP_DISK
#pragma message("The function blk_cleanup_disk() was found.")
#endif
#ifdef HAVE_GENHD_H
#pragma message("The header file 'genhd.h' was found.")
#endif

/*
 * A module can create more than one block device.
 * The configuration of block devices is implemented in the simplest way:
 * using the module parameter, which is passed when the module is loaded.
 * Example:
 *    modprobe sblkdev catalog="sblkdev1,2048;sblkdev2,4096"
 */

static int sblkdev_major;
static LIST_HEAD(sblkdev_device_list);
static char *sblkdev_catalog = "sblkdev1,2048;sblkdev2,4096";

/*
 * sblkdev_init() - Entry point 'init'.
 *
 * Executed when the module is loaded. Parses the catalog parameter and
 * creates block devices.
 */
static int __init sblkdev_init(void)
{
	int ret = 0;
	int inx = 0;
	char *catalog;
	char *next_token;
	char *token;
	size_t length;

	sblkdev_major = register_blkdev(sblkdev_major, KBUILD_MODNAME);
	if (sblkdev_major <= 0) {
		pr_info("Unable to get major number\n");
		return -EBUSY;
	}

	length = strlen(sblkdev_catalog);
	if ((length < 1) || (length > PAGE_SIZE)) {
		pr_info("Invalid module parameter 'catalog'\n");
		ret = -EINVAL;
		goto fail_unregister;
	}

	catalog = kzalloc(length + 1, GFP_KERNEL);
	if (!catalog) {
		ret = -ENOMEM;
		goto fail_unregister;
	}
	strcpy(catalog, sblkdev_catalog);

	next_token = catalog;
	while ((token = strsep(&next_token, ";"))) {
		struct sblkdev_device *dev;
		char *name;
		char *capacity;
		sector_t capacity_value;

		name = strsep(&token, ",");
		if (!name)
			continue;
		capacity = strsep(&token, ",");
		if (!capacity)
			continue;

		ret = kstrtoull(capacity, 10, &capacity_value);
		if (ret)
			break;

		dev = sblkdev_add(sblkdev_major, inx, name, capacity_value);
		if (IS_ERR(dev)) {
			ret = PTR_ERR(dev);
			break;
		}

		list_add(&dev->link, &sblkdev_device_list);
		inx++;
	}
	kfree(catalog);

	if (ret == 0)
		return 0;

fail_unregister:
	unregister_blkdev(sblkdev_major, KBUILD_MODNAME);
	return ret;
}

/*
 * sblkdev_exit() - Entry point 'exit'.
 *
 * Executed when the module is unloaded. Remove all block devices and cleanup
 * all resources.
 */
static void __exit sblkdev_exit(void)
{
	struct sblkdev_device *dev;

	while ((dev = list_first_entry_or_null(&sblkdev_device_list,
					       struct sblkdev_device, link))) {
		list_del(&dev->link);
		sblkdev_remove(dev);
	}

	if (sblkdev_major > 0)
		unregister_blkdev(sblkdev_major, KBUILD_MODNAME);
}

module_init(sblkdev_init);
module_exit(sblkdev_exit);

module_param_named(catalog, sblkdev_catalog, charp, 0644);
MODULE_PARM_DESC(catalog, "New block devices catalog in format '<name>,<capacity sectors>;...'");

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Sergei Shtepa");
