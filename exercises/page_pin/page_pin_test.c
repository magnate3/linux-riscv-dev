#include <linux/cdev.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include <linux/xarray.h>
 
/*
 * All thread in above process can pin/unpin va related pa by:
 * ioctl(fd, TEST_PIN/TEST_UNPIN, addr); addr is pointer of struct test_pin_address.
 *
 * addr and size in struct test_pin_address are pin/unpin address and size.
 */
//struct test_pin_address {
//	unsigned long addr;
//	unsigned long size;
//};
//
//#define TEST_PIN		_IOW('W', 0, struct test_pin_address)
//#define TEST_UNPIN		_IOW('W', 1, struct test_pin_address)
//#define MAX_PIN_PAGE		100
#include "pin.h"
MODULE_LICENSE("Dual BSD/GPL");

struct cdev cdev;
struct class *class;
/* device's memory pool */
struct file_priv {
	struct xarray array;
};

struct pin_pages {
	unsigned long first;
	unsigned long nr_pages;
	struct page **pages;
};

int test_open(struct inode *inode, struct file *file)
{
	struct file_priv *p;

	p = kzalloc(sizeof(*p), GFP_KERNEL);
	if (!p)
		return -ENOMEM;
        file->private_data = p;

	xa_init(&p->array);

        return 0;
}

int test_release(struct inode *inode, struct file *file)
{
	struct file_priv *priv = file->private_data;
	struct pin_pages *p;
	unsigned long idx;

	xa_for_each(&priv->array, idx, p) {
		unpin_user_pages(p->pages, p->nr_pages);
		xa_erase(&priv->array, p->first);
		kfree(p->pages);
		kfree(p);
	}
	
	xa_destroy(&priv->array);
	kfree(priv);

        return 0;
}

static int test_pin_page(struct file_priv *priv, struct test_pin_address *addr)
{
	unsigned int flags = FOLL_FORCE | FOLL_WRITE;
	unsigned long first, last, nr_pages;
	struct page **pages;
	struct pin_pages *p;
	int ret;

	first = (addr->addr & PAGE_MASK) >> PAGE_SHIFT;
	last = ((addr->addr + addr->size - 1) & PAGE_MASK) >> PAGE_SHIFT;
	nr_pages = last - first + 1;

	pages = kmalloc_array(nr_pages, sizeof(struct page *), GFP_KERNEL);
	if (!pages)
		return -ENOMEM;

	p = kzalloc(sizeof(struct pin_pages), GFP_KERNEL);
	if (!p) {
		ret = -ENOMEM;
		goto free;
	}

	/* todo: check to avoid double pin */

	ret = pin_user_pages(addr->addr & PAGE_MASK, nr_pages,
			     flags | FOLL_LONGTERM, pages, NULL);
	if (ret != nr_pages) {
		pr_err("Failed to pin page\n");
		goto free_p;
	}
	p->first = first;
	p->nr_pages = nr_pages;
	p->pages = pages;

	ret = xa_err(xa_store(&priv->array, p->first, p, GFP_KERNEL));
	if (ret)
		goto unpin_pages;

	return 0;

unpin_pages:
	unpin_user_pages(pages, nr_pages);
free_p:
	kfree(p);
free:
	kfree(pages);
	return ret;
}

static int test_unpin_page(struct file_priv *priv,
			   struct test_pin_address *addr)
{
	unsigned long first, last, nr_pages;
	struct pin_pages *p;

	first = (addr->addr & PAGE_MASK) >> PAGE_SHIFT;
	last = ((addr->addr + addr->size - 1) & PAGE_MASK) >> PAGE_SHIFT;
	nr_pages = last - first + 1;

	/* find pin_pages */
	p = xa_load(&priv->array, first);
	if (!p)
		return -ENODEV;

	if (p->nr_pages != nr_pages)
		return -EINVAL;

	/* unpin */
	unpin_user_pages(p->pages, p->nr_pages);

	/* release resource */
	xa_erase(&priv->array, first);
	kfree(p->pages);
	kfree(p);
	
	return 0;
}

static int test_check_param(struct test_pin_address *addr)
{
	if (addr->size > MAX_PIN_PAGE * PAGE_SIZE)
		return -EINVAL;
	
	return 0;
}

static long test_unl_ioctl(struct file *filep, unsigned int cmd,
			   unsigned long arg)
{
	struct file_priv *p = filep->private_data;
	struct test_pin_address addr;
	int ret;

        pr_info("************* pin ioctl, cmd %u, TEST_PIN: %lu, TEST_UNPIN: %lu \n", cmd, TEST_PIN, TEST_UNPIN);
	if (copy_from_user(&addr, (void __user *)arg,
			   sizeof(struct test_pin_address)))
		return -EFAULT;

	ret = test_check_param(&addr);
	if (ret) {
		pr_err("Invalid input\n");
		return -EINVAL;
	}

	switch (cmd) {
	case TEST_PIN:
		pr_info("************* pin cmd\n");
		return test_pin_page(p, &addr);

	case TEST_UNPIN:
		return test_unpin_page(p, &addr);

	default:
		return -EINVAL;
	}
}

struct file_operations test_fops = {
        .owner = THIS_MODULE,
        .open = test_open,
        .release = test_release,
	.unlocked_ioctl	= test_unl_ioctl,
};

static int __init page_pin_init(void)
{
        unsigned int firstminor = 0;
        dev_t dev_id;
        int err = 0;
        unsigned int count = 1;
        char *dev_name = "page_pin_test";

        /* alloc dev_id */
        err = alloc_chrdev_region(&dev_id, firstminor, count, dev_name);
        if (err < 0) {
                pr_err("page_pin: can not allocate a cdev\n");
                return err;
        }

        /* register cdev */
        cdev_init(&cdev, &test_fops);
        err = cdev_add(&cdev, dev_id, count);
        if (err < 0) {
                pr_err("page_pin: can not add a cdev to system\n");
                return err;
        }

        class = class_create(THIS_MODULE, dev_name);
        if (IS_ERR(class)) {
                pr_err("page_pin: fail to create device %s\n", dev_name);
                return -1;
        }

        device_create(class, NULL, dev_id, NULL, "page_pin");

        return 0;
}

static void __exit page_pin_exit(void)
{
        dev_t dev_id = cdev.dev;

        device_destroy(class, dev_id);
        class_destroy(class);
        cdev_del(&cdev);
}

module_init(page_pin_init);
module_exit(page_pin_exit);

MODULE_AUTHOR("Sherlock");
MODULE_DESCRIPTION("The driver is for testing page pin");

//make -C /home/wangzhou/linux-kernel-warpdrive M=/home/wangzhou/tests/page_pin modules
