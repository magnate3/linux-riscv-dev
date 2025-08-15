/*
 * Author - Jared_Wu@moxa.com.tw
 * sram.c - SRAM disk driver - v1.0.
 *
 * This SRAM disk is designed to read/write the 256 Kbytes SRAM device
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/kdev_t.h>
#include <linux/miscdevice.h>
#include <linux/init.h>
#include <linux/errno.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/ioport.h>

#include <linux/uaccess.h>
#include <linux/slab.h>

#define SRAM_BASE	0x52000000
#define SRAM_SIZE	0x00040000  // 256 Kbytes

#define MOXA_SRAM_MINOR		107

/* Various static variables go here.  Most are used only in the RAM disk code.
 */

static struct resource		*sram_res=NULL;
static volatile unsigned char	*sram_addr=NULL;
static struct semaphore sram_mutex;

static ssize_t sram_read(struct file *file, char __user *buf,
                           size_t count, loff_t *ppos)
{
	if (count > SRAM_SIZE)
		count = SRAM_SIZE;

	if (down_interruptible (&sram_mutex))
		return -EINTR;

	count=copy_to_user ((void*)buf, (void*)sram_addr, count);
	if ( count < 0 )
		return(-EFAULT);

	up(&sram_mutex);

	return count;
}

static ssize_t sram_write(struct file * file, const char __user * buf,
		        size_t count, loff_t *ppos)
{
	size_t copy_size = count;

	if (copy_size > SRAM_SIZE)
		copy_size = SRAM_SIZE;

	if (down_interruptible (&sram_mutex))
		return -EINTR;

	copy_size=copy_from_user ((void*)sram_addr, (void*)buf, copy_size);
	if ( copy_size < 0 ) {
		printk("Copy from user space fail: size:%d\n", copy_size);
		return -EFAULT;
	}

	up(&sram_mutex);

	return copy_size;
}

static int sram_release(struct inode *inode, struct file *file)
{
	module_put(THIS_MODULE);

	return 0;
}

static int sram_open(struct inode *inode, struct file *filp)
{
	if(!try_module_get(THIS_MODULE))
	  return -ENODEV;

	return 0;
}

static struct file_operations sram_fops = {
	.owner =	THIS_MODULE,
	.open =		sram_open,
	.read =		sram_read,
	.write =	sram_write,
	.release =	sram_release,
};

static struct miscdevice sram_dev = {
	MOXA_SRAM_MINOR,
	"sram",
	&sram_fops,
};

static void __exit sram_cleanup(void)
{
	printk("Unregistering Moxa SRAM driver\n");

	if(sram_addr)
		iounmap( (unsigned char*)sram_addr );

	if ( sram_res ) {
		release_resource(sram_res);
		kfree(sram_res);
	}

	misc_deregister(&sram_dev);
}

/*
 * This is the registration and initialization section of the RAM disk driver
 */
static int __init sram_init(void)
{
	printk("Register Moxa SRAM driver v.1.0\n");

	sram_res = request_mem_region(SRAM_BASE, SRAM_SIZE, "moxa_sram");
	if ( sram_res == NULL ) {
		printk("Moxa SRAM resease mem region fail !\n");
		return -ENOMEM;
	}

	sram_addr = ioremap(SRAM_BASE, SRAM_SIZE);
	if( !sram_addr) {
		printk("Fail to map SRAM_BASE:%x\n", (unsigned int)sram_addr);
		release_resource(sram_res);
		kfree(sram_res);
		return -ENOMEM;
	}

	sema_init (&sram_mutex, 1);

	if ( misc_register(&sram_dev) ) {
		printk("Moxa SRAM driver fail !\n");
		return -EIO;
	}

	return 0;
}

module_init(sram_init);
module_exit(sram_cleanup);
MODULE_LICENSE("GPL");