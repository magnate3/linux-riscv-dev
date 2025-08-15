/*
* Write by Vincent.wan@amd.com
*/

#include <linux/module.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/device.h>

#define simple_MAJOR 201
#define SIMP_BLKDEV_DEVICEMAJOR 220
#define SIMP_BLKDEV_BYTES (4*1024*1024)

static struct class *mem_class;
struct device *pri_dev;
struct block_device *simplebdev;
char *addr_write;
char *addr_read;
extern int read_bio_page(struct block_device *simplebdev,
				pgoff_t page_off, void *addr, int len);
extern int write_bio_page(struct block_device *simplebdev,
				pgoff_t page_off, void *addr, int len);
extern unsigned int max_vecs;

static const struct file_operations simple_fops={
	.owner=THIS_MODULE,
};

static void show_read_buffer(void) {
	int i, val;

	printk("\nread buffer:\n");

	for (i = 0; i < 40; i++) {
		val = addr_read[i];
		
		printk("0x%x ,", val);

		if((i+1) % 10 == 0)
			printk("\n");
	}
}

static ssize_t
write_blk_show(struct device *dev, struct device_attribute *devattr, char *buf)
{
	printk("TESTDRIVER:write_blk_show!\n");
	return 0;
}

static ssize_t
write_blk(struct device *dev, struct device_attribute *devattr,
					const char *buf, size_t count)
{
	unsigned long val;

	if (kstrtoul(buf, 16, &val))
		return -EINVAL;
	
	write_bio_page(simplebdev, 0, addr_write, max_vecs*PAGE_SIZE);

	return count;
}

static ssize_t
read_blk(struct device *dev, struct device_attribute *attr, char *buf)
{
	unsigned int val;
	val = 0x1;

	read_bio_page(simplebdev, 0, addr_read, max_vecs*PAGE_SIZE);

	show_read_buffer();

	return sprintf(buf, "0x%x\n", val);
}

static DEVICE_ATTR(readblk, S_IRUGO | S_IRUSR, read_blk, NULL);
static DEVICE_ATTR(writeblk, S_IRUGO | S_IWUSR, write_blk_show, write_blk);

void testdriver_cleanup_module(void){

	kfree(addr_write);
	kfree(addr_read);

	device_remove_file(pri_dev, &dev_attr_readblk);
	device_remove_file(pri_dev, &dev_attr_writeblk);

	device_destroy(mem_class, MKDEV(simple_MAJOR, 0));

	class_destroy(mem_class);

	unregister_chrdev(simple_MAJOR,"testdriver");

	printk("TESTDRIVER: testdriver_cleanup_module!\n");
}

int testdriver_init_module(void) {

	int ret;

	addr_write = kmalloc(1024*1024, GFP_KERNEL);
	addr_read = kmalloc(1024*1024, GFP_KERNEL);

	memset(addr_read, 0x0, 1024*1024);
	memset(addr_write, 0x66, 1024*1024);

	ret = register_chrdev(simple_MAJOR,"testdriver",&simple_fops);
	if(ret<0){
		printk("TESTDRIVER: Unable to register device %d!\n",simple_MAJOR);
		return ret;
	}

	printk("TESTDRIVER:Ok to register testdriver device %d!\n",simple_MAJOR);

	mem_class = class_create(THIS_MODULE, "testdriver");
	if (IS_ERR(mem_class))
		return PTR_ERR(mem_class);

	pri_dev = device_create(mem_class, NULL, MKDEV(simple_MAJOR, 0),
			      NULL, "testdriver0");

	device_create_file(pri_dev, &dev_attr_readblk);
	device_create_file(pri_dev, &dev_attr_writeblk);

	simplebdev = blkdev_get_by_dev(MKDEV(SIMP_BLKDEV_DEVICEMAJOR, 0),
					    FMODE_READ | FMODE_WRITE, NULL);
	return 0;
}

module_init(testdriver_init_module);
module_exit(testdriver_cleanup_module);

MODULE_LICENSE("GPL");
