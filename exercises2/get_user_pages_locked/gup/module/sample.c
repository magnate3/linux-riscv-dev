/* GUP example tested in 4.19.75 */

#include <linux/device.h>
#include <linux/module.h>
#include <linux/pagemap.h>

static struct class *sample_class;
static int ret_reg_dev;
static struct device *device;

static int sample_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);
}

static int sample_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);
}

void page_map_write(const char __user *addr, const char *str)
{
        int res;
        struct page *page[1];
        char *myaddr;
        printk(KERN_INFO "%s\n", __FUNCTION__);

        // pin 1 user page in memory
        res = get_user_pages_fast((unsigned long)addr, // start of user address
                                  1, // number of pages to get
                                  1, // write to pages or not
                                  page); // pages returned

        // get_user_pages should be used
        if (res > 0) {
                // page to kernel space address
                myaddr = kmap(page[0]);
                if (myaddr) {
                        printk(KERN_INFO "mmaped user page.\n");
                        strcpy(myaddr, str);
                        set_page_dirty(page[0]);
                }
                // release the pinned page
                put_page(page[0]);
        }
}

static ssize_t sample_read(struct file *file, char __user *buf, size_t count,
                           loff_t *off)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        page_map_write(buf, "CALLED_READ");
        return 0;
}

static ssize_t sample_write(struct file *file, const char __user *buf,
                            size_t count, loff_t *off)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        page_map_write(buf, "CALLED_WRITE");
        return 0;
}

static struct file_operations sample_ops = { .owner = THIS_MODULE,
                                             .open = sample_open,
                                             .release = sample_release,
                                             .write = sample_write,
                                             .read = sample_read };

static int __init sample_init(void)
{
        printk("Loaded sample module\n");

        ret_reg_dev = register_chrdev(42, "sample", &sample_ops);
        if (ret_reg_dev) {
                printk("Error on register_chrdev\n");
                return ret_reg_dev;
        }
        sample_class = class_create(THIS_MODULE, "sample");

        if (IS_ERR(sample_class)) {
                printk("Error on class_create\n");
                return PTR_ERR(sample_class);
        }
        device =
                device_create(sample_class, NULL, MKDEV(42, 0), NULL, "sample");

        if (IS_ERR(device)) {
                printk("Error on device_create\n");
                return PTR_ERR(device);
        }

        return 0;
}

static void __exit sample_exit(void)
{
        printk("Unloaded sample module\n");

        if (!IS_ERR(device)) {
                device_destroy(sample_class, MKDEV(42, 0));
        }

        if (!IS_ERR(sample_class)) {
                class_destroy(sample_class);
        }

        if (!ret_reg_dev) {
                unregister_chrdev(42, "sample");
        }
}

module_init(sample_init);
module_exit(sample_exit);

MODULE_LICENSE("GPL");
