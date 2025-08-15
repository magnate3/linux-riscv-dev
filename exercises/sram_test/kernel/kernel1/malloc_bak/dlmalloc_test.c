#include <linux/module.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <linux/uio_driver.h>
#define   MALLOC_SIZE 1024 
static int __init dlmalloc_init(void)
{
        char * base = (char*)kzalloc(sizeof(MALLOC_SIZE), GFP_KERNEL);
        kfree(base);
        return 0;
}

static void __exit dlmalloc_exit(void)
{
        printk(KERN_INFO "dlmalloc module exit\n");
}

module_init(dlmalloc_init);
module_exit(dlmalloc_exit);

MODULE_AUTHOR("Jerry Cooperstein");
MODULE_DESCRIPTION("LF331:1.6 s_18/lab8_uio_api.c");
MODULE_LICENSE("GPL v2");
