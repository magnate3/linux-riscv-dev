#include <linux/module.h>
#include <linux/slab.h>
#include <linux/device.h>
#include <asm/io.h>
#include "dlmalloc.h"
#define   MALLOC_SIZE 1024 
#define   DLMALLOC_SIZE  512
#define  SRAM_BASE  0x80000200000
#define SRAM_SIZE 0xfffffff
#if 0
static int __init dlmalloc_init(void)
{
        //char * base = (char*)kzalloc(MALLOC_SIZE, GFP_KERNEL);
        char * base =  (char*)ioremap(SRAM_BASE, SRAM_SIZE);
        printk(KERN_INFO "base addr %p \n", base);
        char *p;
        p=create_mspace_with_base((void *)base,SRAM_SIZE,0);
        printk(KERN_INFO "dlmalloc addr %p \n", p);
        //destroy_mspace(p);
        iounmap(base);
        return 0;
}
#else
static int __init dlmalloc_init(void)
{
        char * base = (char*)kzalloc(sizeof(MALLOC_SIZE), GFP_KERNEL);
        printk(KERN_INFO "base addr %p \n", base);
        char *p;
        p=create_mspace_with_base((void *)base,MALLOC_SIZE,0);
        printk(KERN_INFO "dlmalloc addr %p \n", p);
        destroy_mspace(p);
        kfree(base);
        return 0;
}
#endif
static void __exit dlmalloc_exit(void)
{
        printk(KERN_INFO "dlmalloc module exit\n");
}

module_init(dlmalloc_init);
module_exit(dlmalloc_exit);

MODULE_AUTHOR("Jerry Cooperstein");
MODULE_DESCRIPTION("LF331:1.6 s_18/lab8_uio_api.c");
MODULE_LICENSE("GPL v2");
