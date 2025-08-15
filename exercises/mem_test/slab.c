#include <linux/module.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mm.h>

static struct kmem_cache* slub_test;

struct student{
    int age;
    int score;
};

static void mystruct_constructor(void *addr)
{
    memset(addr, 0, sizeof(struct student));
}

struct student* peter;

int slub_test_create_kmem(void)
{
    int ret = -1;
    slub_test = kmem_cache_create("slub_test", sizeof(struct student), 0, 0, mystruct_constructor);
    if(slub_test != NULL){
        printk("slub_test create success!\n");
        ret=0;
    }


    peter = kmem_cache_alloc(slub_test, GFP_KERNEL);
    if(peter != NULL){
        printk("alloc object success!\n");
        ret = 0;
    }

    return ret;
}

static int __init slub_test_init(void)
{
    int ret;
    printk("slub_test kernel module init\n");
    ret = slub_test_create_kmem();
    return 0;
}

static void __exit slub_test_exit(void)
{
    printk("slub_test kernel module exit\n");
    kmem_cache_destroy(slub_test);
}

module_init(slub_test_init);
module_exit(slub_test_exit);