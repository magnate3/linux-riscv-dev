
#include<linux/init.h>
#include<linux/module.h>
#include<linux/kernel.h>
#include<linux/vmalloc.h>
#include<linux/slab.h>


void kmalloc_test(void)
{
char * buff;
printk("----------------------------\n");
printk("kmalloc test...\n");
buff=(char *)kmalloc(1024,GFP_KERNEL);
if(buff)
{
sprintf(buff,"test memory\n");
printk(buff);
kfree(buff);
}
else
{
printk("kmalloc failed!\n");
printk("----------------------------\n");
return ;
}


buff=kmalloc(32*PAGE_SIZE,GFP_KERNEL);
if(buff)
{
printk("Big memory ok\n");
kfree(buff);
}
else
{
printk("Big memory molloc failed!\n");
printk("----------------------------\n");
return ;
}
printk("----------------------------\n");
return ;
}
void vmalloc_test(void)
{
char * buff;
printk("----------------------------\n");
printk("vmalloc test...\n");
buff=vmalloc(32*PAGE_SIZE);
if(buff)
{
sprintf(buff,"vmalloc test ok \n");
printk(buff);
vfree(buff);
}
else
{
printk("vmalloc failed!\n");
printk("----------------------------\n");
return ;
}
printk("----------------------------\n");
return ;
}
void get_free_pages_test(void)
{
char * buff;
int order;
printk("get_free_pages test...\n");
order=get_order(8192*10);
buff=__get_free_pages(GFP_KERNEL,order);
if(buff)
{
sprintf(buff,"__get_free_pages test ok [%d]\n",order);
printk(buff);
free_pages(buff,order);
}
else
{
printk("__get_free_pages failed !\n");
printk("----------------------------\n");
return ;
}
printk("----------------------------\n");
return ;
}
int memtest_init(void)
{
printk("Module Memory Test\n");
printk("The PAGE_SIZE=%d\n",PAGE_SIZE);
kmalloc_test();
vmalloc_test();
get_free_pages_test();
return 0;
}
void memtest_exit(void)
{
printk("Module Memory Test End\n");
}
module_init(memtest_init);
module_exit(memtest_exit);
MODULE_LICENSE("Dual BSD/GPL");