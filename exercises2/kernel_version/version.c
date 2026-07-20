#include <linux/module.h>
#include <linux/init.h>
#include <linux/version.h>
 
 
#define DEBUG_CT(format, ...) printk("%s:%d "format"\n",\
	__func__,__LINE__,##__VA_ARGS__)
 
static int __init ct_init(void)
{
#if KERNEL_VERSION(6,3,2) == LINUX_VERSION_CODE
	DEBUG_CT("new version 5");
#else
	DEBUG_CT("old version4");
#endif
	DEBUG_CT("KERNEL_VERSION(5,15,71) = %d",KERNEL_VERSION(5,15,71));
	DEBUG_CT("KERNEL_VERSION(4,1,15) = %d",KERNEL_VERSION(4,1,15));
 
	
	return 0;
}
static void __exit ct_exit(void)
{
	DEBUG_CT("bye bye");
}
 
module_init(ct_init);
module_exit(ct_exit);
MODULE_LICENSE("GPL");
