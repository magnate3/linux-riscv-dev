#include<linux/init.h>
#include<linux/kernel.h>
#include<linux/module.h>
#include<linux/jiffies.h>
#include<asm/param.h>


/* This function is called when the module is loaded. */


int uptime_init(void)
{
long jf = jiffies-INITIAL_JIFFIES;	
printk(KERN_INFO "System Up-Time: %ld Hours %ld Minutes %ld Seconds\n", jf/HZ/60/60,jf/HZ/60,jf/HZ%60);

return 0;
}
/* This function is called when the module is removed. */
void uptime_exit(void)
{
printk(KERN_INFO "Removing Kernel Module \n");
}
/* Macros for registering module entry and exit points. */
module_init(uptime_init);
module_exit(uptime_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("uptime module");
MODULE_AUTHOR("MR-EIGHT");
