
#include <linux/module.h>    // included for all kernel modules
#include <linux/kernel.h>    // included for KERN_INFO
#include <linux/init.h>      // included for __init and __exit macros
#include <linux/interrupt.h> // included for request_irq and free_irq macros
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Rong Tao");
MODULE_DESCRIPTION("A Simple request irq module");
MODULE_VERSION("0.1");
 
static char *name = "[RToax]";
module_param( name, charp, S_IRUGO);
MODULE_PARM_DESC(name, "[RToax] irq name");	
 
/**
 *  cat  /proc/interrupts |awk -F ":" '{print $1}' 中没有的 中断号
 */
#define IRQ_NUM 2
 
irqreturn_t no_action(int cpl, void *dev_id)
{
    printk(KERN_INFO "[RToax]cpl %d!\n", cpl);
	return IRQ_NONE;
}
 
 
static int __init rtoax_irq_init(void) {
    
	printk(KERN_INFO "[RToax]request irq %s!\n", name);
    /*
     *  注册中断
     */
    if (request_irq(IRQ_NUM, no_action, IRQF_NO_THREAD, name, NULL))
	    printk(KERN_ERR "%s: request_irq() failed\n", name);
	return 0;
}
 
static void __exit rtoax_irq_cleanup(void) {
	printk(KERN_INFO "[RToax]free irq.\n");
    /*
     *  释放中断
     */
    free_irq(IRQ_NUM, NULL);
}
 
module_init(rtoax_irq_init);
module_exit(rtoax_irq_cleanup);
