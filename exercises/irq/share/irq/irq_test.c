
#include <linux/module.h>    // included for all kernel modules
#include <linux/kernel.h>    // included for KERN_INFO
#include <linux/init.h>      // included for __init and __exit macros
#include <linux/interrupt.h> // included for request_irq and free_irq macros
 
MODULE_LICENSE("GPL");
MODULE_AUTHOR("yun");
MODULE_DESCRIPTION("A Simple request irq module");
MODULE_VERSION("0.1");
 
static char *name = "[RToax]";
module_param( name, charp, S_IRUGO);
MODULE_PARM_DESC(name, "[RToax] irq name");	
 
/**
 *  enp5s0中断号
 */
#define IRQ_NUM 265
bool succ = false ;
int  data;
irqreturn_t no_action(int cpl, void *dev_id)
{
    printk(KERN_INFO "share interrupt [%d] happens !\n", cpl);
	return IRQ_NONE;
}
 
 
static int __init rtoax_irq_init(void) {
    
        int virq, err;
	printk(KERN_INFO "[RToax]request irq %s!\n", name);
        //virq = irq_find_mapping(NULL, hwirq);
    /*
     *  注册中断
     */
       int index = 0;
    for(; index < 1; ++ index)
    {
    if (err = request_irq(IRQ_NUM + index, no_action, IRQF_SHARED|IRQF_NO_THREAD, name, &data))
    {
	    printk(KERN_ERR "%s: request_irq(%d) failed: %d \n", name, IRQ_NUM, err);
            return 0;
    }
    }
    succ = true;
    return 0;
}
 
static void __exit rtoax_irq_cleanup(void) {
	printk(KERN_INFO "[RToax]free irq.\n");
    /*
     *  释放中断
     */
    // not free irq will coredump
    if (succ)
        free_irq(IRQ_NUM, &data);
}
 
module_init(rtoax_irq_init);
module_exit(rtoax_irq_cleanup);
