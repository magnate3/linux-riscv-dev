
#include <linux/module.h>    // included for all kernel modules
#include <linux/kernel.h>    // included for KERN_INFO
#include <linux/init.h>      // included for __init and __exit macros
#include <linux/interrupt.h> // included for request_irq and free_irq macros
#include <linux/irqdomain.h> 
#include <linux/irq.h>
#include <linux/irqdesc.h>
#include <linux/device.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_irq.h> 
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
int  data = 999;
//irq_debug_show_data(struct seq_file *m, struct irq_data *data, int ind)
irqreturn_t no_action(int cpl, void *dev_id)
{
#if 0
    printk(KERN_INFO "share interrupt [%d] happens !\n", cpl);
#endif
	return IRQ_NONE;
}
struct irqchip_fwid {
	struct fwnode_handle fwnode;
	char *name;
	void *data;
}; 
void irq_chip_parent(struct irq_data *data)
{
        struct irq_chip *chip ; 
        struct device_node *of_node;
        const char *name;
        struct irq_domain *domain;
        if (!data)
        {
               return ;
        }
        data = data->parent_data;
        if(!data)
        {
               return ;
        }
        chip = data->chip;
        pr_err(" parent chip->name %s  \t", chip->name);
        domain = data->domain;
        if (domain)
        {
#if 0
                    of_node = irq_domain_get_of_node(domain);
                    if (of_node)
                            name = of_node_full_name(of_node);
                    else if (is_fwnode_irqchip(domain->fwnode))
                            name = container_of(domain->fwnode, struct irqchip_fwid,
                                                fwnode)->name;
                    else
                            name = "";
                    pr_err("parent domain name :  %s, name %s \t", domain->name , name);
#else
                    pr_err("parent domain name :  %s , domain->ops->alloc： %p, domain->host_data %p \t", domain->name, domain->ops->alloc, domain->host_data );
#endif
        }
        irq_chip_parent(data);
} 
static void virq_debug_show_one(struct irq_desc *desc)
{
        struct irq_domain *domain;
        struct irq_data *data;

        domain = desc->irq_data.domain;
        data = &desc->irq_data;

        while (domain) {
                unsigned int irq = data->irq;
                unsigned long hwirq = data->hwirq;
                struct irq_chip *chip;
                bool direct;

                if (data == &desc->irq_data)
                        pr_err("%5d  ", irq);
                else
                        pr_err("%5d+ ", irq);
                pr_err("0x%05lx  ", hwirq);

                chip = irq_data_get_irq_chip(data);
                pr_err( "%-15s  ", (chip && chip->name) ? chip->name : "none");

                //pr_err(data ? "0x%p  " : "  %p  ",
                //           irq_data_get_irq_chip_data(data));

                pr_err("   %c    ", (desc->action && desc->action->handler) ? '*' : ' ');
                direct = (irq == hwirq) && (irq < domain->revmap_direct_max_irq);
                pr_err("%6s%-8s  ",
                           (hwirq < domain->revmap_size) ? "LINEAR" : "RADIX",
                           direct ? "(DIRECT)" : "");
                pr_err("%s\n", domain->name);
#if 1
                domain = domain->parent;
                data = data->parent_data;
#else
                domain = NULL;
#endif
        }
}
#if 0
static int virq_debug_show()
{
        unsigned long flags;
        struct irq_desc *desc;
        struct irq_domain *domain;
        struct radix_tree_iter iter;
        void __rcu **slot;
        int i;

        pr_err(" %-16s  %-6s  %-10s  %-10s  %s\n",
                   "name", "mapped", "linear-max", "direct-max", "devtree-node");
        //mutex_lock(&irq_domain_mutex);
        list_for_each_entry(domain, &irq_domain_list, link) {
                struct device_node *of_node;
                const char *name;

                int count = 0;

                of_node = irq_domain_get_of_node(domain);
                if (of_node)
                        name = of_node_full_name(of_node);
                else if (is_fwnode_irqchip(domain->fwnode))
                        name = container_of(domain->fwnode, struct irqchip_fwid,
                                            fwnode)->name;
                else
                        name = "";

                radix_tree_for_each_slot(slot, &domain->revmap_tree, &iter, 0)
                        count++;
                pr_err("%c%-16s  %6u  %10u  %10u  %s\n",
                           domain == irq_default_domain ? '*' : ' ', domain->name,
                           domain->revmap_size + count, domain->revmap_size,
                           domain->revmap_direct_max_irq,
                           name);
        }
        //mutex_unlock(&irq_domain_mutex);

        pr_err("%-5s  %-7s  %-15s  %-*s  %6s  %-14s  %s\n", "irq", "hwirq",
                      "chip name", (int)(2 * sizeof(void *) + 2), "chip data",
                      "active", "type", "domain");

        for (i = 1; i < nr_irqs; i++) {
                desc = irq_to_desc(i);
                if (!desc)
                        continue;

                raw_spin_lock_irqsave(&desc->lock, flags);
                virq_debug_show_one(m, desc);
                raw_spin_unlock_irqrestore(&desc->lock, flags);
        }

        return 0;
}
#endif 
static int __init rtoax_irq_init(void) {
    
     int virq, err;
     printk(KERN_INFO "[RToax]request irq %s!\n", name);
     struct irq_desc *desc;
     struct irqaction *action, **action_ptr;
     int irq = IRQ_NUM;
     struct irq_chip *chip ;

     struct irq_data *i_data;
     struct irq_domain * domain;
        //virq = irq_find_mapping(NULL, hwirq);
    /*
     *  注册中断
     */
       int index = 0;
    for(; index < 1; ++ index)
    {
    if (err = request_irq(irq + index, no_action, IRQF_SHARED|IRQF_NO_THREAD, name, &data))
    {
	    printk(KERN_ERR "%s: request_irq(%d) failed: %d \n", name, IRQ_NUM, err);
            return 0;
    }
    }
    desc = irq_to_desc(irq);
    chip = irq_desc_get_chip(desc);
    i_data = & desc->irq_data;
    pr_err("irq info oupt begin ************************ \n virq:  %d, hwirq: %ld ,desc->depth %d, desc->handle_irq %x \t ",  irq, desc->irq_data.hwirq, desc->depth, desc->handle_irq);
    //struct msi_domain_info *info = domain->host_data;
    //struct msi_domain_ops *ops = info->ops;
    //irq_hw_number_t hwirq = ops->get_hwirq(info, arg);
    //if (desc->irq_data.chip)
    if (chip)
    {
         pr_err(" leaf chip->name %s  \t", chip->name);
         domain = i_data->domain;
         if (domain) {
         pr_err("leaf domain name :  %s, and domain->ops->alloc %p, domain->host_data %p \t", domain->name , domain->ops->alloc, domain->host_data);
        } 
         if (0 == desc->depth)
         {
                
               irq_chip_parent(&desc->irq_data);
         }
    }
    pr_err("\n");
    virq_debug_show_one(desc);
    /*
 *          * There can be multiple actions per IRQ descriptor, find the right
 *                   * one based on the dev_id:
 *                            */
   action_ptr = &desc->action;
   for (;;) {
           action = *action_ptr;

           if (!action) {
                   pr_err(" IRQ %d action end \n", irq);
                   break;
           }

           if (action->dev_id == &data) {
                   pr_err(" IRQ %d dev_id find \n", irq);
                   break;
           }
           action_ptr = &action->next;
    }
    succ = true;
    return 0;
}
 
static void __exit rtoax_irq_cleanup(void) {
	printk(KERN_INFO "[RToax]free irq.\n");
    /*
     *  释放中断
     */
    // free irq will coredump
    if (succ)
         free_irq(IRQ_NUM, &data);
}
 
module_init(rtoax_irq_init);
module_exit(rtoax_irq_cleanup);
