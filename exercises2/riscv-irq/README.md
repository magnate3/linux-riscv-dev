
[linux下玄铁处理器的中断处理](https://zhuanlan.zhihu.com/p/672610817)  



# generic_handle_domain_irq

```
int generic_handle_domain_irq(struct irq_domain *domain, unsigned int hwirq)
{
        struct irq_desc *my= irq_resolve_mapping(domain, hwirq);
        if(0x3d == hwirq)
        {
                dump_stack();
                printk("[nfk test]hwirq=%u\n",hwirq);
                printk("[nfk test] irq =%d,hwirq=%d",my->irq_data.irq,my->irq_data.hwirq);
                printk("[nfk test] name =%s",my->dev_name);
        }
        return handle_irq_desc(irq_resolve_mapping(domain, hwirq));
}
  
```


```
[   92.421173][    C0] CPU: 0 PID: 0 Comm: swapper/0 Not tainted 5.14.12 #9 75cc8f8ff4d3ad4048f366551f4626b034308108
[   92.451214][    C0] Hardware name: sifive,hifive-unleashed-a00 (DT)
[   92.464253][    C0] Call Trace:
[   92.470790][    C0] [<ffffffff80005da4>] dump_backtrace+0x30/0x38
[   92.483475][    C0] [<ffffffff80ca26ce>] dump_stack_lvl+0x44/0x5c
[   92.496159][    C0] [<ffffffff80ca26fe>] dump_stack+0x18/0x20
[   92.508118][    C0] [<ffffffff80071fd0>] generic_handle_domain_irq+0x4a/0xb8
[   92.522785][    C0] [<ffffffff8064a108>] plic_handle_irq+0xa0/0x106
[   92.535839][    C0] [<ffffffff800724c4>] handle_domain_irq+0xa4/0xfc
[   92.549062][    C0] [<ffffffff80649e68>] riscv_intc_irq+0x48/0x70
[   92.561744][    C0] [<ffffffff80003a44>] ret_from_exception+0x0/0xc
[   92.574794][    C0] [<ffffffff80085706>] rcu_idle_enter+0x22/0x2a
[   92.587476][    C0] [nfk test]hwirq=61
[   92.595273][    C0] [nfk test] irq =32,hwirq=61
[   92.595289][    C0] [nfk test] name =(null)
```

+ 中断上下文保存、恢复   
```
int handle_domain_irq(struct irq_domain *domain,
                      unsigned int hwirq, struct pt_regs *regs)
{
        struct pt_regs *old_regs = set_irq_regs(regs);
        struct irq_desc *desc;
        int ret = 0;

        irq_enter();

        /* The irqdomain code provides boundary checks */
        desc = irq_resolve_mapping(domain, hwirq);
        if (likely(desc))
                handle_irq_desc(desc);
        else
                ret = -EINVAL;

        irq_exit();
        set_irq_regs(old_regs);
        return ret;
}

```

+  plic_handle_irq  -->   generic_handle_domain_irq   

```
__irq_do_set_handler(struct irq_desc *desc, irq_flow_handler_t handle,
                     int is_chained, const char *name)
desc->handle_irq = handle;

```

```
static inline void generic_handle_irq_desc(struct irq_desc *desc)
{
        desc->handle_irq(desc);
}
```

```
/*
 * Handling an interrupt is a two-step process: first you claim the interrupt
 * by reading the claim register, then you complete the interrupt by writing
 * that source ID back to the same claim register.  This automatically enables
 * and disables the interrupt, so there's nothing else to do.
 */
static void plic_handle_irq(struct irq_desc *desc)
{
        struct plic_handler *handler = this_cpu_ptr(&plic_handlers);
        struct irq_chip *chip = irq_desc_get_chip(desc);
        void __iomem *claim = handler->hart_base + CONTEXT_CLAIM;
        irq_hw_number_t hwirq;

        WARN_ON_ONCE(!handler->present);

        chained_irq_enter(chip, desc);

        while ((hwirq = readl(claim))) {
                int err = generic_handle_domain_irq(handler->priv->irqdomain,
                                                    hwirq);
                if (unlikely(err))
                        pr_warn_ratelimited("can't find mapping for hwirq %lu\n",
                                        hwirq);
        }

        chained_irq_exit(chip, desc);
}
```


+  struct irq_chip plic_chip    

```
static struct irq_chip plic_chip = {
        .name           = "SiFive PLIC",
        .irq_enable     = plic_irq_enable,
        .irq_disable    = plic_irq_disable,
        .irq_mask       = plic_irq_mask,
        .irq_unmask     = plic_irq_unmask,
        .irq_eoi        = plic_irq_eoi,
#ifdef CONFIG_SMP
        .irq_set_affinity = plic_set_affinity,
#endif
        .irq_set_type   = plic_irq_set_type,
        .flags          = IRQCHIP_SKIP_SET_WAKE |
                          IRQCHIP_AFFINITY_PRE_STARTUP,
};
```
## generic_handle_domain_irq  

generic_handle_domain_irq的chip是  



```

                interrupt-controller@c000000 {
                        #interrupt-cells = <0x1>;
                        compatible = "riscv,plic0";
                        interrupt-controller;
                        interrupts-extended = <0x3 0xb 0x4 0xb 0x4 0x9 0x5 0xb 0x5 0x9 0x6 0xb 0x6 0x9 0x7 0xb 0x7 0x9>;
                        reg = <0x0 0xc000000 0x0 0x4000000>;
                        riscv,max-priority = <0x7>;
                        riscv,ndev = <0x35>;
                        phandle = <0x2>;
                };
```
domain->fwnode->dev->full_name: interrupt-controller@c000000   

```
int generic_handle_domain_irq(struct irq_domain *domain, unsigned int hwirq)
{
#if 1
        struct irq_desc *my= irq_resolve_mapping(domain, hwirq);
        struct irq_chip *chip = irq_desc_get_chip(my);
#if 0
        struct fwnode_handle *fwnode = domain->fwnode;
        struct device *dev =NULL;
        struct device_node      *of_node = NULL;
        if(NULL != fwnode)
            dev = fwnode->dev;
        if(NULL != dev)
            of_node = dev ->of_node;
        if(NULL != of_node && NULL != of_node->full_name)
             printk("%s plic name %s, full_name %s \n",__func__, of_node->name, of_node->full_name);    
#endif
        if(0x3d == hwirq)
        {
                //dump_stack();
                printk("[nfk test]hwirq=%u\n",hwirq);
                printk("[nfk test] irq =%d,hwirq=%d",my->irq_data.irq,my->irq_data.hwirq);
                printk("[nfk test] name =%s",my->name);
                //printk("[nfk test] name =%s",my->dev_name);
        }
        if(0 == strncmp(chip->name, "SiFive PLIC", strlen("SiFive PLIC")))
        {
                    //struct plic_handler *handler = this_cpu_ptr(&plic_handlers);
                    printk("sifive plic controller \n");
                    //desc->fake_handle_irq(desc);
        }               
#endif
        return handle_irq_desc(irq_resolve_mapping(domain, hwirq));
}
```

## handle_domain_irq
handle_domain_irq中的desc的chip如下：

```
                        interrupt-controller {
                                #interrupt-cells = <0x1>;
                                compatible = "riscv,cpu-intc";
                                interrupt-controller;
                                phandle = <0x5>;
                        };
```

```
int handle_domain_irq(struct irq_domain *domain,
                      unsigned int hwirq, struct pt_regs *regs)
{
        struct pt_regs *old_regs = set_irq_regs(regs);
        struct irq_desc *desc;
        int ret = 0;

        irq_enter();

        /* The irqdomain code provides boundary checks */
        desc = irq_resolve_mapping(domain, hwirq);
        if (likely(desc))
        {
                struct irq_chip *chip = irq_desc_get_chip(desc);

                struct fwnode_handle *fwnode = domain->fwnode;
                struct device *dev =NULL;
                struct device_node      *of_node = NULL;
                if(NULL != fwnode)
                    dev = fwnode->dev;
                if(NULL != dev)
                    of_node = dev ->of_node;
                if(NULL != of_node && NULL != of_node->full_name)
                     printk("plic name %s, full_name %s \n", of_node->name, of_node->full_name);
                //if(NULL != of_node && NULL != of_node->full_name && 0 == strncmp(of_node->full_name, "interrupt-controller@c000000", strlen("interrupt-controller@c000000")))
                if(NULL != desc->fake_handle_irq)
                {
                    if(0 == strncmp(chip->name, "RISC-V INTC", strlen("RISC-V INTC")))
                    {
                        printk("riscv intc controller process plic interrupt \n");
                    }
                }
                handle_irq_desc(desc);
        }
        else
                ret = -EINVAL;

        irq_exit();
        set_irq_regs(old_regs);
        return ret;
}
```



# irq_of_parse_and_map
该函数的功能是从设备树中获取某一个中断，并且将中断ID转化为linux内核虚拟IRQ number。 IRQ number用于区别中断ID。

[RISC-V 中断子系统分析——硬件及其初始化](https://tinylab.org/riscv-irq-analysis/)
```
    /*获取中断号*/
    interrupt_number = irq_of_parse_and_map(button_device_node, 0);
    printk("\n irq_of_parse_and_map! =  %d \n",interrupt_number);

    /*申请中断, 记得释放*/
    error = request_irq(interrupt_number,button_irq_hander,IRQF_TRIGGER_RISING,"button_interrupt",NULL);
    if(error != 0)
    {
            printk("request_irq error");
            free_irq(interrupt_number, NULL);
            return -1;
    }
```

# 软中断


handle_domain_irq -->  irq_exit -->  __irq_exit_rcu    

```
void irq_exit(void)
{
        __irq_exit_rcu();
        ct_irq_exit();
         /* must be last! */
        lockdep_hardirq_exit();
}

```


```
static inline void __irq_exit_rcu(void)
{
#ifndef __ARCH_IRQ_EXIT_IRQS_DISABLED
        local_irq_disable();
#else
        lockdep_assert_irqs_disabled();
#endif
        account_hardirq_exit(current);
        preempt_count_sub(HARDIRQ_OFFSET);
        if (!in_interrupt() && local_softirq_pending())
                invoke_softirq();

        tick_irq_exit();
}
```