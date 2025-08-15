

```
[root@centos7 irq]# dmesg | tail -n 30
[59517.191404] [RToax]free irq.
[59520.090305] [RToax]request irq [RToax]!
[59520.094131] irq info oupt begin ************************ 
 virq:  265, hwirq: 2621456 ,desc->depth 0, desc->handle_irq 81459d0     
[59520.094132]  leaf chip->name ITS-MSI  
[59520.107118] leaf domain name :  irqchip@0000000202100000-2, and domain->ops->alloc ffff00000814d010 
[59520.110941]  parent chip->name ITS  
[59520.120123] parent domain name :  irqchip@0000000202100000-4 , domain->ops->alloc： ffff00000843d660
[59520.123788]  parent chip->name GICv3  
[59520.133067] parent domain name :  irqchip@ffff00000a8e0000 , domain->ops->alloc： ffff00000843be94

[59520.147485]   265  
[59520.147486] 0x280010  
[59520.149575] ITS-MSI          
[59520.151929]    *    
[59520.154882]  RADIX          
[59520.157058] irqchip@0000000202100000-2
[59520.163660]   265+ 
[59520.163661] 0x02d69  
[59520.165753] ITS              
[59520.168015]    *    
[59520.170972]  RADIX          
[59520.173148] irqchip@0000000202100000-4
[59520.179745]   265+ 
[59520.179746] 0x02d69  
[59520.181838] GICv3            
[59520.184100]    *    
[59520.187055]  RADIX          
[59520.189230] irqchip@ffff00000a8e0000
[59520.195661]  IRQ 265 dev_id find 
[root@centos7 irq]# cat /proc/kallsyms | grep ffff00000814d010
ffff00000814d010 t msi_domain_alloc
[root@centos7 irq]# cat /proc/kallsyms | grep ffff00000843d660
ffff00000843d660 t its_irq_domain_alloc
[root@centos7 irq]# cat /proc/kallsyms | grep ffff00000843be94
ffff00000843be94 t gic_irq_domain_alloc
[root@centos7 irq]# cat /proc/kallsyms | grep 81459d0
ffff0000081459d0 T handle_fasteoi_irq ////////////////////////////////////
[root@centos7 irq]# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/alloc.png)

## handle_fasteoi_irq


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/handle.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/gicpng.png)

## its_irq_domain_alloc


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/hwirq.png)

```
static int its_irq_gic_domain_alloc(struct irq_domain *domain,
                                    unsigned int virq,
                                    irq_hw_number_t hwirq)
{
        struct irq_fwspec fwspec;

        if (irq_domain_get_of_node(domain->parent)) {
                fwspec.fwnode = domain->parent->fwnode;
                fwspec.param_count = 3;
                fwspec.param[0] = GIC_IRQ_TYPE_LPI;
                fwspec.param[1] = hwirq;
                fwspec.param[2] = IRQ_TYPE_EDGE_RISING;
        } else if (is_fwnode_irqchip(domain->parent->fwnode)) {
                fwspec.fwnode = domain->parent->fwnode;
                fwspec.param_count = 2;
                fwspec.param[0] = hwirq;
                fwspec.param[1] = IRQ_TYPE_EDGE_RISING;
        } else {
                return -EINVAL;
        }

        return irq_domain_alloc_irqs_parent(domain, virq, 1, &fwspec);
}

static int its_irq_domain_alloc(struct irq_domain *domain, unsigned int virq,
                                unsigned int nr_irqs, void *args)
{
        msi_alloc_info_t *info = args;
        struct its_device *its_dev = info->scratchpad[0].ptr;
        irq_hw_number_t hwirq;
        int err;
        int i;

        err = its_alloc_device_irq(its_dev, nr_irqs, &hwirq);
        if (err)
                return err;

        for (i = 0; i < nr_irqs; i++) {
                err = its_irq_gic_domain_alloc(domain, virq + i, hwirq + i); /////virq还是用旧的，用新的
                if (err)
                        return err;

                irq_domain_set_hwirq_and_chip(domain, virq + i,
                                              hwirq + i, &its_irq_chip, its_dev);
                irqd_set_single_target(irq_desc_get_irq_data(irq_to_desc(virq + i)));
                pr_debug("ID:%d pID:%d vID:%d\n",
                         (int)(hwirq + i - its_dev->event_map.lpi_base),
                         (int)(hwirq + i), virq + i);
        }

        return 0;
}

```

# msi_domain_alloc


```
static int msi_domain_alloc(struct irq_domain *domain, unsigned int virq,
                            unsigned int nr_irqs, void *arg)
{
        struct msi_domain_info *info = domain->host_data;
        struct msi_domain_ops *ops = info->ops;
        irq_hw_number_t hwirq = ops->get_hwirq(info, arg);
        int i, ret;

        if (irq_find_mapping(domain, hwirq) > 0)
                return -EEXIST;

        if (domain->parent) {
                ret = irq_domain_alloc_irqs_parent(domain, virq, nr_irqs, arg);
                if (ret < 0)
                        return ret;
        }

        for (i = 0; i < nr_irqs; i++) {
                ret = ops->msi_init(domain, info, virq + i, hwirq + i, arg);
                if (ret < 0) {
                        if (ops->msi_free) {
                                for (i--; i > 0; i--)
                                        ops->msi_free(domain, info, virq + i);
                        }
                        irq_domain_free_irqs_top(domain, virq, nr_irqs);
                        return ret;
                }
        }

        return 0;
}

```





![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/hi.png)

gic中断控制器初始化时会去add gic irq_domain, gic irq_domain是its irq_domain的parent节点，
its irq_domain中的host data对应的pci_msi irq_domain.

 ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/hi2.png)
			  
	
# irq_chip_enable_parent	
	
	

	
# msi_domain_alloc_irqs	

__pci_enable_msix_range()
	+-> __pci_enable_msix()
		+-> msix_capability_init()
			+-> pci_msi_setup_msi_irqs
pci_msi_setup_msi_irqs--> msi_domain_alloc_irqs->irq_domain_activate_irq->__irq_domain_activate_irq->msi_domain_activate
->irq_chip_write_msi_msg->pci_msi_domain_write_msg->__pci_write_msi_msg