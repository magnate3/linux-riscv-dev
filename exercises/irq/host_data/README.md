
#  struct gic_chip_data

```
struct gic_chip_data {
        struct fwnode_handle    *fwnode;
        void __iomem            *dist_base;
        struct redist_region    *redist_regions;
        struct rdists           rdists;
        struct irq_domain       *domain;
        u64                     redist_stride;
        u32                     nr_redist_regions;
        unsigned int            irq_nr;
        struct partition_desc   *ppi_descs[16];
};
```

```
   if(0 == strcmp(chip->name,"GICv3") )
                    {
                         gic_data = (struct gic_chip_data *)domain->host_data;
                         pr_err("gic_data->irq_nr: %d \t", gic_data->irq_nr);
                    }
```

***gic_data->irq_nr: 0 ***

```
[74968.185446] irq info oupt begin ************************ 
 virq:  265, hwirq: 2621456 ,desc->depth 0, desc->handle_irq 81459d0     
[74968.185447]  leaf chip->name ITS-MSI  
[74968.198430] leaf domain name :  irqchip@0000000202100000-2, and domain->ops->alloc ffff00000814d010, domain->host_data ffff000008eaa398 
[74968.202250]  parent chip->name ITS  
[74968.214540] parent domain name :  irqchip@0000000202100000-4 , domain->ops->alloc： ffff00000843d660, domain->host_data ffff803fc0a18e80 
[74968.218187]  parent chip->name GICv3  
[74968.230652] gic_data->irq_nr: 0 
[74968.234476] parent domain name :  irqchip@ffff00000a8e0000 , domain->ops->alloc： ffff00000843be94, domain->host_data ffff000008dc0228 

[74968.251552]   265  
[74968.251553] 0x280010  
[74968.253643] ITS-MSI          
[74968.255997]    *    
[74968.258952]  RADIX          
[74968.261129] irqchip@0000000202100000-2
[74968.267730]   265+ 
[74968.267731] 0x02de9  
[74968.269823] ITS              
[74968.272085]    *    
[74968.275045]  RADIX          
[74968.277223] irqchip@0000000202100000-4
[74968.283820]   265+ 
[74968.283820] 0x02de9  
[74968.285915] GICv3            
[74968.288178]    *    
[74968.291132]  RADIX          
[74968.293308] irqchip@ffff00000a8e0000
[74968.299737]  IRQ 265 dev_id find 
```



