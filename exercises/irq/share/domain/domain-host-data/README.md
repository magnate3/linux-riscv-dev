# gic_data

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/domain-host-data/gic_data.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/domain-host-data/code.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/domain/domain-host-data/res.png)


```
[root@centos7 irq]# dmesg | tail -n 30
[  610.537028] [RToax]request irq [RToax]!
[  610.540868] irq info oupt begin ************************ 
 virq:  265, hwirq: 2621456 ,desc->depth 0, desc->handle_irq 81459d0     
[  610.540869]  leaf chip->name ITS-MSI  
[  610.553852] leaf domain name :  irqchip@0000000202100000-2, and domain->ops->alloc ffff00000814d010, domain->host_data ffff000008eaa398 
[  610.557670]  parent chip->name ITS  
[  610.569960] parent domain name :  irqchip@0000000202100000-4 , domain->ops->alloc： ffff00000843d660, domain->host_data ffff803fc0a18e80 
[  610.573606]  parent chip->name GICv3  
[  610.586067] gic_data->irq_nr: 0 
[  610.589890] parent domain name :  irqchip@ffff00000a8e0000 , domain->ops->alloc： ffff00000843be94, domain->host_data ffff000008dc0228 

[  610.606961]   265  
[  610.606962] 0x280010  
[  610.609051] ITS-MSI          
[  610.611403]    *    
[  610.614358]  RADIX          
[  610.616533] irqchip@0000000202100000-2
[  610.623135]   265+ 
[  610.623136] 0x02ce9  
[  610.625235] ITS              
[  610.627497]    *    
[  610.630455]  RADIX          
[  610.632631] irqchip@0000000202100000-4
[  610.639230]   265+ 
[  610.639231] 0x02ce9  
[  610.641325] GICv3            
[  610.643587]    *    
[  610.646539]  RADIX          
[  610.648715] irqchip@ffff00000a8e0000
[  610.655145]  IRQ 265 dev_id find 
```