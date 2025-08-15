

#  GICv4 中断虚拟化

[GIC 中断虚拟化](https://zhuanlan.zhihu.com/p/535997324)   
GICv4支持直接中断注入，此时KVM对GIC的模拟分为2部分：   

•控制面：即Guest OS里GIC driver配置vGIC的控制路径上依然采用软件模拟的方式，这意味着Guest OS对vGIC的每次读写都会陷入到KVM内的vGIC模块。   

•数据面：即设备发出MSIX消息经IOMMU->ITS->Redistributor->virtual CPU interface，这条路径是全硬件支持，不再经过Host OS绕行转发中断。   

从前一页我们可知，设备直通时QEMU会配置KVM通过IRQFD转发转发中断：QEMU配置IRQ bypass时会用ioctl KVM_IRQFD和KVM_SET_GSI_ROUTING在KVM设置2级映射：IRQFD -> GSI ->MSIX。这一步如果检测到系统支持直接中断注入，会调用kvm_vgic_v4_set_forwarding执行另一种更快速形式的中断路由：直接中断注入。   

KVM会调用physical GIC驱动配置物理ITS的interrupt translation table以建立虚拟中断到vPE的映射：device ID, event ID -> (vpe, vintid, pintid)。这里device ID和event ID来自MSIX消息，而vintid(virtual LPI)来自guest OS配置vGIC时MAPTI命令里传入的vLPI。之所以需要pintid(physical LPI)是防止中断触发时vPE不处于运行状态，这时仍然需要使用physical LPI触发hypervisor KVM介入以上一页介绍的软件慢速形式向guest OS注入中断。  


#  KVM_SET_GSI_ROUTING




# references

[Armv8架构虚拟化介绍](https://calinyara.github.io/technology/2019/11/03/armv8-virtualization.html)   