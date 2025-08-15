
# irq map

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/irq_map/map1.png)

```
eth0: ethernet@20030000{
                        compatible = "sifive,fu540-c000-gem";
                        interrupt-parent = <&L5>;
                        //interrupts = <61>, <62>;
                        interrupts = <7>, <8>;
```
硬件中断号：7,  虚拟中断号：1

硬件中断号：7,  映射到多个虚拟中断号


## 为什么一个硬件中断号映射到多个虚拟中断号

***第一次gpio_irq_desc[B].handle_irq***

***第二次gic_irq_desc[A].handle_irq***


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/irq_map/map4.png)

外部设备1、外部设备n共享一个GPIO中断B，多个GPIO中断汇聚到GIC(通用中断控制器)的A号中断，GIC再去中断CPU。那么软件处理时就是反过来，先读取GIC获得中断号A，再细分出GPIO中断B，最后判断是哪一个外部芯片发生了中断。

所以，中断的处理函数来源有三：

① GIC的处理函数：

假设irq_desc[A].handle_irq是XXX_gpio_irq_handler(XXX指厂家)，这个函数需要读取芯片的GPIO控制器，细分发生的是哪一个GPIO中断(假设是B)，再去调用irq_desc[B]. handle_irq。

注意：

irq_desc[A].handle_irq细分出中断后B，调用对应的irq_desc[B].handle_irq。

显然中断A是CPU感受到的顶层的中断，GIC中断CPU时，CPU读取GIC状态得到中断A。

② 模块的中断处理函数：

比如对于GPIO模块向GIC发出的中断B，它的处理函数是irq_desc[B].handle_irq。

BSP开发人员会设置对应的处理函数，一般是handle_level_irq或handle_edge_irq，从名字上看是用来处理电平触发的中断、边沿触发的中断。

注意：

导致GPIO中断B发生的原因很多，可能是外部设备1，可能是外部设备n，可能只是某一个设备，也可能是多个设备。所以irq_desc[B].handle_irq会调用某个链表里的函数，这些函数由外部设备提供。这些函数自行判断该中断是否自己产生，若是则处理。

③ 外部设备提供的处理函数：

这里说的“外部设备”可能是芯片，也可能总是简单的按键。它们的处理函数由自己驱动程序提供，这是最熟悉这个设备的“人”：它知道如何判断设备是否发生了中断，如何处理中断。

对于共享中断，比如GPIO中断B，它的中断来源可能有多个，每个中断源对应一个中断处理函数。所以irq_desc[B]中应该有一个链表，存放着多个中断源的处理函数。

一旦程序确定发生了GPIO中断B，那么就会从链表里把那些函数取出来，一一执行。

这个链表就是action链表。

# irq_find_mapping / irq_create_mapping

驱动中通常会使用platform_get_irq或irq_of_parse_and_map接口，去根据设备树的信息去创建映射关系（硬件中断号到linux irq中断号映射）
## 中断映射

of_amba_device_create -->irq_of_parse_and_map-->irq_find_mapping

```
	if (index < irq_count) {
		    queue->irq = platform_get_irq(pdev, index++);
                } 
                else
                {
                    index = 0;
		    queue->irq = platform_get_irq(pdev, index++);// queue->irq虚拟中断号
                }
		err = devm_request_irq(&pdev->dev, queue->irq, macb_interrupt,
				       IRQF_SHARED, dev->name, queue);
```

***IRQF_SHARED***

##  中断发生

```
static void plic_handle_irq(struct pt_regs *regs)
{
    struct plic_handler *handler = this_cpu_ptr(&plic_handlers);
    void __iomem *claim = plic_hart_offset(handler->ctxid) + CONTEXT_CLAIM;
    irq_hw_number_t hwirq;

    WARN_ON_ONCE(!handler->present);

    csr_clear(sie, SIE_SEIE);
    while ((hwirq = readl(claim))) {
        int irq = irq_find_mapping(plic_irqdomain, hwirq);  // irq虚拟中断号

        if (unlikely(irq <= 0))
            pr_warn_ratelimited("can't find mapping for hwirq %lu\n",
                    hwirq);
        else
            generic_handle_irq(irq);
        writel(hwirq, claim);
    }
    csr_set(sie, SIE_SEIE);
}
```

#
 从处理流程上看，对于 gic 的每个中断源，Linux 系统分配一个 irq_desc 数据结构与之对应。irq_desc 结构中有两个中断处理函数 desc->handle_irq() 和 desc->action->handler()，这两个函数代表中断处理的两个层级：

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/irq_map/map2.png)

## desc->handle_irq &  desc->action->handler

desc->handle_irq()。第一层次的中断处理函数，这个是系统在初始化时根据中断源的特征统一分配的，不同类型的中断源的 gic 操作是不一样的，把这些通用 gic 操作提取出来就是第一层次的操作函数。具体实现包括：
handle_fasteoi_irq()
handle_simple_irq()
handle_edge_irq()
handle_level_irq()
handle_percpu_irq()
handle_percpu_devid_irq()
desc->action->handler() 第二层次的中断处理函数，由用户注册实现具体设备的驱动服务程序，都是和 GIC 操作无关的代码。同时一个中断源可以多个设备共享，所以一个 desc 可以挂载多个 action，由链表结构组织起来。
 


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/irq_map/map3.png)




# references


[linux IRQ Management（七）- 中断处理流程](https://blog.csdn.net/weixin_41028621/article/details/102649154)


[韦东山：剥丝抽茧分析linux中断系统的重要数据结构](https://cloud.tencent.com/developer/article/1709007)