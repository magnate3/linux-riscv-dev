
# no share

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/hinic.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/hinic_irq1.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/coredump1.png)

# share


```

static int __init rtoax_irq_init(void) {

        printk(KERN_INFO "[RToax]request irq %s!\n", name);
    /*
     *  注册中断
     */
    if (request_irq(IRQ_NUM, no_action, IRQF_SHARED|IRQF_NO_THREAD, name, NULL))
    {
            printk(KERN_ERR "%s: request_irq() failed\n", name);
            return 0;
    }
    succ = true;
    return 0;
}


```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/hinic_irq2.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/fail.png)


```
[24876.581915] [RToax]request irq [RToax]!
[24876.585744] [RToax]: request_irq(265) failed: -22 
```

## request_irq returns error code -22

You can't pass a NULL context (last parameters of the request_irq() call) when dealing with a shared interrupt line (IRQF_SHARED flag is on).

To understand why consider the following scenario: you have two identical network cards sharing the same IRQ. The same driver will pass the same interrupt handler function, the same irq number and the same description. There is no way to distinguish the two instances of the registration except via the context parameter.

Therefore, as a precaution, you can't pass a NULL context parameter if you pass the IRQF_SHARED flag.



# succful demo

```
static int rx_request_irq(struct hinic_rxq *rxq)
{
        struct hinic_dev *nic_dev = netdev_priv(rxq->netdev);
        struct hinic_hwdev *hwdev = nic_dev->hwdev;
        struct hinic_rq *rq = rxq->rq;
        int err;

        rx_add_napi(rxq);

        hinic_hwdev_msix_set(hwdev, rq->msix_entry,
                             RX_IRQ_NO_PENDING, RX_IRQ_NO_COALESC,
                             RX_IRQ_NO_LLI_TIMER, RX_IRQ_NO_CREDIT,
                             RX_IRQ_NO_RESEND_TIMER);


        pr_info("irq %d, irq name %s \n",  rq->irq, rxq->irq_name);
        err = request_irq(rq->irq, rx_irq, IRQF_SHARED|IRQF_NO_THREAD, rxq->irq_name, rxq);
        //err = request_irq(rq->irq, rx_irq, 0, rxq->irq_name, rxq);
        if (err) {
                rx_del_napi(rxq);
                return err;
        }

        return 0;
}
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/iperf3.png)

```
 echo 0x19e5 0x0200 > /sys/bus/pci/drivers/hinic/new_id 
 ip  a add 192.168.10.251/24  dev enp6s0
 ip  a add 192.168.11.251/24  dev  enp5s0
 iperf3 -s 
 iperf3 -c 192.168.11.81 -p 5201 -t 3600 -P 10
 iperf3 -c 192.168.10.81 -p 5201 -t 3600
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/share/my265.png)



# action 

```
void enable_irq(unsigned int irq)
{
        unsigned long flags;
        struct irq_desc *desc = irq_get_desc_buslock(irq, &flags, IRQ_GET_DESC_CHECK_GLOBAL);

        if (!desc)
                return;
        if (WARN(!desc->irq_data.chip,
                 KERN_ERR "enable_irq before setup/request_irq: irq %u\n", irq))
                goto out;

        __enable_irq(desc);
out:
        irq_put_desc_busunlock(desc, flags);
}
```

```
[root@centos7 irq]# dmesg | grep 'begin' -A 20
[  999.286157] irq info oupt begin ************************ 
 desc->depth 0   
[  999.286158]  leaf chip->name ITS-MSI  
[  999.294477] domain name :  irqchip@0000000202100000-2
[  999.298295]  parent chip->name ITS  
[  999.303413] domain name :  irqchip@0000000202100000-4 
[  999.307059]  parent chip->name GICv3  
[  999.312262] domain name :  irqchip@ffff00000a8e0000 
```
