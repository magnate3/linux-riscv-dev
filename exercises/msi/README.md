#  test1

```

static int test_init(bool isbp)
{
 
    if(isbp)return 0;
    struct pci_device pci = {._vendor_id = VIRTIO_VENDOR_ID,
                         ._device_id = VIRTIO_NET_SUBID};
    if (find_pci(&pci)){
        pci_device_parse_config(&pci);
        parse_pci_config(&pci);
        pci_dump_config(&pci);
    }
    return 0;
}

[0:3.0] vid:id = 1af4:1000
    bar[1]: 32bits addr=000000000000C000 size=20
    bar[2]: 32bits addr=00000000FEBD0000 size=1000
    IRQ = 11
    Have MSI-X!        msix_location: 64        msix_ctrl: 2        msix_msgnum: 3        
msix_table_bar: 1        msix_table_offset: 0        msix_pba_bar: 1        

```
# test2

```
void hpet_msi_unmask(int idx)
{
    unsigned int cfg;
 
    cfg = hpet_readl(HPET_Tn_CFG(idx));
    cfg |= HPET_TN_ENABLE | HPET_TN_FSB;
    hpet_writel(cfg, HPET_Tn_CFG(idx));
}
 
void hpet_msi_mask(int idx)
{
    unsigned int cfg;
 
    cfg = hpet_readl(HPET_Tn_CFG(idx));
    cfg &= ~(HPET_TN_ENABLE | HPET_TN_FSB);
    hpet_writel(cfg, HPET_Tn_CFG(idx));
}
 
 
void hpet_msi_write(int idx, struct msi_msg *msg)
{
    hpet_writel(msg->data, HPET_Tn_ROUTE(idx));
    hpet_writel(msg->address_lo, HPET_Tn_ROUTE(idx) + 4);
}
 
void hpet_oneshoot(int channelid, u64 delta)
{
    u64 cmp = hpet_readq(HPET_COUNTER) + delta;
    hpet_writeq(cmp, HPET_Tn_CMP(channelid));
}
 
void hpet_set_msi(int vector,int distid,int channelid)
{
    struct msi_msg msg;
    void msi_compose_msg(int,int,struct msi_msg *);
    msi_compose_msg(vector,distid,&msg);
    printk("msi_msg：%lx,%lx,%lxn",msg.address_lo,msg.address_hi,msg.data);
    hpet_msi_unmask(channelid);
    hpet_msi_write(channelid,&msg);
 
}

```
hpet_set_msi设置定时器msi方式中断。vector为中断号,distid为CPUID，channelid为hpet定时号。


```

static __init int init_hpet_call(bool isbp)
{
    if (isbp) {
        init_hpet();
        hpet_set_msi(45,1,2);
        hpet_oneshoot(2,100000UL*2000UL);//10seconds
 
        hpet_set_msi(46,1,3);
        hpet_oneshoot(3,100000UL*5000UL);//10seconds
 
    }
    return 0;
}

```

在初始化的时候注册2次单次触发时钟。中断号分别为45与46。cpu为1号，即非启动CPU。
45,46号中断没有注册中断向量，会到default_irq_handler里面。
arch/x86_64/irq.c

```

void default_irq_handler(int n)
{
    if (++count > 0) {
        printk("****default_irq_handler**** irq:%d,cpu:%dn", n,smp_processor_id());
        count = 0;
    }
    ack_lapic_irq();

}
```
运行结果：
```
msix_table_bar:1,1
[0:3.0] vid:id = 1af4:1000
    bar[1]: 32bits addr=000000000000C000 size=20
    bar[2]: 32bits addr=00000000FEBD0000 size=1000
    IRQ = 11
    Have MSI-X!        msix_location: 64        msix_ctrl: 2        
msix_msgnum: 3        msix_table_bar: 1        msix_table_offset: 0       
 msix_pba_bar: 1        msix_pba_offset: 2048
ap start done
KHeap: Free:ffffffff81013000,addr,size:2000
****default_irq_handler**** irq:45,cpu:1
****default_irq_handler**** irq:46,cpu:1

```

# reference

```

git clone https://github.com/saneee/x86_64_kernel.git
cd 0019
make qemu
```