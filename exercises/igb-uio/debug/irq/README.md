
#  run test.c

##  open /dev/uiox

```
//uio_open调用到最后会申请中断，并注册中断处理函数 igbuio_pci_irqhandler
uio_open -> igbuio_pci_open -> igbuio_pci_enable_interrupts
    //更新pci配置空间中msix capability字段，并申请中断号
    pci_enable_msix(udev->pdev, &msix_entry, 1)
    dev_dbg(&udev->pdev->dev, "using MSI-X");
    udev->info.irq_flags = IRQF_NO_THREAD;
    udev->info.irq = msix_entry.vector;
    udev->mode = RTE_INTR_MODE_MSIX;

    //注册中断处理函数
    request_irq(udev->info.irq, igbuio_pci_irqhandler,
          udev->info.irq_flags, udev->info.name, udev);
```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/igb-uio/pics/request_irq.png)

##  lspci -s 0000:05:00.0  -vv
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/igb-uio/pics/b_open.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/igb-uio/pics/a_open.png)

lspci -v可以查看设备支持的capability, 如果有MSI或者MSI-x或者message signal interrupt的描述，并且这些描述后面都有一个enable的flag, 
“+”表示enable，"-"表示disable。

##  igb uio interrupt
```
igb-uio:
rte_intr_disable->uio_intr_disable->igbuio_pci_irqcontrol->pci_msi_mask_irq
rte_intr_enable->uio_intr_enable->igbuio_pci_irqcontrol->pci_msi_unmask_irq

igbuio_pci_open->igbuio_pci_enable_interrupts->pci_alloc_irq_vectors/request_irq
igbuio_pci_release->igbuio_pci_disable_interrupts->free_irq->pci_free_irq_vectors

vfio-pci:
rte_intr_disable->vfio_disable_msix->vfio_pci_ioctl->vfio_msi_disable->pci_free_irq_vectors
rte_intr_enable->vfio_enable_msix->vfio_pci_ioctl->vfio_msi_enable->pci_alloc_irq_vectors/vfio_msi_set_vector_signal->request_irq
```

# references

[DPDK 中断处理流程](https://www.jianshu.com/p/9eb47110cf91)
[PCIe学习笔记之MSI/MSI-x中断及代码分析](https://blog.csdn.net/yhb1047818384/article/details/106676560)