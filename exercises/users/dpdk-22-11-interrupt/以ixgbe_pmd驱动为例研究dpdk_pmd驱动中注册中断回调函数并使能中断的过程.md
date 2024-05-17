# 以 ixgbe pmd 驱动为例研究 dpdk pmd 驱动中注册中断回调函数并使能中断的过程
## dpdk pmd 中注册中断回调函数并使能中断
eth_ixgbe_dev_init 是 ixgbe 网卡的初始化函数，在这个函数的最后注册中断回调并使能中断。

在配置了收发队列等等步骤后，注册中断回调函数，调用如下代码注册：

```c
	rte_intr_callback_register(&pci_dev->intr_handle,
				   ixgbe_dev_interrupt_handler,
				   (void *)eth_dev);
```
使能 uio、vfio 中断、事件描述符映射通过如下代码来完成：

```c
	/* enable uio/vfio intr/eventfd mapping */
	rte_intr_enable(&pci_dev->intr_handle);
```

rte_intr_enable 函数会根据不同的中断处理类型分发到不同的函数上，在 rte_intr_handle_type 中定义不同的中断的类型，定义内容如下：

```c
enum rte_intr_handle_type {
	RTE_INTR_HANDLE_UNKNOWN = 0,
	RTE_INTR_HANDLE_UIO,          /**< uio device handle */
	RTE_INTR_HANDLE_UIO_INTX,     /**< uio generic handle */
	RTE_INTR_HANDLE_VFIO_LEGACY,  /**< vfio device handle (legacy) */
	RTE_INTR_HANDLE_VFIO_MSI,     /**< vfio device handle (MSI) */
	RTE_INTR_HANDLE_VFIO_MSIX,    /**< vfio device handle (MSIX) */
	RTE_INTR_HANDLE_ALARM,    /**< alarm handle */
	RTE_INTR_HANDLE_EXT, /**< external handler */
	RTE_INTR_HANDLE_MAX
};
```
ret_intr_enable 函数的代码如下所示：

```c
int
rte_intr_enable(struct rte_intr_handle *intr_handle)
{
	if (!intr_handle || intr_handle->fd < 0 || intr_handle->uio_cfg_fd < 0)
		return -1;

	switch (intr_handle->type){
	/* write to the uio fd to enable the interrupt */
	case RTE_INTR_HANDLE_UIO:
		if (uio_intr_enable(intr_handle))
			return -1;
		break;
	case RTE_INTR_HANDLE_UIO_INTX:
		if (uio_intx_intr_enable(intr_handle))
			return -1;
		break;
	/* not used at this moment */
	case RTE_INTR_HANDLE_ALARM:
		return -1;
#ifdef VFIO_PRESENT
	case RTE_INTR_HANDLE_VFIO_MSIX:
		if (vfio_enable_msix(intr_handle))
			return -1;
		break;
	case RTE_INTR_HANDLE_VFIO_MSI:
		if (vfio_enable_msi(intr_handle))
			return -1;
		break;
	case RTE_INTR_HANDLE_VFIO_LEGACY:
		if (vfio_enable_intx(intr_handle))
			return -1;
		break;
#endif
	/* unknown handle type */
	default:
		RTE_LOG(ERR, EAL,
			"Unknown handle type of fd %d\n",
					intr_handle->fd);
		return -1;
	}

	return 0;
}
```
可以看到它根据 intr_handle->type 这个中断控制类型分发到不同的子函数上，这里我着重研究下标准的 UIO 设备控制中断的方式。

## 标准 UIO 设备控制中断
对于标准的 uio 设备，通过**向设备文件中写入 1** 来**使能**中断，与之类似**关闭中断**的过程是**向设备文件中写入 0**。

uio_intr_enable 函数的代码摘录如下：
```c
static int
uio_intr_enable(struct rte_intr_handle *intr_handle)
{
	const int value = 1;

	if (write(intr_handle->fd, &value, sizeof(value)) < 0) {
		RTE_LOG(ERR, EAL,
			"Error enabling interrupts for fd %d (%s)\n",
			intr_handle->fd, strerror(errno));
		return -1;
	}
	return 0;
}
```
可以看到，这个函数通过写 1 到 uio 设备文件中来完成使能中断的过程。

## 写入 uio 设备文件代表怎样的行为？
uio 可以看做是一种**字符设备驱动**，在此驱动中注册了**单独的 file_operations 函数表**，可以看做是一种**独立**的设备类型。

file_operations 函数内容如下：

```c
static const struct file_operations uio_fops = {
	.owner		= THIS_MODULE,
	.open		= uio_open,
	.release	= uio_release,
	.read		= uio_read,
	.write		= uio_write,
	.mmap		= uio_mmap,
	.poll		= uio_poll,
	.fasync		= uio_fasync,
	.llseek		= noop_llseek,
};
```

该函树表在 uio_major_init 中初始化 cdev 结构体时使用，相关代码如下：

```c
    cdev->owner = THIS_MODULE;
	cdev->ops = &uio_fops;
	kobject_set_name(&cdev->kobj, "%s", name);

	result = cdev_add(cdev, uio_dev, UIO_MAX_DEVICES);
```
## uio_write 函数
uio_write 是写入 uio 设备文件时内核中最终调用到的写入函数，其代码如下：

```c
static ssize_t uio_write(struct file *filep, const char __user *buf,
			size_t count, loff_t *ppos)
{	
	struct uio_listener *listener = filep->private_data;
	struct uio_device *idev = listener->dev;
	ssize_t retval;
	s32 irq_on;

	if (count != sizeof(s32))
		return -EINVAL;

	if (copy_from_user(&irq_on, buf, count))
		return -EFAULT;

	mutex_lock(&idev->info_lock);
	if (!idev->info) {
		retval = -EINVAL;
		goto out;
	}

	if (!idev->info || !idev->info->irq) {
		retval = -EIO;
		goto out;
	}

	if (!idev->info->irqcontrol) {
		retval = -ENOSYS;
		goto out;
	}

	retval = idev->info->irqcontrol(idev->info, irq_on);

out:
	mutex_unlock(&idev->info_lock);
	return retval ? retval : sizeof(s32);
}
```

可以看到它**从用户态获取**到 **irq_on** 这个变量的值，为 1 对应要使能中断，为 0 则表示关闭中断，在获取了这个参数后，它首先**占用互斥锁**，然后调用 **info** 结构体中实例化的 **irqcontrol 子函数**来完成工作。

## uio_info 结构体及其实例化过程
uio_write 函数中的 idev 变量是一个**指向 struct uio_device 的指针**，**struct uio_device** 中又包含 一个**指向 struct uio_info 的指针**，**struct uio_info** 结构体内容如下：

```c
struct uio_info {
	struct uio_device	*uio_dev;
	const char		*name;
	const char		*version;
	struct uio_mem		mem[MAX_UIO_MAPS];
	struct uio_port		port[MAX_UIO_PORT_REGIONS];
	long			irq;
	unsigned long		irq_flags;
	void			*priv;
	irqreturn_t (*handler)(int irq, struct uio_info *dev_info);
	int (*mmap)(struct uio_info *info, struct vm_area_struct *vma);
	int (*open)(struct uio_info *info, struct inode *inode);
	int (*release)(struct uio_info *info, struct inode *inode);
	int (*irqcontrol)(struct uio_info *info, s32 irq_on);
};
```

每一个 uio 设备都会**实例化**一个 **uio_info 结构体**，uio 驱动自身**不会**实例化 uio_info 结构体，它只**提供一个框架**，可以在其它模块中调用 **uio_register_device** 来实例化 uio_info 结构体，在 dpdk 中，常见方式是**在驱动绑定 igb_uio 的时候调用 uio_register_device 进行实例化。**

## igb_uio.c 中的相关代码
可以在 igb_uio.c 的 probe 函数 **igbuio_pci_probe** 中找到实例化的相关代码，摘录如下：

```c
	/* fill uio infos */
	udev->info.name = "igb_uio";
	udev->info.version = "0.1";
	udev->info.handler = igbuio_pci_irqhandler;
	udev->info.irqcontrol = igbuio_pci_irqcontrol;
#ifdef CONFIG_XEN_DOM0
	/* check if the driver run on Xen Dom0 */
	if (xen_initial_domain())
		udev->info.mmap = igbuio_dom0_pci_mmap;
#endif
	udev->info.priv = udev;
	udev->pdev = dev;
	
...........................................................

	/* register uio driver */
	err = uio_register_device(&dev->dev, &udev->info);
	if (err != 0)
		goto fail_remove_group;
```
可以看到这里对 udev->info 中的字段进行了**填充**，同时**设置**了 **handler** 与 **irqcontrol 回调函数等字段的值**，最后通过 **uio_register_device** **实例化**一个 uio 设备。

## write 写入 uio 设备文件的完整过程
上文中我已经提到过使用 write 系统调用写入 uio 设备文件最终将会调用到

info 结构体中实例化的 irqcontrol 子函数来完成工作，这里 igb_uio 就完成了这样的过程。

**也就是说在绑定网卡到 igb_uio 时，写入接口对应的 uio 设备文件时将会调用 igb_uio 中实例化的 info->irqcontrol 函数来控制中断状态。**

这里提到的 irqcontrol 的实例化函数，在 igb_uio 中对应的就是 igbuio_pci_irqcontrol 函数。其代码如下：

```c
static int
igbuio_pci_irqcontrol(struct uio_info *info, s32 irq_state)
{
	struct rte_uio_pci_dev *udev = info->priv;
	struct pci_dev *pdev = udev->pdev;

	pci_cfg_access_lock(pdev);
	if (udev->mode == RTE_INTR_MODE_LEGACY)
		pci_intx(pdev, !!irq_state);

	else if (udev->mode == RTE_INTR_MODE_MSIX) {
		struct msi_desc *desc;

#if (LINUX_VERSION_CODE < KERNEL_VERSION(4, 3, 0))
		list_for_each_entry(desc, &pdev->msi_list, list)
			igbuio_msix_mask_irq(desc, irq_state);
#else
		list_for_each_entry(desc, &pdev->dev.msi_list, list)
			igbuio_msix_mask_irq(desc, irq_state);
#endif
	}
	pci_cfg_access_unlock(pdev);

	return 0;
}
```
这里需要访问 pci 配置空间，根据不同的中断类型来控制中断状态，这就完成了所有的过程。

## 完整的过程草图

write uio -> uio_write -> idev->info->irqcontrol -> igbuio_pci_irqcontrol

## 设定网卡中断寄存器
完成了上面描述的使能 uio、vfio 中断、事件描述符映射的过程后，网卡初始化函数会设定网卡自身的硬件中断寄存器来使能硬件中断。

对应 ixgbe 驱动中使能网卡硬件中断的函数调用如下：

```c
	/* enable support intr */
	ixgbe_enable_intr(eth_dev);
```

ixgbe_enable_intr 函数通过写入 EIMS 来使能需要的中断源，其代码如下：

```c
static inline void
ixgbe_enable_intr(struct rte_eth_dev *dev)
{
	struct ixgbe_interrupt *intr =
		IXGBE_DEV_PRIVATE_TO_INTR(dev->data->dev_private);
	struct ixgbe_hw *hw =
		IXGBE_DEV_PRIVATE_TO_HW(dev->data->dev_private);

	IXGBE_WRITE_REG(hw, IXGBE_EIMS, intr->mask);
	IXGBE_WRITE_FLUSH(hw);
}
```
从 82599 的手册中找到了如下内容：

>Software enables the required interrupt causes by setting the EIMS register.

与这里设定 EIMS 寄存器的行为一致，至此就完成了所有的初始化过程。