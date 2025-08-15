#define DRIVER_NAME "svm_demo_driver"
#define pr_fmt(fmt) DRIVER_NAME ": " fmt

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/iommu.h>
#include <linux/pci-ats.h>
#include <linux/miscdevice.h>
#include <linux/uaccess.h>
#include <linux/wait.h>

#include "dma_test_common.h"

/**
 * Driver for the svm demo device in Qemu
 *
 * The purpose of this driver is to run a simple demo
 * of in-guest SVM with Intel IOMMU in Qemu.
 *
 * The driver only supports:
 *      - 1 device per VM
 *      - 1 PASID at a time
 *
 * CSR :
 *      - ctrl : trigger the operations (OP_WRITE, OP_READ)
 *      - addr : buffer address
 *      - crc : expected crc of the data to be read in the buffer
 *      - size : buffer size
 *      - pasid : the pasid to use for RW operations
 *      - status : get result of the previous operation (valid after the interrupt)
 */

MODULE_LICENSE("GPL");

#define DEVICE_DISABLE_PASID (1 << 20)

#define BAR0 0
#define DEV_NAME "svm"
#define DEVICE_ID 0x0b11
#define VENDOR_ID 0x1234

/* Write */
#define MMIO_CTRL_OFFSET 0x00
#define MMIO_ADDR_OFFSET 0x8
#define MMIO_CRC_OFFSET 0x10
#define MMIO_SIZE_OFFSET 0x18
#define MMIO_OPT_OFFSET 0x20
#define MMIO_PASID_OFFSET 0x50

/* Read */
#define MMIO_STATUS_OFFSET 0x8

#define OP_WRITE 1
#define OP_READ 2

/* The device does not support multiple PASIDs */
DEFINE_MUTEX(driver_mutex);

static struct pci_device_id pci_ids[] = { { PCI_DEVICE(VENDOR_ID, DEVICE_ID) },
					  { 0 } };
MODULE_DEVICE_TABLE(pci, pci_ids);

static long svm_demo_device_ioctl(struct file *, unsigned int, unsigned long);
static int svm_demo_device_open(struct inode *inode, struct file *file);
static int svm_demo_device_release(struct inode *inode, struct file *file);

static int svm_demo_pci_probe(struct pci_dev *dev,
			      const struct pci_device_id *id);
static void svm_demo_pci_remove(struct pci_dev *dev);

struct device_data {
	struct miscdevice miscdev;
	struct pci_dev *dev;
	void __iomem *remapped;
	struct iommu_sva *sva_full;
	bool open;
	bool pending_op;
	uint64_t src;
	uint64_t dst;
	uint64_t size;
	struct mutex op_mutex;
	int irq;
	wait_queue_head_t wq;
};

static struct pci_driver pci_driver = {
	.name = DRIVER_NAME,
	.id_table = pci_ids,
	.probe = svm_demo_pci_probe,
	.remove = svm_demo_pci_remove,
};

static struct file_operations fops = {
	.owner = THIS_MODULE,
	.open = svm_demo_device_open,
	.release = svm_demo_device_release,
	.unlocked_ioctl = svm_demo_device_ioctl,
};

static struct device_data device = {
	.miscdev = { .name = DEV_NAME,
		     .minor = MISC_DYNAMIC_MINOR,
		     .fops = &fops },
};

/* Helper functions */

static int svm_demo_device_set_user_64(uint64_t arg, uint32_t offset)
{
	uint64_t value;
	if (copy_from_user(&value, (uint64_t *)arg, sizeof(value))) {
		return -EFAULT;
	}
	writeq(value, device.remapped + offset);
	return 0;
}

static uint64_t svm_demo_device_get_64(uint32_t offset)
{
	return readq(device.remapped + offset);
}

/* allocate kernel memory and fill it with user-defined data */
static int svm_allocate_kernel_memory(void __user *arg)
{
	struct kmem_alloc_request req;

	if (copy_from_user(&req, arg, sizeof(req))) {
		return -EFAULT;
	}

	req.res = vmalloc_user(req.size);
	if (copy_from_user(req.res, req.src, req.size)) {
		return -EFAULT;
	}

	if (copy_to_user(arg, &req, sizeof(req))) {
		return -EFAULT;
	}

	printk("first byte from userland : %d\n", ((u8*)req.res)[0]);
	return 0;
}

static void svm_demo_device_run_op(uint32_t cmd)
{
	unsigned long flags;
	spin_lock_irqsave(&device.wq.lock, flags);
	device.pending_op = true;
	writeq(cmd, device.remapped + MMIO_CTRL_OFFSET);
	wait_event_interruptible_locked_irq(device.wq, !device.pending_op);
	spin_unlock_irqrestore(&device.wq.lock, flags);
}

static enum irqreturn svm_demo_device_op_done_irq_handler(int irq, void *data)
{
	unsigned long flags;
	spin_lock_irqsave(&device.wq.lock, flags);
	device.pending_op = false;
	wake_up_locked(&device.wq);
	spin_unlock_irqrestore(&device.wq.lock, flags);
	return IRQ_HANDLED;
}

static int svm_demo_init_irq_handler(void)
{
	int status = pci_alloc_irq_vectors(device.dev, 1, 1, PCI_IRQ_MSI);
	if (status != 1) {
		return status;
	}

	device.irq = pci_irq_vector(device.dev, 0);
	status = request_irq(device.irq, svm_demo_device_op_done_irq_handler, 0,
			     "svm-irq", &device);
	return status;
}

static void svm_demo_cleanup_irq(void)
{
	free_irq(device.irq, &device);
	pci_free_irq_vectors(device.dev);
}

static void svm_demo_device_set_pasid(uint32_t pasid)
{
	writel(pasid, device.remapped + MMIO_PASID_OFFSET);
}

static int svm_demo_enable_sva(void)
{
	int ret;

	if (!device.dev->ats_enabled) {
		ret = pci_enable_ats(device.dev, PAGE_SHIFT);
		if (ret) {
			pr_err("pci_enable_ats error : %d\n", ret);
			return ret;
		}
	}

	ret = iommu_dev_enable_feature(&device.dev->dev, IOMMU_DEV_FEAT_IOPF);
	if (ret) {
		pr_err("IOMMU_DEV_FEAT_IOPF error : %d\n", ret);
		return ret;
	}

	ret = iommu_dev_enable_feature(&device.dev->dev, IOMMU_DEV_FEAT_SVA);
	if (ret) {
		pr_err("IOMMU_DEV_FEAT_SVA error : %d\n", ret);
		return ret;
	}

	return 0;
}

static void svm_demo_disable_sva(void)
{
	if (iommu_dev_disable_feature(&device.dev->dev, IOMMU_DEV_FEAT_SVA)) {
		pr_err("sva cannot be disabled\n");
	}
	if (iommu_dev_disable_feature(&device.dev->dev, IOMMU_DEV_FEAT_IOPF)) {
		pr_err("iopf cannot be disabled\n");
	}
	pr_info("features disabled\n");
}

long svm_demo_device_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	long ret;
	switch (cmd) {
	case IOCTL_SET_ADDR:
		ret = svm_demo_device_set_user_64(arg, MMIO_ADDR_OFFSET);
		break;
	case IOCTL_SET_CRC:
		ret = svm_demo_device_set_user_64(arg, MMIO_CRC_OFFSET);
		break;
	case IOCTL_SET_SIZE:
		ret = svm_demo_device_set_user_64(arg, MMIO_SIZE_OFFSET);
		break;
	case IOCTL_SET_OPT:
		ret = svm_demo_device_set_user_64(arg, MMIO_OPT_OFFSET);
		break;
	case IOCTL_START_WRITE:
	case IOCTL_START_READ:
		svm_demo_device_run_op(cmd == IOCTL_START_READ ? OP_READ :
								 OP_WRITE);
		ret = svm_demo_device_get_64(MMIO_STATUS_OFFSET);
		break;
	case IOCTL_ALLOC_KMEM:
		ret = svm_allocate_kernel_memory((void __user *)arg);
		break;
	default:
		pr_err("Error, invalid IOCTL\n");
		ret = -EINVAL;
		break;
	}

	return ret;
}

static int svm_demo_device_open(struct inode *inode, struct file *file)
{
	int ret = 0;
	uint32_t pasid;
	mutex_lock(&driver_mutex);
	/* prevent multiple processes from using the device concurrently */
	if (device.open) {
		ret = -EBUSY;
		pr_info("PID %d failed to open the device (busy)\n",
			current->pid);
		goto end;
	}

	device.sva_full = iommu_sva_bind_device(&device.dev->dev, current->mm);
	if (IS_ERR(device.sva_full)) {
		dev_err(&device.dev->dev, "pasid allocation failed: %ld\n",
			PTR_ERR(device.sva_full));
		ret = PTR_ERR(device.sva_full);
		goto end;
	}
	pasid = iommu_sva_get_pasid(device.sva_full);
	if (pasid == IOMMU_PASID_INVALID) {
		pr_err("Invalid PASID\n");
		ret = -EIO;
		goto end;
	}
	pr_info("PASID - PID : %d - %d\n", pasid, current->pid);
	svm_demo_device_set_pasid(pasid);

	device.open = true;

end:
	mutex_unlock(&driver_mutex);
	return ret;
}

static int svm_demo_device_release(struct inode *inode, struct file *file)
{
	mutex_lock(&driver_mutex);

	pr_info("Unbind\n");
	iommu_sva_unbind_device(device.sva_full);
	svm_demo_device_set_pasid(DEVICE_DISABLE_PASID);

	device.open = false;
	mutex_unlock(&driver_mutex);
	return 0;
}

static int svm_demo_pci_probe(struct pci_dev *dev,
			      const struct pci_device_id *id)
{
	int ret;
	device.dev = dev;

	ret = pci_request_region(dev, BAR0, "bar0_region");
	if (ret) {
		pr_err("Request region error\n");
		return ret;
	}

	device.remapped =
		pci_iomap(device.dev, BAR0, pci_resource_len(device.dev, BAR0));
	ret = pci_enable_device(device.dev);
	if (ret) {
		pr_err("Cannot enable the device\n");
		return ret;
	}

	pci_set_master(device.dev);

	ret = svm_demo_init_irq_handler();
	if (ret) {
		pr_err("IRQ initialization error\n");
		return ret;
	}

	return svm_demo_enable_sva();
}

static void svm_demo_pci_remove(struct pci_dev *dev)
{
	svm_demo_cleanup_irq();
	pci_iounmap(device.dev, device.remapped);
	pci_release_region(dev, BAR0);
	pci_clear_master(device.dev);
	pci_disable_device(device.dev);
}

static int svm_demo_driver_init(void)
{
	int ret = pci_register_driver(&pci_driver);
	if (ret) {
		pr_err("Cannot register driver\n");
		return ret;
	}

	device.pending_op = false;
	device.open = false;

	misc_register(&device.miscdev);
	init_waitqueue_head(&device.wq);
	mutex_init(&device.op_mutex);

	return 0;
}

static void svm_demo_driver_exit(void)
{
	svm_demo_disable_sva();
	pci_unregister_driver(&pci_driver);
	misc_deregister(&device.miscdev);
}

module_init(svm_demo_driver_init);
module_exit(svm_demo_driver_exit);