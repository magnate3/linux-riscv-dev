#include <linux/init.h>
#include <asm/io.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/pci.h>

extern dma_addr_t iommu_dma_map_page(struct device *dev, struct page *page,
                                     unsigned long offset, size_t size, enum dma_data_direction dir,
                                     unsigned long attrs);

int magic_value = 0xF0F0F0F0F0F0F;

struct page page_ = {
    .flags = 0xF0F0F0F0F0F0F,
    .dma_addr = 0x0000002f000f0000,
};

static int my_init(void)
{
    dma_addr_t dma_addr;
    struct pci_dev *dummy = pci_get_device(0x10EE, 0x0666, NULL);
    if (dummy != NULL)
    {
        printk(KERN_INFO "module loaded.\n");
        dma_addr = iommu_dma_map_page(&(dummy->dev), &page_, 0, 4096, DMA_BIDIRECTIONAL, DMA_ATTR_SKIP_CPU_SYNC);
        printk(KERN_INFO "DMA_addr: %llx", dma_addr);
    }
    else
    {
        printk("Error getting device");
    }

    return 0;
}

static void my_exit(void)
{
    printk(KERN_INFO "iommu_alloc unloaded.\n");

    return;
}

module_init(my_init);
module_exit(my_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("benedict.schlueter@inf.ethz.ch");
MODULE_DESCRIPTION("Alloc IOMMU entry");