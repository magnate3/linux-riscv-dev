

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/kallsyms.h>
#include <linux/dma-map-ops.h>

#define DEV_NAME	"test"
#define CMA_BUF_SIZE	0x1000000 /* 16MiB */

static struct test_device {
	struct platform_device *pdev;
	/* DMA Memory Physical Address */
	dma_addr_t cma_addr;
	/* DMA Memory Virtual Address */
	char *cma_buffer;
} bdev;

/* character device open method */
static int mmap_open(struct inode *inode, struct file *filp)
{
	printk(KERN_INFO "mmap_alloc: device open\n");
        return 0;
}
/* character device last close method */
static int test_release(struct inode *inode, struct file *filp)
{
	printk(KERN_INFO "mmap_alloc: device is being released\n");
        return 0;
}
#if 0
// helper function, mmap's the allocated area which is physically contiguous
int test_mmap(struct file *filp, struct vm_area_struct *vma)
{
        int ret;
        long length = vma->vm_end - vma->vm_start;

        /* check length - do not allow larger mappings than the number of
           pages allocated */
        if (length > NPAGES * PAGE_SIZE)
                return -EIO;
/* #ifdef ARCH_HAS_DMA_MMAP_COHERENT */
	if (vma->vm_pgoff == 0) {
		printk(KERN_INFO "Using dma_mmap_coherent\n");
		ret = dma_mmap_coherent(NULL, vma, alloc_ptr,
					dma_handle, length);
	} else
/* #else */
	{
		printk(KERN_INFO "Using remap_pfn_range\n");
		vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
		vma->vm_flags |= VM_IO;
		printk(KERN_INFO "off=%d\n", vma->vm_pgoff);
	        ret = remap_pfn_range(vma, vma->vm_start,
			      PFN_DOWN(virt_to_phys(bus_to_virt(dma_handle))) +
			      vma->vm_pgoff, length, vma->vm_page_prot);
	}
/* #endif */
        /* map the whole physically contiguous area in one piece */
        if (ret < 0) {
		printk(KERN_ERR "mmap_alloc: remap failed (%d)\n", ret);
		return ret;
        }
        
        return 0;
}
#else
static inline bool force_dma_unencrypted(struct device *dev)
{
	        return false;
}
#ifdef CONFIG_MMU
/*
 *  * Return the page attributes used for mapping dma_alloc_* memory, either in
 *   * kernel space if remapping is needed, or to userspace through dma_mmap_*.
 *    */
pgprot_t dma_pgprot(struct device *dev, pgprot_t prot, unsigned long attrs)
{
	        if (force_dma_unencrypted(dev))
			                prot = pgprot_decrypted(prot);
		        if (dev_is_dma_coherent(dev))
				                return prot;
#ifdef CONFIG_ARCH_HAS_DMA_WRITE_COMBINE
			        if (attrs & DMA_ATTR_WRITE_COMBINE)
					                return pgprot_writecombine(prot);
#endif
				        return pgprot_dmacoherent(prot);
}
#endif /* CONFIG_MMU */
/*
 * Create userspace mapping for the DMA-coherent memory.
 */
//int dma_common_mmap(struct device *dev, struct vm_area_struct *vma,
//		void *cpu_addr, dma_addr_t dma_addr, size_t size,
//		unsigned long attrs)
//{
//#ifdef CONFIG_MMU
//	unsigned long user_count = vma_pages(vma);
//	unsigned long count = PAGE_ALIGN(size) >> PAGE_SHIFT;
//	unsigned long off = vma->vm_pgoff;
//	int ret = -ENXIO;
//
//	vma->vm_page_prot = dma_pgprot(dev, vma->vm_page_prot, attrs);
//
//	if (dma_mmap_from_dev_coherent(dev, vma, cpu_addr, size, &ret))
//		return ret;
//
//	if (off >= count || user_count > count - off)
//		return -ENXIO;
//
//	return remap_pfn_range(vma, vma->vm_start,
//			page_to_pfn(virt_to_page(cpu_addr)) + vma->vm_pgoff,
//			user_count << PAGE_SHIFT, vma->vm_page_prot);
//#else
//	return -ENXIO;
//#endif /* CONFIG_MMU */
//}
static int test_mmap(struct file *filp, struct vm_area_struct *vma)
{
#if 0
		return dma_common_mmap(&bdev.pdev->dev, vma, bdev.cma_buffer,
							bdev.cma_addr, vma->vm_end - vma->vm_start, 0);
#else
	return 0;
#endif
}
#endif

static struct file_operations test_fops = {
	.owner          = THIS_MODULE,
	.mmap		= test_mmap,
	.release = test_release,
};

static struct miscdevice test_dev= {
	.minor  = MISC_DYNAMIC_MINOR,
	.name   = DEV_NAME,
	.fops   = &test_fops,
};
#ifdef CONFIG_DMA_OPS
#include <asm/dma-mapping.h>
//static inline const struct dma_map_ops *get_arch_dma_ops(struct bus_type *bus)
//{
//	        return dma_ops;
//}
static inline const struct dma_map_ops *test_get_dma_ops(struct device *dev)
{
	        if (dev->dma_ops)
			                return dev->dma_ops;
		        return get_arch_dma_ops(dev->bus);
}
#if 0
static inline struct dma_coherent_mem *dev_get_coherent_memory(struct device *dev)
{
	        if (dev && dev->dma_mem)
			                return dev->dma_mem;
		        return NULL;
}
#endif
#else /* CONFIG_DMA_OPS */
static inline const struct dma_map_ops *get_dma_ops(struct device *dev)
{
	        return NULL;
}
#endif
static int test_probe(struct platform_device *pdev)
{
	gfp_t flag =  GFP_KERNEL;
	unsigned long attrs;
	const struct dma_map_ops *ops ;
	void *(*dma_direct_alloc)(struct device *, size_t , dma_addr_t *, gfp_t , unsigned long ) = NULL;
       	dma_direct_alloc = 0xffffffff8008ab7e;
       	//dma_direct_alloc = kallsyms_lookup_name("dma_direct_alloc");;
	printk("************* %s enter.\n", __func__);
	ops = test_get_dma_ops(&pdev->dev);
	/* Force */
	//printk("before set, common  dma ops %p,   dma ops %p \n",dma_ops, test_get_dma_ops(&pdev->dev));
	//pdev->dev.dma_ops = &gart_dma_ops;
	//pdev->dev.dma_ops = NULL;
        //pdev->dev.dma_ops = 0xffffffffa140d2c0;
	printk("after set , dma ops %p, coherent_dma_mask %llx bus_dma_limit %llx \n", ops, pdev->dev.coherent_dma_mask, pdev->dev.bus_dma_limit);

#if 0
	/* CMA Memory Allocate */
	bdev.cma_buffer = dma_alloc_coherent(&pdev->dev,
				CMA_BUF_SIZE, &bdev.cma_addr, GFP_KERNEL);
#else
	attrs =      (flag & __GFP_NOWARN) ? DMA_ATTR_NO_WARN : 0;
	     /* let the implementation decide on the zone to allocate from: */
	flag &= ~(__GFP_DMA | __GFP_DMA32 | __GFP_HIGHMEM);
	bdev.cma_buffer = dma_direct_alloc(&pdev->dev,
				CMA_BUF_SIZE, &bdev.cma_addr, flag,attrs );
#endif
	if (!bdev.cma_buffer) {
		printk("System Error: DMA Memory Allocate.\n");
		return -ENOMEM;
	}
	else
	{
		printk(" DMA Memory Allocate. %p\n", bdev.cma_buffer);
	}
	/* MISC */
	return misc_register(&test_dev);
}

static int test_remove(struct platform_device *pdev)
{
	misc_deregister(&test_dev);
	dma_free_coherent(&pdev->dev, CMA_BUF_SIZE,
				bdev.cma_buffer, bdev.cma_addr);
	return 0;
}
static const struct of_device_id match_ids[] = {
		{ .compatible = "test", },
			{  }
};
MODULE_DEVICE_TABLE(of, match_ids);
static struct platform_driver test_driver = {
	.probe    = test_probe,
	.remove   = test_remove,
	.driver	= {
		.owner	= THIS_MODULE,
		.name	= DEV_NAME,
		.of_match_table = of_match_ptr(match_ids),
	},
};
/*
注册驱动必须保证 platform_driver 的 driver.name 字段必须和 platform_device 的 name 相
同， 否则无法将驱动和设备进行绑定而注册失败。
*/
static int __init test_init(void)
{
	int ret;

	ret = platform_driver_register(&test_driver);
	if (ret) {
		printk("Error: Platform driver register.\n");
		return -EBUSY;
	}

	bdev.pdev = platform_device_register_simple(DEV_NAME, 1, NULL, 0);
	if (IS_ERR(bdev.pdev)) {
		printk("Error: Platform device register\n");
		return PTR_ERR(bdev.pdev);
	}
	return 0;
}

static void __exit test_exit(void)
{
	platform_device_unregister(bdev.pdev);
	platform_driver_unregister(&test_driver);

	return ;
}
module_init(test_init);
module_exit(test_exit);
MODULE_LICENSE("GPL");
