/*
 * SCE394 - Lab 14 - Memory Mapping
 *
 * Code skeleton.
 */

#include <linux/version.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/cdev.h>

#include <linux/sched.h>
#include <linux/slab.h>

#include <asm/pgtable.h>
#include <linux/mm.h>
#include <asm/io.h> // virt_to_phys()

#define MY_MAJOR		42
#define MY_MINOR		0
#define MODULE_NAME		"mymmap"
#define MAX_PAGES		16

MODULE_DESCRIPTION("Simple mmap driver");
MODULE_AUTHOR("SCE394");
MODULE_LICENSE("GPL");

static struct my_device_data {
	struct cdev cdev;
	size_t size;
} dev;

static char *kmalloc_area;
static char *vmalloc_area;
static char *page_area;
static struct page *page;
static char *area;

static int mymmap_open(struct inode *inode, struct file *file)
{
	struct my_device_data *my_data =
		container_of(inode->i_cdev, struct my_device_data, cdev);
	file->private_data = my_data;
	pr_info("[mymmap_open] Device opened\n");
	return 0;
}

static int mymmap_release(struct inode *inode, struct file *file)
{
	pr_info("[mymmap_release] Device released\n");
	return 0;
}

static ssize_t mymmap_read(struct file *file, char __user *user_buffer,
		size_t size, loff_t *offset) 
{
	char *buf = NULL;
#if defined(USE_KMALLOC)
	buf = kmalloc_area;
#elif defined(USE_VMALLOC)
	buf = vmalloc_area;
#elif defined(USE_ALLOC_PAGES)
	buf = page_area;
#endif
	pr_info("[mymmap_read] current %s\n", buf);
	return 0;
}

static int mymmap_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct my_device_data *my_data = (struct my_device_data*) file->private_data;
	unsigned long len;
	unsigned long pfn;
	unsigned long start;

	int nr_pages;
	int i;
	int ret;

	char *vmalloc_area_ptr;

	pr_info("[mymmap_mmap] mmap() is called\n");

	if (my_data->size == 0)
		return -ENOMEM;

	/* TODO: Calculate required size of memory allocating */
	len = vma->vm_end - vma->vm_start;
	/* TODO: Check that the device has sufficient space */
	if(len > my_data->size)
		return -ENOMEM;
	/* TODO: Check the mmap() require whether zero byte or not */
	if(len == 0)
		return -EINVAL;

#if defined(USE_KMALLOC)
	/* TODO 1-2: Convert virtual address to physical address */
	pfn = virt_to_phys((void*)kmalloc_area) >> PAGE_SHIFT;
	/* TODO 1-3: Remapping pfn into vm_area_struct */
	if(remap_pfn_range(vma, vma->vm_start, pfn, len, vma->vm_page_prot))
		return -EAGAIN;
#elif defined(USE_VMALLOC)
	/* TODO 2-3: Convert virtual address to page frame number */
	start = vma->vm_start;
	vmalloc_area_ptr = vmalloc_area;
	while(len > 0){
		pfn = vmalloc_to_pfn(vmalloc_area_ptr);
		if(remap_pfn_range(vma, start, pfn, PAGE_SIZE, vma->vm_page_prot))
			return -EAGAIN;
		start += PAGE_SIZE;
		vmalloc_area_ptr += PAGE_SIZE;
		len -= PAGE_SIZE;
	}
	/* TODO 2-3: Remapping pfns into vm_area_struct */
	
#elif defined(USE_ALLOC_PAGES)
	/* TODO 3-3: Convert page(or page_area) to page frame number */
	pfn = page_to_pfn(page);
	//pfn = virt_to_phys((void*)page_area) >> PAGE_SHIFT;
	/* TODO 3-4: Remapping pfn into vm_area_struct */
	if(remap_pfn_range(vma, vma->vm_start, pfn, len, vma->vm_page_prot))
                        return -EAGAIN;	
#endif
	return 0;
}


struct file_operations my_fops = {
	.owner = THIS_MODULE,
	.read = mymmap_read,
	.open = mymmap_open,
	.release = mymmap_release,
	.mmap = mymmap_mmap,
};

static int mymmap_init(void)
{
	int err;
	int order = 0;
	int max_pages;
	int i;

	pr_info("[mymmap_init] Init module\n");
	err = register_chrdev_region(MKDEV(MY_MAJOR, MY_MINOR), 1, MODULE_NAME);
	if (err) {
		pr_info("[mymmap_init] register_chrdev_region: %d\n", err);
		return err;
	}

	cdev_init(&dev.cdev, &my_fops);
	cdev_add(&dev.cdev, MKDEV(MY_MAJOR, MY_MINOR), 1);

	/* Allocate memory by MAX_PAGES */
#if defined(USE_KMALLOC)
	/* TODO 1-1: Allocate contiguous memory by using kmalloc(), kmalloc_area */
	kmalloc_area = (char*)kmalloc(MAX_PAGES * PAGE_SIZE, GFP_KERNEL);
	area = kmalloc_area;
	pr_info("[mymmap_init] used kmalloc()");
#elif defined(USE_VMALLOC)
	/* TODO 2-1: Allocate virtually contiguous memory by using vmalloc(), vmalloc_area */
	vmalloc_area = (char*)vmalloc(MAX_PAGES * PAGE_SIZE);
	area = vmalloc_area;
	pr_info("[mymmap_init] used vmalloc()");
#elif defined(USE_ALLOC_PAGES)
	max_pages = MAX_PAGES;
	do{
		max_pages >>= 1;
		order++;
	}while(max_pages > 1);
	//printk("order: %d\n", order);
	/* TODO 3-1: Allocate page by using alloc_pages(), page */
	page = alloc_pages(GFP_KERNEL, order);
	/* TODO 3-2: Convert page to virtural address with page_area */
	page_area = (char*)page_to_virt(page);
	area = page_area;
	pr_info("[mymmap_init] used alloc_pages()");
#endif

#if defined(USE_KMALLOC) || defined(USE_VMALLOC) || defined(USE_ALLOC_PAGES)
	for (i = 0; i < MAX_PAGES * PAGE_SIZE; i += PAGE_SIZE) {
		area[i]     = 0xfa;
		area[i + 1] = 0xce;
		area[i + 2] = 0xb0;
		area[i + 3] = 0x0c;
		if (i == 0) {
			pr_info("[mymmap_init] 0x%02x%02x%02x%02x\n", 
					area[i]    & 0x00000000ff
					,area[i+1] & 0x00000000ff
				       	,area[i+2] & 0x00000000ff
				       	,area[i+3] & 0x00000000ff);
		}
	}
	dev.size = MAX_PAGES * PAGE_SIZE;
#else
	pr_info("You must to build with arguments"
		        " ALLOC=-DUSE_KMALLOC"
			" or ALLOC=-DUSE_VMALLOC"
			" or ALLOC=-DUSE_ALLOC_PAGES\n");
	dev.size = 0;
#endif


	return 0;
}

static void mymmap_exit(void)
{
	pr_info("[mymmap_exit] Exit module\n" );

	cdev_del(&dev.cdev);
	unregister_chrdev_region(MKDEV(MY_MAJOR, MY_MINOR), 1);
}

module_init(mymmap_init);
module_exit(mymmap_exit);
