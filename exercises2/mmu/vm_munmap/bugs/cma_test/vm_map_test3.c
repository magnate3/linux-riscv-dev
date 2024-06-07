#include <linux/miscdevice.h>
#include <linux/platform_device.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/mm.h>
#include <linux/list.h>
#include <linux/mutex.h>
#include <linux/debugfs.h>
#include <linux/mempolicy.h>
#include <linux/sched.h>
#include <linux/module.h>
#include <asm/io.h>
#include <asm/uaccess.h>
#include <asm/cacheflush.h>
#include <linux/dma-mapping.h>
#include <linux/export.h>
#include <linux/syscalls.h>
#include <linux/mman.h>

#include "cma_mem.h"

#define DEVICE_NAME "cma_mem" 

#define MEM_DEBUG 1

enum cma_status{
	UNKNOW_STATUS = 0,
	HAVE_ALLOCED = 1,
	HAVE_MMAPED =2,
};

struct cmamem_dev {
	unsigned int count;
	struct miscdevice dev;
	struct mutex cmamem_lock;
};

struct cmamem_block {
	char name[10];
	char is_use_buffer;
	char is_free;
	int id;
	unsigned long offset;
	unsigned long len;
	unsigned long phy_base;
	unsigned long mem_base;
	void *kernel_base;
	struct list_head memqueue_list;
};

struct current_status{
		int status;
		int id_count;
		dma_addr_t phy_base;
};

static struct current_status cmamem_status;
static struct cmamem_dev cmamem_dev;
static struct cmamem_block *cmamem_block_head;
static int mem_block_count = 0;

static void dump_mem(struct cmamem_block *memory_block)
{
	printk("%s:CMA name:%s\n",__func__,  memory_block->name);
	printk("%s:CMA id:%d\n",__func__,  memory_block->id);
	printk("%s:Is usebuf:%d\n",__func__,  memory_block->is_use_buffer);	
	printk("%s:PHY Base:0x%08lx\n",__func__,  memory_block->phy_base);
	printk("%s:KER Base:0x%016lx\n",__func__,  (unsigned long)(memory_block->kernel_base));	
	printk("%s:USR Base:0x%08lx\n",__func__,  memory_block->mem_base);
}
static long cmamem_alloc(struct file *file, unsigned long arg)
{
	struct cmamem_block *memory_block;
	struct mem_block cma_info_temp;
	int size;
	int ret;

	if ((ret = copy_from_user(&cma_info_temp, (void __user *)arg,
	sizeof(struct mem_block))))
	{
		printk(KERN_ERR"cmamem_alloc:copy_from_user error:%d\n", ret);
		return -1;
	}
	
	if(cma_info_temp.name[0] == '\0')
	{
		printk(KERN_ERR "%s, no set mem name, please set\n", __func__);
		return -1;
	}

	if(cma_info_temp.len){

		size = PAGE_ALIGN(cma_info_temp.len);

		cma_info_temp.len = size;
#ifdef	MEM_DEBUG
	//	printk(KERN_INFO "%s len:%ld, is_use_buffer:%d\n", __func__, cma_info_temp.len, cma_info_temp.is_use_buffer);
#endif
		if(cma_info_temp.is_use_buffer)
			cma_info_temp.kernel_base = dma_alloc_writecombine(NULL, size, (dma_addr_t *)(&(cma_info_temp.phy_base)), GFP_KERNEL);
		else
			cma_info_temp.kernel_base = dma_alloc_coherent(NULL, size, (dma_addr_t *)(&(cma_info_temp.phy_base)), GFP_KERNEL);


		if (!cma_info_temp.phy_base){
				printk(KERN_ERR "dma alloc fail:%d!\n", __LINE__);
				return -ENOMEM;
			}

		cma_info_temp.id = ++mem_block_count;

		cmamem_status.phy_base = 	cma_info_temp.phy_base;
		cmamem_status.id_count =  	cma_info_temp.id;
		cmamem_status.status = HAVE_ALLOCED;

		cma_info_temp.mem_base = vm_mmap(file, 0, size, PROT_READ | PROT_WRITE, MAP_SHARED, 0);
		if(cma_info_temp.mem_base < 0)
		{
				printk(KERN_ERR "do_mmap fail:%d!\n", __LINE__);
				cma_info_temp.id = --mem_block_count;
				return -ENOMEM;
		}
		printk(KERN_INFO "cma_info_temp.mem_base:0x%lx\n", cma_info_temp.mem_base);	
		//mem_block_count ++;

	}
	else{
	
		printk(KERN_ERR"cmamem_alloc: the len is NULL\n");
		return -1;
	}	

	if(copy_to_user((void __user *)arg, (void *)(&cma_info_temp), sizeof(struct mem_block)))
		return -EFAULT;

	/* setup the memory block */
	memory_block = (struct cmamem_block *)kmalloc(sizeof(struct cmamem_block), GFP_KERNEL);
	if(memory_block == NULL)
	{
		printk(KERN_ERR "%s error line:%d\n", __func__, __LINE__);
		mem_block_count --;
		return -1;
	}

	if(cma_info_temp.name[0] != '\0')
		memcpy(memory_block->name, cma_info_temp.name, 10);

	memory_block->id		=	cma_info_temp.id;
	memory_block->is_free	=	0;
	memory_block->is_use_buffer	=	cma_info_temp.is_use_buffer;
	memory_block->mem_base 	=	cma_info_temp.mem_base;
	memory_block->kernel_base 	=	cma_info_temp.kernel_base;
	memory_block->phy_base 	=	cma_info_temp.phy_base;
	memory_block->len		=	cma_info_temp.len;

#ifdef	MEM_DEBUG
	dump_mem(memory_block);
#endif	
#ifdef CMA_TEST
	int i;
	for(i = 0; i < 10; i++)
		((char *)(cma_info_temp.kernel_base))[i] = (cma_info_temp.id * i);
#endif
	/* add to memory block queue */
	list_add_tail(&memory_block->memqueue_list, &cmamem_block_head->memqueue_list);

	return 0;
}
static int cmamem_free(struct file *file, unsigned long arg)
{
	struct cmamem_block *memory_block;
	struct mem_block cma_info_temp;
	int ret;

	if ((ret = copy_from_user(&cma_info_temp, (void __user *)arg,
	sizeof(struct mem_block))))
	{
		printk(KERN_ERR"cmamem_alloc:copy_from_user error:%d\n", ret);
		return -1;
	}
	printk(KERN_INFO "will delete the mem name:%s\n", cma_info_temp.name);

	list_for_each_entry(memory_block, &cmamem_block_head->memqueue_list, memqueue_list)
	{
		if(memory_block){
			//if(memory_block->id == cma_info_temp.id || !strcmp(cma_info_temp.name, memory_block->name)){
			if(!strcmp(cma_info_temp.name, memory_block->name)){
				if(memory_block->is_free == 0){

					printk(KERN_INFO "delete the mem id:%d, name:%s\n", cma_info_temp.id, cma_info_temp.name);

					vm_munmap(memory_block->mem_base, memory_block->len);
					
					if(memory_block->is_use_buffer)
						dma_free_coherent(NULL,	memory_block->len, memory_block->kernel_base, memory_block->phy_base);
					else
						dma_free_writecombine(NULL, memory_block->len, memory_block->kernel_base, memory_block->phy_base);

					memory_block->is_free = 1;

					list_del(&memory_block->memqueue_list);
        			
					break;
				}

			}
		}
	}  

	return 0;
}
static int cmamem_freeall(void)
{
	struct cmamem_block *memory_block;

	printk(KERN_INFO "will delete all cma mem\n");

	list_for_each_entry(memory_block, &cmamem_block_head->memqueue_list, memqueue_list)
	{
		if(memory_block && memory_block->id > 0){
				if(memory_block->is_free == 0){
					printk(KERN_INFO "delete the mem id:%d, name:%s\n", memory_block->id, memory_block->name);
						
					if(memory_block->is_use_buffer)
						dma_free_coherent(NULL, memory_block->len, memory_block->kernel_base, memory_block->phy_base);
					else
						dma_free_writecombine(NULL, memory_block->len, memory_block->kernel_base, memory_block->phy_base);

					memory_block->is_free = 1;

				}
		}
	}  

	return 0;
}
static long cmamem_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{

	int ret = 0;

	switch(cmd){
		case CMEM_ALLOCATE:
		{
			printk(KERN_ERR"cmamem_ioctl:CMEM_ALLOCATE\n");
			mutex_lock(&cmamem_dev.cmamem_lock);

			ret = cmamem_alloc(file, arg);
			if(ret < 0)
				goto alloc_err;
	
			mutex_unlock(&cmamem_dev.cmamem_lock);
			break;
		}
		case CMEM_UNMAP:
		{
			printk(KERN_ERR"cmamem_ioctl:CMEM_UNMAP\n");
			mutex_lock(&cmamem_dev.cmamem_lock);

			ret = cmamem_free(file, arg);
			if(ret < 0)
				goto free_err;
	
			mutex_unlock(&cmamem_dev.cmamem_lock);
			break;
		}			
		default:
		{
			printk(KERN_INFO "cma mem not support command\n");
			break;
		}
	}
	return 0;
	alloc_err:
		mutex_unlock(&cmamem_dev.cmamem_lock);
		printk(KERN_ERR "%s alloc error\n", __func__);
		return ret;
	free_err:
		mutex_unlock(&cmamem_dev.cmamem_lock);
		printk(KERN_ERR "%s free error\n", __func__);

	return ret;
}


static int cmamem_mmap(struct file *filp, struct vm_area_struct *vma)
{
	unsigned long start = vma->vm_start;
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
	unsigned long page, pos;

	//if(size > MMAP_MEM_SIZE)
	//	return -EINVAL; 
	if(cmamem_status.status != HAVE_ALLOCED)
	{
		printk(KERN_ERR"%s, you should allocted memory firstly\n", __func__);
		return -EINVAL; 
	}
	
	
//	printk( "cmamem_mmap:vma:start=0x%08x offset=0x%08x\n", (unsigned int)start, (unsigned int)offset );

	pos = (unsigned long)cmamem_status.phy_base + offset;
	page = pos >> PAGE_SHIFT ;

//	printk( "cmamem_status.phy_base:0x%08x\n", (unsigned int)cmamem_status.phy_base);
	
	if( remap_pfn_range( vma, start, page, size, PAGE_SHARED )) {
		return -EAGAIN;
	}
	else{
	//	printk( "remap_pfn_range %u\n success\n", (unsigned int)page );
	}
	vma->vm_flags &= ~VM_IO; 
	vma->vm_flags |=  (VM_DONTEXPAND | VM_DONTDUMP);

	cmamem_status.status = HAVE_MMAPED;
	return 0;
}

static struct file_operations dev_fops = {  
    .owner          = THIS_MODULE,  
    .unlocked_ioctl = cmamem_ioctl,  
    .mmap = cmamem_mmap,
};



static int __init cmamem_init(void)
{
	printk(KERN_INFO "%s\n", __func__);
	mutex_init(&cmamem_dev.cmamem_lock);
//NIT_LIST_HEAD(&cmamem_dev.info_list);
	cmamem_dev.count = 0;
	cmamem_dev.dev.name = DEVICE_NAME;
	cmamem_dev.dev.minor = MISC_DYNAMIC_MINOR;
	cmamem_dev.dev.fops = &dev_fops;

	cmamem_block_head = (struct cmamem_block *)kmalloc(sizeof(struct cmamem_block), GFP_KERNEL);
	cmamem_block_head->id = -1;
	mem_block_count = 0;
	INIT_LIST_HEAD(&cmamem_block_head->memqueue_list);
/*	 
	cmamem_status.status = UNKNOW_STATUS;
	cmamem_status.id_count = -1;
	cmamem_status.phy_base = 0;
*/
	return misc_register(&cmamem_dev.dev);
}

static void __exit cmamem_exit(void)  
{  
    printk(KERN_ERR"%s\n", __func__);
	cmamem_freeall();
	misc_deregister(&cmamem_dev.dev);  
} 


module_init(cmamem_init);
module_exit(cmamem_exit);
MODULE_LICENSE("GPL");
