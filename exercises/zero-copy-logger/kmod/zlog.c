#include <linux/uaccess.h>
#include <linux/debugfs.h>
#include <linux/sched.h>
#include <linux/fs.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/net.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>
#include <linux/timer.h>
#include <linux/jiffies.h>
#include <linux/dma-mapping.h>

#include "../zlog_ring_buf.h"

/////////////////////////

static int max_mem_size_order=MAX_ZLOG_MEM_ORDER;
module_param(max_mem_size_order,int,0660);

static int zlog_slot_size=ZLOG_SLOT_SIZE;
module_param(zlog_slot_size,int,0660);

#define STRINGIFY(x) #x
#define VERSION_STR(A,B,C) STRINGIFY(A) "." STRINGIFY(B) "." STRINGIFY(C)

#define ZLOG_VER_MAJOR 0
#define ZLOG_VER_MINOR 1
#define ZLOG_VER_REV   1

static char *version_string = VERSION_STR(ZLOG_VER_MAJOR, ZLOG_VER_MINOR, ZLOG_VER_REV);

//static DEFINE_MUTEX(zlog_device_mutex);
static DEFINE_RAW_SPINLOCK(logbuf_lock);

// zero copy logger kernel driver

static const char *zlog_proc_name = "zlog";

typedef struct zlog_dev_s {
    void                *ptr;
    dma_addr_t          dma_handle;
	wait_queue_head_t   wq;
    uint32_t            ref;
    uint32_t            enable;
	zlog_ring_t 		*zring;
	uint32_t 		    ring_size;
} zlog_dev_t;

zlog_dev_t g_zlogdev;

// for test
static void mykmod_timer_handler(unsigned long data);
static unsigned long onesec;
DEFINE_TIMER(mytimer, mykmod_timer_handler, 0, 0);

///////////////////////////////////

#ifdef DEBUG
 #define zlog_debug(fmt, ...) pr_debug(fmt "\n", ##__VA_ARGS__)
#else
 #define zlog_debug(fmt, ...) ({})
#endif

#define zlog_info(fmt, ...)   pr_info(fmt "\n", ##__VA_ARGS__)
#define zlog_err(fmt, ...)    pr_err(fmt "\n", ##__VA_ARGS__)


void zlog_free_vmem(unsigned long virtaddr, uint32_t buf_size);
int zlog_init_dev(void);

///////////////////////////////////

void zlog_vm_open(struct vm_area_struct *vma)
{
    zlog_dev_t *zlogdev;

	zlog_debug("zlog_vm_open");

    zlogdev = (zlog_dev_t*)vma->vm_private_data;
    if (zlogdev != NULL) {
        //zlogdev->ref++;
    }
}

void zlog_vm_close(struct vm_area_struct *vma)
{
    zlog_dev_t *zlogdev;

	zlog_debug("zlog_vm_close");

    zlogdev = (zlog_dev_t*)vma->vm_private_data;
    if (zlogdev != NULL) {
        //zlogdev->ref--;
    }
}

int zlog_vm_fault(struct vm_area_struct *vma, struct vm_fault *vmf)
{
	struct page *page;
	zlog_dev_t *zlogdev;
	zlog_ring_t *zring;
	unsigned long offset;

	zlog_debug("zlog_vm_fault");

	offset = vmf->pgoff << PAGE_SHIFT;

	zlog_debug("zlog_vm_fault:offset=%lu\n", offset);

	zlogdev = (zlog_dev_t*)vma->vm_private_data;
    if (zlogdev == NULL) {
	    zlog_err("zlog info is null");

        return 0;
    }

	zring = zlogdev->zring;

	if (zring) {
		page = virt_to_page(zring);
		get_page(page);
		vmf->page = page;
	}

	return 0;
}

static struct vm_operations_struct zlog_vm_ops =
{
	.open  = zlog_vm_open,
	.close = zlog_vm_close,
	.fault = zlog_vm_fault,
};

int zlog_vmem_mmap(struct file *filp, struct vm_area_struct *vma)
{
	zlog_dev_t *zlogdev;
	zlog_ring_t *zring;

	unsigned long start = vma->vm_start;
	unsigned long size = (vma->vm_end - vma->vm_start);
	void *pos;

	vma->vm_ops = &zlog_vm_ops;
	vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;
	vma->vm_private_data = filp->private_data;

	vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
	zlogdev = (zlog_dev_t*)vma->vm_private_data;
	zring = zlogdev->zring;
    
	pos = zring;
	vma->vm_pgoff = virt_to_phys(pos) >> PAGE_SHIFT;

	while (size > 0) {
		unsigned long pfn;

		pfn = virt_to_phys((void *) pos) >> PAGE_SHIFT;
		if (remap_pfn_range(vma, start, pfn, PAGE_SIZE, PAGE_SHARED))
			return -EAGAIN;

		start += PAGE_SIZE;
		pos += PAGE_SIZE;
		size -= PAGE_SIZE;
	}

	zlog_vm_open(vma);

	return 0;
}

int zlog_kmem_mmap(struct file *filp, struct vm_area_struct *vma)
{
	zlog_dev_t *zlogdev;
    int ret;
    long length = vma->vm_end - vma->vm_start;

	zlogdev = (zlog_dev_t*)filp->private_data;

    /* check length - do not allow larger mappings than the number of pages allocated */
    if (length > max_mem_size_order * PAGE_SIZE) {
        return -EIO;
    }

    if (vma->vm_pgoff == 0) {
        ret = dma_mmap_coherent(NULL, vma, zlogdev->ptr, zlogdev->dma_handle, length);
    } 
    else {
        vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
        vma->vm_flags |= VM_IO;

        ret = remap_pfn_range(vma, vma->vm_start,
                              PFN_DOWN(virt_to_phys(bus_to_virt(zlogdev->dma_handle))) +
                              vma->vm_pgoff, length, vma->vm_page_prot);
    }

    /* map the whole physically contiguous area in one piece */
    if (ret < 0) {
        zlog_err("zlog_kmem_mmap: remap failed (%d)", ret);
        return ret;
    }

    return 0;
}

int zlog_open(struct inode *inode, struct file *filp)
{
    if (g_zlogdev.ref > 0) {
		return -EBUSY;
    }

    filp->private_data = &g_zlogdev;
    g_zlogdev.ref++;

    return 0;
}

int zlog_release(struct inode *inode, struct file *filp)
{
    g_zlogdev.ref--;
	filp->private_data = NULL;
	
	return 0;
}

ssize_t zlog_read(struct file *filp, char __user *buf, size_t len, loff_t *off)
{
    int ret=0, rlen;
    uint32_t i;
	zlog_ring_t *zring;
	zlog_dev_t *zlog_info;
	zlog_read_t zread;

	rlen = sizeof(zlog_read_t);
    if (rlen != len) {
	    zlog_info("zlog_read: dest buf len is not same: %d != %lu", rlen, len);
		return -EINVAL;
    }

    if (copy_from_user(&zread, buf, rlen)) {
	    zlog_info("zlog_read: cannot read buf");
        return -EINVAL;
	}

	zlog_info = (zlog_dev_t*)filp->private_data;
	zring = zlog_info->zring;

	wait_event_interruptible(zlog_info->wq, zring->head != zring->tail);

    raw_spin_lock_irq(&logbuf_lock);
	zread.start = zring->tail;

	if (zring->head > zring->tail) {
		zread.cnt = zring->head - zring->tail;
	}
	else {
		zread.cnt = (zring->slot_cnt - zring->tail) + zring->head;
	}

	for (i=0; i<zread.cnt; i++) {
        uint32_t *slot = (uint32_t*)zlog_get_slot_addr(zring, zread.start+i);

        // mark the owner of slot
        *slot = zread.owner;
	}

    if (copy_to_user(buf, &zread, rlen)) {
        ret = -EFAULT;
	}

    zlog_info("owner=%d, start=%d, cnt=%d\n", zread.owner, zread.start, zread.cnt);
    raw_spin_unlock_irq(&logbuf_lock);

	return ret;
}

ssize_t zlog_write(struct file *filp, const char __user *buf, size_t len, loff_t *off)
{

#if 0
	zlog_debug("zlog_write");
	struct mmap_info *info;
	offset = (vmf->address - vma->vm_start) >> PAGE_SHIFT;
	offset = (vmf->address - vma->vm_start) >> PAGE_SHIFT;
	page = bo->pages[offset];

	err = vm_insert_page(vma, vmf->address, page);
	switch (err) {
	case -EAGAIN:
	page = bo->pages[offset];

	err = vm_insert_page(vma, vmf->address, page);
	switch (err) {
	case -EAGAIN:

	zlog_debug("zlog_write\n");
	info = filp->private_data;
    if (copy_from_user(info->data, buf, min(len, (size_t)BUFFER_SIZE))) {
        return -EFAULT;
	} 

	return len;
#else
	return len;
#endif
}

/////////////////////////////////////////////////////
//

static const struct file_operations log_ring_fops = {
	//.mmap = zlog_vmem_mmap,
	.mmap = zlog_kmem_mmap,
	.open = zlog_open,
	.read = zlog_read,
	.release = zlog_release,
	.write = zlog_write,
};

void zlog_free_vmem(unsigned long virtaddr, uint32_t buf_size)
{
	uint32_t order, size;
	unsigned long addr = virtaddr;

	size = PAGE_ALIGN(buf_size);
	order = get_order(size);

	while (size > 0) {
		ClearPageReserved(virt_to_page(addr));
		addr += PAGE_SIZE;
		size -= PAGE_SIZE;
	}

	free_pages((unsigned long) virtaddr, order);
}

uint32_t zlog_alloc_vmem(char **virt_addr)
{
	uint32_t order, size = 0, allocted_size = 0;
	unsigned long vaddr, addr;
	uint32_t i;

	for (i=max_mem_size_order; i >= 0; i --) {
		size = PAGE_SIZE * (1 << i);
		size = PAGE_ALIGN(size);
		order = get_order(size);

		allocted_size = size;

		vaddr = __get_free_pages(GFP_KERNEL|GFP_NOWAIT, order);
		addr = vaddr;

		if (vaddr == 0) {
			size = 0;
			zlog_err("Cannot alloc log mem: continuous mem=%u", size);
			continue;
		}

		memset((char*)vaddr, 0, size);

		while (size > 0) {
			SetPageReserved(virt_to_page(addr));
			addr += PAGE_SIZE;
			size -= PAGE_SIZE;
		}

		//*phys_addr = (char*)virt_to_phys((void *)vaddr);
		*virt_addr = (char*)vaddr;

		break;
	}

	return allocted_size;
}

void zlog_free_kmem(void *ptr, dma_addr_t dma_handle)
{
	dma_free_coherent (NULL, max_mem_size_order * PAGE_SIZE, ptr, dma_handle);
}

uint32_t zlog_alloc_kmem(zlog_dev_t *zlogdev)
{
    int *alloc_area;
    uint32_t alloced_size = max_mem_size_order * PAGE_SIZE;

    zlogdev->ptr = dma_alloc_coherent (NULL, alloced_size, &zlogdev->dma_handle, GFP_KERNEL);

    if (!zlogdev->ptr) {
        zlog_err("zlog_alloc_kmem: dma_alloc_coherent error");
        return 0;
    }

    zlog_info("zlog_alloc_kmem: physical address is 0x%x", (uint32_t)zlogdev->dma_handle);
    zlog_info("zlog_alloc_kmem: bus_to_virt 0x%llx", virt_to_phys(bus_to_virt(zlogdev->dma_handle)));

    alloc_area = zlogdev->ptr;
    // XXX: access all memory range
    memset(alloc_area, 0, alloced_size);

    return alloced_size;
}

int zlog_init_dev(void)
{
    zlog_dev_t *zlogdev;
	zlog_ring_t *zring = NULL;
	uint32_t len = 0;
	int i, cnt, ret;
	//char *u = NULL;

	zlog_info("Alloc zLog driver memory");

    /*
    if (!mutex_trylock(&zlog_device_mutex)) {
    zlog_err("Another process is accessing the zlog device, try again");
    return -EBUSY;
    }

	zlogdev = kmalloc(sizeof(zlog_dev_t), GFP_ATOMIC);
    if (zlogdev == NULL) {
		zlog_err("Fail to alloc zlog info");
		return -ENOMEM;
    }
    */

    zlogdev = &g_zlogdev;

	memset(zlogdev, 0, sizeof(zlog_dev_t));
	init_waitqueue_head(&zlogdev->wq);

	//len = zlog_alloc_vmem(&u);
	//zring = (zlog_ring_t*)u;

    len = zlog_alloc_kmem(zlogdev);
    zring = (zlog_ring_t*)zlogdev->ptr;

	if (zring == NULL || len == 0) {
		zlog_err("Fail to alloc page memory");
		ret = -ENOMEM;
        goto ERROR;
	}

	zring->magic = 0x26;
	zring->ver   = 0x63;
	zring->head  = 0;
	zring->tail  = 0;
    zring->ring_mem_size = len;
	zring->slot_size = (uint16_t)zlog_slot_size;

	zlogdev->ring_size = len;
	zlog_info("zLog ring total memory size: %u bytes", len);

	len = len - (uint32_t)sizeof(zlog_ring_t);
	cnt = len / (uint32_t)zring->slot_size;
	zring->slot_cnt = cnt;

	zlog_info("zLog ring slot memory size: %u bytes", len);
	zlog_info("zLog slots: %u slots of %d bytes", cnt, zring->slot_size);

	// init all slots
	for (i=0; i<zring->slot_cnt; i++) {
		//char *base = (char*)zring->data + (i*zring->slot_size);
        char *base = zlog_get_slot_addr(zring, i);
        uint32_t *slot = (uint32_t*)base;

        // XXX: access the memory
        *slot = i;
	}

	zlogdev->zring = zring;
    zlogdev->enable = 1;

    return 0;

ERROR:
    /*
    if (zlogdev != NULL) {
        kfree(zlogdev);
    }
    */

    return ret;
}


int zlog_init(void)
{
    int ret;

	zlog_info("Start zLog driver v%s", version_string);

    ret = zlog_init_dev();
    if (ret != 0) {
        return ret;
    }

	proc_create(zlog_proc_name, 0, NULL, &log_ring_fops);

    // for test
	onesec = msecs_to_jiffies(1000 * 1);
	mod_timer(&mytimer, jiffies + onesec);

	return 0;
}

void zlog_exit(void)
{
	del_timer(&mytimer);

    g_zlogdev.enable = 0;

	if (waitqueue_active(&g_zlogdev.wq)) {
		wake_up_interruptible(&g_zlogdev.wq);
	}

	remove_proc_entry(zlog_proc_name, NULL);

	zlog_info("Stop zLog driver v%s", version_string);

	if (g_zlogdev.zring) {
		//zlog_free_vmem((unsigned long)g_zlogdev.zring, g_zlogdev.ring_size);
        
		zlog_free_kmem(g_zlogdev.ptr, g_zlogdev.dma_handle);
        g_zlogdev.ptr = NULL;
        g_zlogdev.dma_handle = 0;

        g_zlogdev.zring = NULL;
	}

}

int zlog_put_log(char *buf, int size) 
{
    zlog_ring_t *zring;
    uint32_t next_idx;
    char *slot;
    uint32_t *owner;

    if (g_zlogdev.enable != 1) {
        return -EBUSY;
    }

    raw_spin_lock_irq(&logbuf_lock);

    zring = g_zlogdev.zring;
    if (zring == NULL) {
        return -EBUSY;
    }

    slot = zlog_get_slot_addr(zring, zring->head);
    owner = (uint32_t*)slot;
    // reset owner
    *owner = 0;
    slot += 4;

    if (size > (zring->slot_size - 4)) {
        size = zring->slot_size - 4;
    }

    memcpy(slot, buf, size);

    next_idx = (zring->head + 1) % zring->slot_cnt;
    zring->head = next_idx;
    if (zring->tail == next_idx) {
        //net_info_ratelimited("zlog ring is full, overwrite tail(%d)\n", zring->tail);
        zring->tail = (next_idx + 1) % zring->slot_cnt;
    }

    zring->head = next_idx;

    raw_spin_unlock_irq(&logbuf_lock);

    if (waitqueue_active(&g_zlogdev.wq)) {
        wake_up_interruptible(&g_zlogdev.wq);
    }

    return 0;
}

EXPORT_SYMBOL(zlog_put_log);

////////////////
// for test

int zlog_push_log(uint64_t jiff)
{
    static int msg_cnt;
    zlog_t log;

    msg_cnt ++;
    sprintf(log.msg, "msg[%d]: %llu", msg_cnt, jiff);

    zlog_put_log(log.msg, sizeof(log.msg));

    return 0;
}

static void mykmod_timer_handler(unsigned long data)
{
	zlog_push_log(jiffies);

	mod_timer(&mytimer, jiffies + onesec);
}

module_init(zlog_init)
module_exit(zlog_exit)
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Zerocopy Logger Driver");
MODULE_VERSION(VERSION_STR(ZLOG_VER_MAJOR, ZLOG_VER_MINOR, ZLOG_VER_REV));

