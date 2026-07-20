#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/version.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(3,0,0)
#include <linux/memblock.h>
#endif
#include <linux/ioport.h>
#include <linux/highmem.h>
#include <linux/fs.h>
#include <linux/mm_types.h>
#include <linux/kexec.h>
#include <linux/uaccess.h>
#include <linux/proc_fs.h>
#include <asm-generic/iomap.h>
#include <asm/tlbflush.h>
#include <linux/highmem.h>
#include <linux/efi.h>
#include <asm/bios_ebda.h>
#include <linux/ioport.h>
#include <linux/io.h>
#include <linux/memory_hotplug.h>
#include <linux/hashtable.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/vmalloc.h>
#include <linux/cpu.h>

#include <linux/pagemap.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(4,10,0)
#include <asm-generic/io.h>
#include <asm/e820/types.h>
#endif
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,6,0)
#include <linux/kmsg_dump.h>
#endif

extern void vunmap(const void *addr);
extern void *vmap(struct page **pages, unsigned int count,
                unsigned long flags, pgprot_t prot);

#define FILLER_ENTRY_NAME "filler_start"

#define RESERVE_BUFFER_SZ		(PAGE_SIZE*2)
#define PAGES_PER_RESERVE_BUFFER	RESERVE_BUFFER_SZ/PAGE_SIZE

#define FILLER_SIGN 0xdeadbeef

#define DMA "DMA"
#define DMA_LEN strlen(DMA)

#define NORMAL "Normal"
#define NORMAL_LEN strlen(NORMAL)


static atomic_t filler_opened = ATOMIC_INIT(0);

static struct proc_dir_entry *filler_entry;

static int filler_open(struct inode *inode, struct file *file)
{
        if (atomic_cmpxchg(&filler_opened, 0, 1))
                return -EBUSY;

        return 0;
}

static int filler_release(struct inode *inode, struct file *file)
{
	atomic_set(&filler_opened, 0);
        return 0;
}

static int walk_all_free_pages_kmap_and_write(char *kbuf, struct list_head *list)
{
	struct page *p, *next;
	void *kmapped_buffer = NULL;

	if (!list)
		return -EINVAL;

	list_for_each_entry_safe(p, next, list, lru) {
		kmapped_buffer = kmap(p);
		if (!kmapped_buffer) {
			pr_err("[%s] cannot kmap page [%p]\n", __func__, page_address(p));
			continue;
		}
		memcpy_toio(kmapped_buffer, (void*)kbuf, RESERVE_BUFFER_SZ);
		kunmap(kmapped_buffer);

		pr_info("[%s] written to page [%p]\n", __func__, page_address(p));
	}

	return 1;
}

static int walk_all_free_pages_kmap_and_read(char *kbuf, struct list_head *list)
{
	struct page *p, *next;
	unsigned long signature = 0;

	if (!list)
		return -EINVAL;

	list_for_each_entry_safe(p, next, list, lru) {
		void *kmapped_buffer = kmap(p);
		if (!kmapped_buffer) {
			pr_err("[%s] Cannot kmap page [%p]\n", __func__, p);
		}

		memcpy_fromio((void*)&signature, (void*)kmapped_buffer, sizeof(unsigned long));

		if (signature == FILLER_SIGN) {
			memcpy_fromio((void*)kbuf, (void*)kmapped_buffer, RESERVE_BUFFER_SZ);
			kunmap(kmapped_buffer);
			pr_info("READ from  [%p][%lx]\n",
				page_address(p), signature);
			return 0;
		} else {
			pr_info("NOT READ from  [%p][%p][%lx]\n",
				page_address(p), kmapped_buffer, signature);
		}
		kunmap(kmapped_buffer);
	}

	return -ESRCH;
}

static int walk_all_mtypes(char *kbuf, struct free_area *free_area,
			   int (*walk_cb)(char *, struct list_head *))
{
	struct list_head *list;
	unsigned int type;
	int ret = -ESRCH;

	if (! kbuf || !walk_cb) {
		pr_err("[%s] error kbuf[%p] free_area[%p] cb[%p]\n",
			kbuf, free_area, walk_cb);
		return -EINVAL;
	}


	for (type = 0; type < MIGRATE_TYPES; type++) {
		pr_info("[%s] start walk through type [%d] nr_free[%u]\n",
			__func__, type, free_area->nr_free);
		list = &free_area->free_list[type];
		ret = walk_cb(kbuf, list);
		if (!ret)
			break;
	}

	return ret;
}

static int walk_all_orders(char *kbuf, struct zone *zone,
			   int (*walk_cb)(char *, struct list_head *))
{
	struct free_area *free_area;
	unsigned long flags;
	unsigned int order;
	int ret = -ESRCH;

	spin_lock_irqsave(&zone->lock, flags);
	for (order = PAGES_PER_RESERVE_BUFFER; order < MAX_ORDER; order++) {
		pr_info("[%s] start walk through order [%d]\n", __func__, order);
		free_area = &zone->free_area[order];
		ret = walk_all_mtypes(kbuf, free_area, walk_cb);
		if (!ret)
			break;
	}
	spin_unlock_irqrestore(&zone->lock, flags);

	return ret;
}

static int walk_all_zones(char *kbuf, struct pglist_data *pgdat, bool write)
{
	struct zone *zone;
	unsigned long flags;
	int (*cb_kmap)(char *, struct list_head *);
	int ret = -ESRCH;

	if (!pgdat) {
		return -EINVAL;
	}

	cb_kmap = (write) ? walk_all_free_pages_kmap_and_write : walk_all_free_pages_kmap_and_read; 

	pgdat_resize_lock(pgdat, &flags);
	for (zone = pgdat->node_zones; zone < pgdat->node_zones + MAX_NR_ZONES; zone++) {
		if (!populated_zone(zone)) {
			pr_err("[%s] 0 zone [%p][%s] not populated [%lx]-[%lx]\n", __func__,
				 zone, zone->name, zone->zone_start_pfn, zone_end_pfn(zone));
			continue;
		}

		if (!strncmp(zone->name, DMA, DMA_LEN)) {
			ret = walk_all_orders(kbuf, zone, cb_kmap);
			if (!ret)
				break;
		}

		if (!strncmp(zone->name, NORMAL, NORMAL_LEN)) {
			ret = walk_all_orders(kbuf, zone, cb_kmap);
			if (!ret)
				break;
		}
	}
	pgdat_resize_unlock(pgdat, &flags);

	return ret;
}


static int walk_all_nodes(char *kbuf, bool write)
{
	struct pglist_data *pgdat;
	unsigned int cpu;
	int nid, ret = -ESRCH;

	pr_info("[%s] start walking\n", __func__);
	get_online_cpus();
	for_each_online_cpu(cpu) {
		nid = cpu_to_node(cpu);
		pgdat = NODE_DATA(nid);

		pr_info("[%s] nid [%d] pgdat [%p]\n", __func__, nid, pgdat);
		ret = walk_all_zones(kbuf, pgdat, write);
		if (!ret)
			break;
	}
	put_online_cpus();

	pr_info("[%s] end walking \n", __func__);

	__flush_tlb_all();

	return ret;
}

static int write_to_free_mem(char *kbuf)
{
	return walk_all_nodes(kbuf, 1);
}

static ssize_t filler_write(struct file *file, const char __user *buf,
			     size_t count, loff_t *ppos)
{
        char buffer[RESERVE_BUFFER_SZ+1] = { 0 };
        unsigned long signature = FILLER_SIGN;

        memset((void*)buffer, 'A', RESERVE_BUFFER_SZ);
        memcpy((void*)buffer, (void*)&signature, sizeof(unsigned long));

	write_to_free_mem(buffer);

	return count;
}

static int read_from_free_mem(char *kbuf)
{
	return walk_all_nodes(kbuf, 0);
}

static ssize_t filler_read(struct file *file, char __user *buf, size_t size, loff_t *ppos)
{
	char kbuf[RESERVE_BUFFER_SZ+1] = { 0 };
	int err;

	if (!buf) {
		printk(KERN_ERR "Invalid params need buf[%p]\n", buf);
		return -EINVAL;
	}

	err = read_from_free_mem(kbuf);
	if (err)
		return -ESRCH;

	printk(KERN_INFO "DUMPED buffer [%x][%x][%x][%x]\n", kbuf[0], kbuf[1], kbuf[2], kbuf[3]);
	copy_to_user(buf, (void *)kbuf, RESERVE_BUFFER_SZ);

	return RESERVE_BUFFER_SZ;
}


#if LINUX_VERSION_CODE < KERNEL_VERSION(5,6,0)
struct file_operations filler_fops = {
	.owner	= THIS_MODULE,
	.read	= filler_read,
	.write	= filler_write,
	.open	= filler_open,
	.release= filler_release,
};
#else
struct proc_ops filler_fops = {
        .proc_open      = filler_open,
        .proc_read      = filler_read,
        .proc_write     = filler_write,
	.proc_release	= filler_release,
};
#endif


static int __init filler_init(void)
{
	filler_entry = proc_create(FILLER_ENTRY_NAME, 0, NULL,
					&filler_fops);
	if (!filler_entry) {
		printk(KERN_ERR "Cannot create filler entry\n");
		return -ENODEV;
	}

	return 0;
}
module_init(filler_init);


static void __exit filler_exit(void)
{
	if (filler_entry)
		remove_proc_entry(FILLER_ENTRY_NAME, NULL);
}
module_exit(filler_exit);

MODULE_LICENSE("GPL");
