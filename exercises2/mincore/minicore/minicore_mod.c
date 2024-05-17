#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <asm/uaccess.h>
#include <linux/pagemap.h>
#include <linux/slab.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/version.h>
#include <asm/io.h>

static  struct  class *sample_class;

static int sample_open(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);
}

static int sample_release(struct inode *inode, struct file *file)
{
        printk(KERN_INFO "%s\n", __FUNCTION__);
        return (0);

}
static inline bool xa_is_value(const void *entry)
{
	return (unsigned long)entry & 1;
}
#if 0
/*
 *  * Given the page we found in the page cache, return the page corresponding
 *   * to this index in the file
 *    */
static inline struct page *find_subpage(struct page *head, pgoff_t index)
{
	/* HugeTLBfs wants the head page regardless */
	if (PageHuge(head))
		return head;

	return head + (index & (thp_nr_pages(head) - 1));
}
/*
 *
 *
 * #define FGP_ACCESSED            0x00000001
 * #define FGP_LOCK                0x00000002
 * #define FGP_CREAT               0x00000004
 * #define FGP_WRITE               0x00000008
 * #define FGP_NOFS                0x00000010
 * #define FGP_NOWAIT              0x00000020
 * */
// taken from Linux source.
static struct page *find_get_incore_page(struct address_space *mapping, pgoff_t index)
{
    struct page *page = pagecache_get_page(mapping, index, FGP_LOCK | FGP_CREAT , 0);
    if (!page) 
        return page;
    if (!xa_is_value(page)) {
        return find_subpage(page, index);
    }
    return NULL;
}
#else
//static inline struct page *find_get_page(struct address_space *mapping,
//                                        pgoff_t offset)
static inline struct page *find_get_incore_page(struct address_space *mapping,
                                        pgoff_t offset)
{
        return pagecache_get_page(mapping, offset, 0, 0);
}
#endif
static unsigned char mincore_page(struct address_space *mapping, pgoff_t index)
{
    struct page *page = find_get_incore_page(mapping, index);
    unsigned char val = 0;

    if (page) {
        if (PageUptodate(page)) {
            val |= (1 << 0);
        }
        if (PageDirty(page)) {
            val |= (1 << 1);
        }
        put_page(page);
    }
    pr_info("pagecache index %lu , value %u , page @%p \n",index,val, page);
    return val;
}

static ssize_t
pch_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    struct file *mmap_fp = file->private_data;

    pgoff_t idx = 0;
    size_t iter_offset = 0;
    uint8_t *kern_buf = kvzalloc(PAGE_SIZE, GFP_KERNEL);
    if (!kern_buf) {
        return -EINVAL;
    }

    for (iter_offset = 0; iter_offset < count; iter_offset += PAGE_SIZE) {
        size_t iter_count = min(count - iter_offset, PAGE_SIZE);
        pgoff_t start_idx = *pos + iter_offset;
        pgoff_t last_idx = start_idx + iter_count - 1;

        size_t off = 0;

#if 0
        XA_STATE(xas, &mmap_fp->f_mapping->i_pages, start_idx);
        void *entry;
        rcu_read_lock();
        // entry: is it page or folio? whatever.
        xas_for_each(&xas, entry, last_idx) {
            uint8_t v = 1;
            if (xas_retry(&xas, entry) || xa_is_value(entry)) {
                v = 0;
            }
            kern_buf[off] = v;
            off++;
        }

        rcu_read_unlock();
#else
        for (idx = start_idx; idx <= last_idx; idx++) {
            //pr_info("pagecache index %lu \n",idx);
            kern_buf[off] = mincore_page(mmap_fp->f_mapping, idx>>PAGE_SHIFT);
            off++;
        }
#endif        

        if (copy_to_user(buf + iter_offset, kern_buf, iter_count)) {
            pr_err("failed to copy_to_user\n");
            kvfree(kern_buf);
            return -EINVAL;
        }
    }
    kvfree(kern_buf);
    *pos += count;
    return count;
}

static ssize_t
pch_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    char path[128];
    struct file *mmap_fp;

    if (file->private_data) {
        pr_err("private has been set.\n");
        return -EINVAL;
    }

    if (count > 64) {
        pr_err("path too long\n");
        return -EINVAL;
    }

    if (copy_from_user(path, buf, count)) {
        pr_err("fail to copy from user.\n");
        return -EINVAL;
    }
    path[count] = '\0';

    mmap_fp = filp_open(path, O_RDWR | O_LARGEFILE, 0);
    if (IS_ERR(mmap_fp)) {
        pr_err("fail to open %s\n", path);
        return PTR_ERR(mmap_fp);
    }

    // pr_info("pch: %s\n", path);

    file->private_data = mmap_fp;

    return count;
}

static int pch_release(struct inode *inode, struct file *file)
{
    if (file->private_data) {
        filp_close(file->private_data, 0);
    }
    return 0;
}

static const struct file_operations sample_ops = {
        .owner = THIS_MODULE,
        .open   = sample_open,
        .read  =  pch_read,
        .write = pch_write,
        .release = pch_release,
};

static int __init pch_init(void)
{
        int ret;
        ret = register_chrdev(42, "Sample", &sample_ops);
        sample_class = class_create(THIS_MODULE, "Sample");
        device_create(sample_class, NULL, MKDEV(42, 0), NULL, "Sample");
        return (ret);
}

static void __exit pch_exit(void)
{
        device_destroy(sample_class, MKDEV(42, 0));
        class_destroy(sample_class);
        unregister_chrdev(42, "Sample");
}

module_init(pch_init);
module_exit(pch_exit);

MODULE_LICENSE("GPL");
