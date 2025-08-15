#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/mm.h>
#include <linux/device.h>
#include <linux/cdev.h>

#include <linux/init.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/ioctl.h>

#define DEVICE_NAME "physmem"
#define IOCTL_PA_ALLOC _IOWR('p', 0, size_t)
#define IOCTL_PA_FREE _IOR('p', 1, void *)

static dev_t  dev;
static struct cdev c_dev;
static struct class *cl;

struct addr_list {
        struct addr_list *next;
        struct addr_list *prev;
        void *p;
        void *k;
        size_t size;
};

struct addr_list *al;
static int pa_open_count = 0;
struct mutex pa_mutex;

static int pa_open(struct inode *inode, struct file *file) {
        if (pa_open_count) {
                return -EBUSY;
        }

        pa_open_count++;
        try_module_get(THIS_MODULE);
        return 0;
}

static int pa_release(struct inode *inode, struct file *file) {
        pa_open_count--;
        module_put(THIS_MODULE);
        return 0;
}

struct addr_list * create_node(void *k, void *p, size_t size) {
        struct addr_list *l = kmalloc(sizeof(struct addr_list), GFP_KERNEL);
        if (!l) return l;
        l->next = NULL;
        l->prev = NULL;
        l->k = k;
        l->p = p;
        l->size = size;
        return l;
}

int add_addr(void *k, void *p, unsigned int size) {
        struct addr_list *l;
        struct addr_list *tal;

        l = create_node(k, p, size);
        if (!l) return -1;
        if (!al) {
                al = l;
                return 0;
        }

        tal = al;
        while(tal->next) tal = tal->next;
        tal->next = l;
        l->prev = tal;
        return 0;
}

int find_and_delete(void *p) {
        struct addr_list *tal;
        if (!al) return -1;

        tal = al;

        do {
                if (tal->p == p) {
                        if (tal->prev) tal->prev->next = tal->next;
                        if (tal->next) tal->next->prev = tal->prev;

                        if  (tal == al) {
                                if (al->next) {
                                        al = al->next;
                                        al->prev = NULL;
                                }
                                else al = NULL;
                        }
                        kfree(tal->k);
                        kfree(tal);
                        return 0;
                }

                tal = tal->next;
        } while(tal);

        return -1;
}

size_t page_align(size_t s) {
        return ALIGN(s, PAGE_SIZE);
}

static long pa_ioctl(struct file *file,
                 unsigned int ioctl_num,
                 unsigned long ioctl_param)
{
        size_t size = 0;
        void *p, *k;
        int err;
        int ret = 0;
        mutex_lock(&pa_mutex);

        switch (ioctl_num) {
                case IOCTL_PA_ALLOC:
                        printk(KERN_INFO "IOCTL_PA_ALLOC\n");

                        err = copy_from_user(&size, (size_t *)ioctl_param, sizeof(size));
                        printk(KERN_INFO "Received the size: %ld\n", size);
                        if (err) {
                                ret = -EINVAL;
                                goto END;
                        }

                        if (size > 5 * 1024 * 1024) {
                                ret = -ENOMEM;
                                goto END;
                        }

                        size = page_align(size);
                        k = kmalloc(GFP_KERNEL, size);
                        if (k == NULL) {
                                ret = -ENOMEM;
                                goto END;
                        }
                        // Apparently virt_to_phys might be deprecated soon? using __pa() for compiling on 5.11.15 kernel
                        // p = (void *)virt_to_phys(k);
                        p = (void *)__pa(k);

                        printk(KERN_INFO "virt_to_phys(): 0x%lx\n", (long unsigned int)p);

                        err = add_addr(k, p, size);
                        if (err) {
                                kfree(k);
                        }

                        if (copy_to_user((void *)ioctl_param, &p, sizeof(p))) {
                                pr_err("copy_to_user error!!\n");
                        }
                        break;

                case IOCTL_PA_FREE:

                        err = copy_from_user(&p, (void *)ioctl_param, sizeof(p));
                        if (err) {
                                ret = -EINVAL;
                                goto END;
                        }
                        printk(KERN_INFO "Freeing the physical address: 0x%lx\n", (long unsigned int)p);
                        err = find_and_delete(p);
                        if (err) {
                                printk(KERN_INFO "No such address!\n");
                                ret = -ENXIO; // no such address 
                                goto END;

                        }
                        break;

                default:
                        ret = -EINVAL;
                        break;
        }

END:
        mutex_unlock(&pa_mutex);
        return ret;
}

int device_mmap(struct file *filp, struct vm_area_struct *vma)
{
    unsigned long offset = vma->vm_pgoff;
    pr_info("offset : 0x%lx \n", offset);
    if (offset >= __pa(high_memory) || (filp->f_flags & O_SYNC))
        vma->vm_flags |= VM_IO;
    vma->vm_flags |= (VM_DONTEXPAND | VM_DONTDUMP);

    if (io_remap_pfn_range(vma, vma->vm_start, offset, 
        vma->vm_end-vma->vm_start, vma->vm_page_prot))
        return -EAGAIN;
    return 0;
}

static struct file_operations fops = {
    .open = pa_open,
    .unlocked_ioctl = pa_ioctl,
    .release = pa_release,
    .mmap = device_mmap,
};

int init_module(void)
{

    al = NULL;
    mutex_init(&pa_mutex);

    if (alloc_chrdev_region(&dev, 0, 1, DEVICE_NAME) < 0)
        goto fail;

    if ((cl = class_create(THIS_MODULE, DEVICE_NAME)) == NULL)
        goto class_create_fail;

    if (device_create(cl, NULL, dev, NULL, DEVICE_NAME) == NULL)
        goto device_create_fail;

    cdev_init(&c_dev, &fops);
    if (cdev_add(&c_dev, dev, 1) == -1)
        goto device_add_fail;

    printk(KERN_INFO "Loaded physmem device");
	return 0;

    device_add_fail:
        device_destroy(cl, dev);
    device_create_fail:
        class_destroy(cl);
    class_create_fail:
        unregister_chrdev_region(dev, 1);
    fail:
        return -1;
}

void cleanup_module(void)
{
    cdev_del(&c_dev);
    device_destroy(cl, dev);
    class_destroy(cl);
    unregister_chrdev_region(dev, 1);
    printk(KERN_INFO "Unloaded physmem device");
}

// physical memory allocation code provided by Ilja van Sprundel
MODULE_LICENSE("GPL");
MODULE_AUTHOR("https://github.com/droogie; https://github.com/iljavs;");
MODULE_DESCRIPTION("An unrestricted /dev/mem implementation that can also be used to allocate physical memory");
