//////////////////////////////////////////////////////////////////////
//                             PMC-Sierra, Inc.
//
//
//
//                             Copyright 2015
//
////////////////////////////////////////////////////////////////////////
//
// This program is free software; you can redistribute it and/or modify it
// under the terms and conditions of the GNU General Public License,
// version 2, as published by the Free Software Foundation.
//
// This program is distributed in the hope it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
//
////////////////////////////////////////////////////////////////////////
//
//   Author:  Stephen Bates
//
//   Description:
//     Loadable Kernel Module for looking at system memory.
//
////////////////////////////////////////////////////////////////////////

#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/mm.h>
#include <linux/sched.h>
#include <linux/mmzone.h>
#include <linux/debugfs.h>
#include <linux/slab.h>
#include <linux/miscdevice.h>
#include <asm/pgtable.h>

#define MODULENAME "mem_map"

static struct dentry *debugfs;

static void prettyprint_struct_page(unsigned long pfn, struct page *page)
{
    printk(KERN_INFO "Hello, this is prettyprint_struct_page() for pfn 0x%lx.\n",
        pfn);
    printk(KERN_INFO "Looks like pfn 0x%lx resides in memory zone %u.\n",
	   pfn, pfn_to_nid(pfn));

    printk(KERN_INFO "page->flags     = %lx.\n", page->flags);
    printk(KERN_INFO "page->_mapcount = %d.\n", atomic_read(&page->_mapcount));
    printk(KERN_INFO "page->_refcount    = %d.\n", atomic_read(&page->_refcount));
}

static int write_pfn(void *data, u64 pfn)
{
    struct page *page;

    printk(KERN_INFO "Hello, this is write_pfn() for pfn 0x%lx.\n",
           (unsigned long)pfn);
    if (!pfn_valid(pfn)) {
        printk(KERN_INFO "PFN is invalid!\n");
    } else {
        printk(KERN_INFO "PFN is valid!\n");

        page = pfn_to_page((unsigned long)pfn);
        prettyprint_struct_page(pfn, page);
    }

    return 0;
}

static int write_free(void *data, u64 pfn)
{
    struct page *page;
    unsigned long i, start, end, free = 0, total = 0;

    printk(KERN_INFO "Hello, this is write_free() for pfn 0x%lx.\n",
           (unsigned long)pfn);
    if ( !pfn_valid((unsigned long)pfn) ){
	    printk(KERN_INFO "Looks like pfn 0x%lx is not valid.\n",
		   (unsigned long)pfn);
	    return 0;
    }

    start = node_start_pfn(pfn_to_nid(pfn));
    end   = node_end_pfn(pfn_to_nid(pfn));

    for (i=start; i<end ;i++){
	    total++;
	    page = pfn_to_page((unsigned long)i);
	    if ( page->flags & 0x1 )
		    free++;
    }
    printk(KERN_INFO "Looks like %lu pages of %lu are free.\n",
           free, total);

    return 0;
}

DEFINE_SIMPLE_ATTRIBUTE(pfn_fops, NULL, write_pfn, "%llu\n");
DEFINE_SIMPLE_ATTRIBUTE(free_fops, NULL, write_free, "%llu\n");

static void print_zones(pg_data_t *pgdat)
{
    struct zone *zone;
    struct zone *node_zones = pgdat->node_zones;
    unsigned long flags;

    for (zone = node_zones; zone - node_zones < MAX_NR_ZONES; ++zone) {
        spin_lock_irqsave(&zone->lock, flags);
        printk(KERN_INFO "Zone %s - %d\n", zone->name, populated_zone(zone));
        printk(KERN_INFO "  %lx  %ld %ld %ld\n", zone->zone_start_pfn,
               zone->managed_pages, zone->spanned_pages, zone->present_pages);
        spin_unlock_irqrestore(&zone->lock, flags);
    }
}

static long mm_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct page **pages;
    struct zone *zone;
    long ret;
    struct vm_area_struct *vma = NULL;

    pages = (struct page **) __get_free_page(GFP_KERNEL);
    if (!pages)
        return -ENOMEM;

    vma = find_vma(current->mm, arg);
    if (!vma)
        return -EIO;

    printk(KERN_INFO "Start: %lx - %ld\n", arg, vma->vm_flags);
//    ret = get_user_pages_remote(current, current->mm,
//				arg, 1, 0, 1, pages, NULL);
//
    if (ret < 1) {
        printk(KERN_INFO "GUP error: %ld\n", ret);
        free_page((unsigned long) pages);
        return -EFAULT;
    }

    ret = page_to_pfn(pages[0]);
    zone = page_zone(pages[0]);

    printk(KERN_INFO "Userspace PFN: %08lx (ZONE: %s)\n", ret,
           zone->name);

    put_page(pages[0]);
    free_page((unsigned long) pages);

    if (cmd == 0)
        return ret << PAGE_SHIFT;
    else
        return strcmp(zone->name,"Device") ? 0 : 1;
}

static const struct file_operations fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = mm_ioctl,
    .compat_ioctl = mm_ioctl,
};

static struct miscdevice miscdev = {
        .minor = MISC_DYNAMIC_MINOR,
        .name = MODULENAME,
        .fops = &fops,
};

static int __init init_mem_map(void)
{
    int nid, nodes = 0;
    int error;

    error = misc_register(&miscdev);
    if (error) {
        pr_err("can't misc_register :(\n");
        return error;
    }

    debugfs = debugfs_create_dir(MODULENAME, NULL);
    if (debugfs == NULL )
        printk(KERN_INFO "Unable to create debugfs directory.");
    if (debugfs_create_file("pfn", 0222, debugfs, NULL, &pfn_fops) == NULL)
        printk(KERN_INFO "Unable to create debugfs file pfn.");
    if (debugfs_create_file("free", 0222, debugfs, NULL, &free_fops) == NULL)
        printk(KERN_INFO "Unable to create debugfs file free.");

    printk(KERN_INFO "\n\n********************************************\n");
    printk(KERN_INFO "Hello, this is init_mem_map().\n");
    printk(KERN_INFO "You have %lu pages to play with!\n",
           get_num_physpages());

    for_each_online_node(nid){
            printk(KERN_INFO "node %d info ****************************************\n", nid);
	    nodes++;
	    printk(KERN_INFO "node_data[%d]->node_start_pfn = %lu.\n",
		   nid, node_data[nid]->node_start_pfn);
	    printk(KERN_INFO "node_data[%d]->node_present_pages = %lu.\n",
		   nid, node_data[nid]->node_present_pages);
	    printk(KERN_INFO "node_data[%d]->node_spanned_pages = %lu.\n",
		   nid, node_data[nid]->node_spanned_pages);

        print_zones(node_data[nid]);
    }
    printk(KERN_INFO "You have %d node(s) in your system!\n",
           nodes);

    return 0;
}

static void __exit exit_mem_map(void)
{
    printk(KERN_INFO "Goodbye, this is exit_mem_map().\n");
    debugfs_remove_recursive(debugfs);
    misc_deregister(&miscdev);
}

module_init(init_mem_map);
module_exit(exit_mem_map);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Stephen Bates <stephen.bates@pmcs.com>");
MODULE_DESCRIPTION("Displays information on system memory.");
