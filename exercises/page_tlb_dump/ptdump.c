/*
 * Copyright (C) 2017 Canonical Group Ltd
 * Copyright (C) 2017 Hewlett Packard Enterprise Development, L.P.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 */

#include <linux/device.h>
#include <linux/fs.h>
#include <linux/kallsyms.h>
#include <linux/module.h>
#include <linux/uaccess.h>

#include <asm/pgtable.h>

#include "ptdump.h"

static int ptdump_major;
static struct class *ptdump_class;
static struct mm_struct *__init_mm;

#define PT_LEVEL_NONE 0
#define PT_LEVEL_PGD  1
#define PT_LEVEL_P4D  2
#define PT_LEVEL_PUD  3
#define PT_LEVEL_PMD  4
#define PT_LEVEL_PTE  5

/* -------------------------------------------------------------------------
   x86-64
   ------------------------------------------------------------------------- */

#ifdef CONFIG_X86_64

static const char * const PT_LEVEL_NAME[] = {
	"   ", "pgd", "p4d", "pud", "pmd", "pte"
};

static const char * const PT_LEVEL_SIZE[] = {
	"  ", "  ", "512G", "1G", "2M", "4K"
};

static void printk_prot(unsigned long val, int level)
{
	printk(KERN_CONT "| ");

	if (!val) {
		printk(KERN_CONT "                              ");
		goto out;
	}
	
	printk(KERN_CONT "%s ", (val & _PAGE_USER) ? "USR" : "   ");
	printk(KERN_CONT "%s ", (val & _PAGE_RW) ? "RW" : "ro");
	printk(KERN_CONT "%s ", (val & _PAGE_PWT) ? "PWT" : "   ");
	printk(KERN_CONT "%s ", (val & _PAGE_PCD) ? "PCD" : "   ");
	printk(KERN_CONT "%s ", (val & _PAGE_PSE && level <= 3) ?
	       "PSE" : "   ");
	printk(KERN_CONT "%s ", ((val & _PAGE_PAT_LARGE &&
				  (level == 2 || level == 3)) ||
				 (val & _PAGE_PAT && level == 4)) ?
	       "PAT" : "   ");
	printk(KERN_CONT "%s ", (val & _PAGE_GLOBAL) ? "GLB" : "   ");
	printk(KERN_CONT "%s ", (val & _PAGE_NX) ? "NX" : "x ");

out:
	printk(KERN_CONT "| %s %s\n", PT_LEVEL_NAME[level],
	       PT_LEVEL_SIZE[level]);
}

#endif

/* -------------------------------------------------------------------------
   arm64
   ------------------------------------------------------------------------- */

#ifdef CONFIG_ARM64

#define p4d_pfn(x)	(0)
#define p4d_large(x)	(0)
#define pud_large(x)	(pud_sect(x))
#define pmd_large(x)	(pmd_sect(x))

static const char * const PT_LEVEL_NAME[] = {
	"   ", "pgd", "p4d",
	CONFIG_PGTABLE_LEVELS > 3 ? "pud" : "pgd",
	CONFIG_PGTABLE_LEVELS > 2 ? "pmd" : "pgd",
	"pte"
};

#ifdef CONFIG_ARM64_4K_PAGES
#define _NONE_SIZE "  "
#define _PGD_SIZE  "  "
#define _P4D_SIZE  "  "
#define _PUD_SIZE  "1G"
#define _PMD_SIZE  "2M"
#define _PTE_SIZE  "4K"
#endif

#ifdef CONFIG_ARM64_16K_PAGES
#define _NONE_SIZE "   "
#define _PGD_SIZE  "   "
#define _P4D_SIZE  "   "
#define _PUD_SIZE  "   "
#define _PMD_SIZE  "32M"
#define _PTE_SIZE  "16K"
#endif

#ifdef CONFIG_ARM64_64K_PAGES
#define _NONE_SIZE "    "
#define _PGD_SIZE  "    "
#define _P4D_SIZE  "    "
#define _PUD_SIZE  "    "
#define _PMD_SIZE  "512M"
#define _PTE_SIZE  "64K "
#endif

static const char * const PT_LEVEL_SIZE[] = {
	_NONE_SIZE, _PGD_SIZE, _P4D_SIZE,
	CONFIG_PGTABLE_LEVELS > 3 ? _PUD_SIZE : _PGD_SIZE,
	CONFIG_PGTABLE_LEVELS > 2 ? _PMD_SIZE : _PGD_SIZE,
	_PTE_SIZE
};

static void printk_prot(unsigned long val, int level)
{
	printk(KERN_CONT "| ");

	if (!val) {
		printk(KERN_CONT "                                          ");
		goto out;
	}

	printk(KERN_CONT "%s ", (val & PTE_TABLE_BIT) ? "   " : "blk");
	printk(KERN_CONT "%s ", (val & PTE_USER) ? "USR" : "   ");
	printk(KERN_CONT "%s ", (val & PTE_RDONLY) ? "RO" : "rw");
	printk(KERN_CONT "%s ", (val & PTE_SHARED) ? "SHD" : "   ");
	printk(KERN_CONT "%s ", (val & PTE_AF) ? "AF" : "  ");
	printk(KERN_CONT "%s ", (val & PTE_NG) ? "NG" : "  ");
	printk(KERN_CONT "%s ", (val & PTE_DBM) ? "DBM" : "   ");
	printk(KERN_CONT "%s ", (val & PTE_CONT) ? "CONT" : "    ");
	printk(KERN_CONT "%s ", (val & PTE_PXN) ? "NX" : "x ");
	printk(KERN_CONT "%s ", (val & PTE_UXN) ? "UXN" : "   ");

	switch (val & PTE_ATTRINDX_MASK) {
	case PTE_ATTRINDX(MT_DEVICE_nGnRnE):
		printk(KERN_CONT "DEVICE/nGnRnE ");
		break;
	case PTE_ATTRINDX(MT_DEVICE_nGnRE):
		printk(KERN_CONT "DEVICE/nGnRE  ");
		break;
	case PTE_ATTRINDX(MT_DEVICE_GRE):
		printk(KERN_CONT "DEVICE/GRE    ");
		break;
	case PTE_ATTRINDX(MT_NORMAL_NC):
		printk(KERN_CONT "MEM/NORMAL-NC ");
		break;
	case PTE_ATTRINDX(MT_NORMAL):
		printk(KERN_CONT "MEM/NORMAL    ");
		break;
	default:
		printk(KERN_CONT "              ");
		break;
	}

out:
	printk(KERN_CONT "| %s %s\n", PT_LEVEL_NAME[level],
	       PT_LEVEL_SIZE[level]);
}

#endif

/* -------------------------------------------------------------------------
   generic
   ------------------------------------------------------------------------- */

static int bad_address(void *p)
{
	unsigned long dummy;

	return probe_kernel_address((unsigned long *)p, dummy);
}

static void printk_pagetable(unsigned long addr)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud = NULL;
	pmd_t *pmd = NULL;
	pte_t *pte = NULL;
	unsigned long phys_addr, offset;
	struct page *page = virt_to_page(addr);

	printk("  ------------------------------\n");
	printk("  virtual %s addr: %016lx\n", addr > PAGE_OFFSET ? "kernel" :
	       "user", addr);
	printk("  page: %016lx\n", (unsigned long)page);

	if (addr > PAGE_OFFSET) {
		/* kernel virtual address */
		pgd = pgd_offset(__init_mm, addr);
	} else {
		/* user (process) virtual address */
		pgd = pgd_offset(current->mm, addr);
	}
	printk("  pgd: %016lx (%016lx) ", (unsigned long)pgd,
	       (unsigned long)pgd_val(*pgd));
	printk_prot(pgd_val(*pgd), PT_LEVEL_PGD);

	p4d = p4d_offset(pgd, addr);
	printk("  p4d: %016lx (%016lx) ", (unsigned long)p4d,
	       (unsigned long)p4d_val(*p4d));
	printk_prot(p4d_val(*p4d), PT_LEVEL_P4D);
	if (p4d_large(*p4d) || !p4d_present(*p4d)) {
		phys_addr = (unsigned long)p4d_pfn(*p4d) << PAGE_SHIFT;
		offset = addr & ~P4D_MASK;
		goto out;
	}

	pud = pud_offset(p4d, addr);
	printk("  pud: %016lx (%016lx) ", (unsigned long)pud,
	       (unsigned long)pud_val(*pud));
	printk_prot(pud_val(*pud), PT_LEVEL_PUD);
	if (pud_large(*pud) || !pud_present(*pud)) {
		phys_addr = (unsigned long)pud_pfn(*pud) << PAGE_SHIFT;
		offset = addr & ~PUD_MASK;
		goto out;
	}

	pmd = pmd_offset(pud, addr);
	printk("  pmd: %016lx (%016lx) ", (unsigned long)pmd,
	       (unsigned long)pmd_val(*pmd));
	printk_prot(pmd_val(*pmd), PT_LEVEL_PMD);
	if (pmd_large(*pmd) || !pmd_present(*pmd)) {
		phys_addr = (unsigned long)pmd_pfn(*pmd) << PAGE_SHIFT;
		offset = addr & ~PMD_MASK;
		goto out;
	}

	pte =  pte_offset_kernel(pmd, addr);
	printk("  pte: %016lx (%016lx) ", (unsigned long)pte,
	       (unsigned long)pte_val(*pte));
	printk_prot(pte_val(*pte), PT_LEVEL_PTE);
	phys_addr = (unsigned long)pte_pfn(*pte) << PAGE_SHIFT;
	offset = addr & ~PAGE_MASK;

out:
	printk("  p4d_page: %016lx\n", (unsigned long)p4d_page(*p4d));
	if (pud)
		printk("  pud_page: %016lx\n", (unsigned long)pud_page(*pud));
	if (pmd)
		printk("  pmd_page: %016lx\n", (unsigned long)pmd_page(*pmd));
	if (pte)
		printk("  pte_page: %016lx\n", (unsigned long)pte_page(*pte));
	printk("  physical addr: %016lx\n", phys_addr | offset);
	printk("  page addr: %016lx\n", phys_addr);
	printk("  ------------------------------\n");
}

static pte_t *__lookup_addr(unsigned long addr, unsigned int *level)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud;
	pmd_t *pmd;

	if (addr > PAGE_OFFSET) {
		/* kernel virtual address */
		pgd = pgd_offset(__init_mm, addr);
	} else {
		/* user (process) virtual address */
		pgd = pgd_offset(current->mm, addr);
	}

	*level = PT_LEVEL_NONE;

	if (pgd_none(*pgd))
		return NULL;

	p4d = p4d_offset(pgd, addr);
	if (p4d_none(*p4d))
		return NULL;

	*level = PT_LEVEL_P4D;
	if (p4d_large(*p4d) || !p4d_present(*p4d))
		return (pte_t *)p4d;

	pud = pud_offset(p4d, addr);
	if (pud_none(*pud))
		return NULL;

	*level = PT_LEVEL_PUD;
	if (pud_large(*pud) || !pud_present(*pud))
		return (pte_t *)pud;

	pmd = pmd_offset(pud, addr);
	if (pmd_none(*pmd))
		return NULL;

	*level = PT_LEVEL_PMD;
	if (pmd_large(*pmd) || !pmd_present(*pmd))
		return (pte_t *)pmd;

	*level = PT_LEVEL_PTE;

	return pte_offset_kernel(pmd, addr);
}

/*
 * Convert a virtual (process or kernel) address to a physical address
 *
 * Based on slow_virt_to_phys() from arch/x86/mm/pageattr.c
 */
static unsigned long any_virt_to_phys(unsigned long addr)
{
	unsigned long phys_addr;
	unsigned long offset;
	unsigned int level;
	pte_t *pte;

	pte = __lookup_addr(addr, &level);
	if (!pte)
		return 0;

	/*
	 * pXX_pfn() returns unsigned long, which must be cast to phys_addr_t
	 * before being left-shifted PAGE_SHIFT bits -- this trick is to
	 * make 32-PAE kernel work correctly.
	 */
	switch (level) {
	case PT_LEVEL_P4D:
		phys_addr = (unsigned long)p4d_pfn(*(p4d_t *)pte) << PAGE_SHIFT;
		offset = addr & ~P4D_MASK;
		break;
	case PT_LEVEL_PUD:
		phys_addr = (unsigned long)pud_pfn(*(pud_t *)pte) << PAGE_SHIFT;
		offset = addr & ~PUD_MASK;
		break;
	case PT_LEVEL_PMD:
		phys_addr = (unsigned long)pmd_pfn(*(pmd_t *)pte) << PAGE_SHIFT;
		offset = addr & ~PMD_MASK;
		break;
	default:
		phys_addr = (unsigned long)pte_pfn(*pte) << PAGE_SHIFT;
		offset = addr & ~PAGE_MASK;
	}

	return (phys_addr | offset);
}

/*
 * Convert a physical address to a kernel virtual address
 */
static unsigned long phys_to_kern(unsigned long phys_addr)
{
	return (unsigned long)phys_to_virt(phys_addr);
}

static int ptdump_open(struct inode *i, struct file *f)
{
	return 0;
}

static int ptdump_release(struct inode *i, struct file *f)
{
	return 0;
}

static long ptdump_ioctl(struct file *file, unsigned int cmd,
			 unsigned long arg)
{
	struct ptdump_req __user karg, *req = &karg;
	unsigned long phys_addr, kern_addr, paddr;
	unsigned long buf;

	printk("--------------------------------------------------------------"
	       "-----------------\n");

	printk("pid: %d, comm: %s\n", current->pid, current->comm);

	if (copy_from_user(&karg, (void *)arg, sizeof(struct ptdump_req))) {
		printk("failed to copy_from_user()\n");
		return -EFAULT;
	}

	printk("user addr: %016lx, order: %d\n", req->addr, req->order);
//	printk("user data: %s\n", (char *)req->addr);
	printk_pagetable(req->addr);

	switch (cmd) {

	case PTDUMP_DUMP:
		printk("ioctl cmd: PTDUMP_DUMP\n");

		phys_addr = any_virt_to_phys(req->addr);
		kern_addr = phys_to_kern(phys_addr);
		printk("kernel addr: %016lx\n", kern_addr);
		if (bad_address((void *)kern_addr))
			printk("kernel data: *** BAD ADDRESS ***\n");
		else
			printk("kernel data: %s\n", (char *)kern_addr);
		printk_pagetable(kern_addr);

		/* Validate our address translation */
		paddr = virt_to_phys((void *)kern_addr);
		if (paddr != phys_addr)
			printk("+++ Incorrect address translation +++\n");

		break;

	case PTDUMP_WRITE:
		printk("ioctl cmd: PTDUMP_WRITE\n");

		buf = __get_free_pages(GFP_KERNEL, req->order);
		if (!buf) {
			printk("failed to __get_free_pages()\n");
			return -ENOMEM;
		}
		printk("buf addr: %016lx\n", buf);
		if (copy_from_user((char *)buf, (char *)req->addr,
				   PAGE_SIZE * (1 << req->order))) {
			printk("failed to copy_from_user()\n");
			return -EFAULT;
		}
		printk("buf data: %s\n", (char *)buf);
		printk_pagetable(buf);
		free_pages(buf, req->order);

		break;

	default:
		printk("ioctl cmd: UNKNOWN\n");
		break;
	}

	return 0;
}

static struct file_operations ptdump_fops = {
	.open = ptdump_open,
	.release = ptdump_release,
	.unlocked_ioctl = ptdump_ioctl,
};

static int __init ptdump_init(void)
{
	int ret;

	printk("=============================================================="
	       "=================\n");

	__init_mm = (struct mm_struct *)kallsyms_lookup_name("init_mm");
	if (!__init_mm) {
		printk("failed to lookup 'init_mm'\n");
		ret = -ENXIO;
		goto out;
	}
	printk("init_mm: %p\n", __init_mm);

	ptdump_major = register_chrdev(0, "ptdump", &ptdump_fops);
	if (ptdump_major < 0) {
		printk("failed to register device\n");
		ret = ptdump_major;
		goto out;
	}

	ptdump_class = class_create(THIS_MODULE, "ptdump");
	if (IS_ERR(ptdump_class)) {
		printk("failed to create class\n");
		ret = PTR_ERR(ptdump_class);
		goto out_unregister;
	}

	device_create(ptdump_class, NULL, MKDEV(ptdump_major, 0), NULL,
		      "ptdump");

	printk("ptdump module loaded\n");
	return 0;

out_unregister:
	unregister_chrdev(ptdump_major, "ptdump");
out:
	return ret;

}

static void __exit ptdump_exit(void)
{
	device_destroy(ptdump_class, MKDEV(ptdump_major, 0));
	class_destroy(ptdump_class);
	unregister_chrdev(ptdump_major, "ptdump");

	printk("ptdump module unloaded\n");
	printk("=============================================================="
	       "=================\n");
}

module_init(ptdump_init);
module_exit(ptdump_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Juerg Haefliger <juerg.haefliger@hpe.com>");
