
# PAGESIZE 

```
[root@centos7 alloc_page]# getconf -a | grep -i page
PAGESIZE                           65536
PAGE_SIZE                          65536
_AVPHYS_PAGES                      8180479
_PHYS_PAGES                        8365864
[root@centos7 alloc_page]# 
```

# PAGE_MASK
```
[root@centos7 alloc_page]# lsmod | tail test3
tail: cannot open ‘test3’ for reading: No such file or directory
[root@centos7 alloc_page]# lsmod | grep  test3
alloc_page_test3      262144  0 
[root@centos7 alloc_page]# dmesg | tail -n 10
[20601.864804] PTRS_PER_P4D = 1
[20601.867675] PTRS_PER_PUD = 1
[20601.870543] PTRS_PER_PMD = 8192
[20601.873669] PTRS_PER_PTE = 8192
[20601.876797] PAGE_MASK = 0xffffffffffff0000
[20601.880878] vaddr to phy addr entry!
[20601.884440] __get_free_page, alloc the free page vaddr=0xffffa05fc7e70000
[20601.891201] pgd_val=0x0, pdg_index=0x28
[20601.895020] pud_val=0x0
[20601.897464] not mapped in pud
[root@centos7 alloc_page]# 
```


#  CONFIG_PGTABLE_LEVELS

```
[root@centos7 boot]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 boot]# 
[root@centos7 boot]# grep CONFIG_PGTABLE_LEVELS  config-4.14.0-115.el7a.0.1.aarch64
CONFIG_PGTABLE_LEVELS=3
[root@centos7 boot]#
[root@centos7 boot]# grep CONFIG_ARM64_VA_BITS  config-4.14.0-115.el7a.0.1.aarch64
# CONFIG_ARM64_VA_BITS_42 is not set
CONFIG_ARM64_VA_BITS_48=y
CONFIG_ARM64_VA_BITS=48
[root@centos7 boot]# 
```

```
root@ubuntux86:/home/ubuntu# uname -a
Linux ubuntux86 5.15.0-41-generic #44~20.04.1-Ubuntu SMP Fri Jun 24 13:27:29 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:/home/ubuntu#
root@ubuntux86:/boot# grep CONFIG_PGTABLE_LEVELS  config-5.13.0-39-generic
CONFIG_PGTABLE_LEVELS=5
root@ubuntux86:/boot# 
```

代码路径：
arch/arm64/include/asm/pgtable-types.h：定义pgd_t, pud_t, pmd_t, pte_t等类型；
arch/arm64/include/asm/pgtable-prot.h：针对页表中entry中的权限内容设置；
arch/arm64/include/asm/pgtable-hwdef.h：主要包括虚拟地址中PGD/PMD/PUD等的划分，这个与虚拟地址的有效位及分页大小有关，此外还包括硬件页表的定义， TCR寄存器中的设置等；
arch/arm64/include/asm/pgtable.h：页表设置相关；

在这些代码中可以看到，

当CONFIG_PGTABLE_LEVELS=4时：pgd-->pud-->pmd-->pte;
当CONFIG_PGTABLE_LEVELS=3时，没有PUD页表：pgd(pud)-->pmd-->pte;当没有pud的时候，pud等于p4d，则pgd=p4d=pud
当CONFIG_PGTABLE_LEVELS=2时，没有PUD和PMD页表：pgd(pud, pmd)-->pte

# arch/arm64/mm/fault.c
```
static void show_pte(unsigned long addr)
{
	struct mm_struct *mm;
	pgd_t *pgdp;
	pgd_t pgd;

	if (is_ttbr0_addr(addr)) {
		/* TTBR0 */
		mm = current->active_mm;
		if (mm == &init_mm) {
			pr_alert("[%016lx] user address but active_mm is swapper\n",
				 addr);
			return;
		}
	} else if (is_ttbr1_addr(addr)) {
		/* TTBR1 */
		mm = &init_mm;
	} else {
		pr_alert("[%016lx] address between user and kernel address ranges\n",
			 addr);
		return;
	}

	pr_alert("%s pgtable: %luk pages, %llu-bit VAs, pgdp=%016lx\n",
		 mm == &init_mm ? "swapper" : "user", PAGE_SIZE / SZ_1K,
		 vabits_actual, mm_to_pgd_phys(mm));
	pgdp = pgd_offset(mm, addr);
	pgd = READ_ONCE(*pgdp);
	pr_alert("[%016lx] pgd=%016llx", addr, pgd_val(pgd));

	do {
		p4d_t *p4dp, p4d;
		pud_t *pudp, pud;
		pmd_t *pmdp, pmd;
		pte_t *ptep, pte;

		if (pgd_none(pgd) || pgd_bad(pgd))
			break;

		p4dp = p4d_offset(pgdp, addr);
		p4d = READ_ONCE(*p4dp);
		pr_cont(", p4d=%016llx", p4d_val(p4d));
		if (p4d_none(p4d) || p4d_bad(p4d))
			break;

		pudp = pud_offset(p4dp, addr);
		pud = READ_ONCE(*pudp);
		pr_cont(", pud=%016llx", pud_val(pud));
		if (pud_none(pud) || pud_bad(pud))
			break;

		pmdp = pmd_offset(pudp, addr);
		pmd = READ_ONCE(*pmdp);
		pr_cont(", pmd=%016llx", pmd_val(pmd));
		if (pmd_none(pmd) || pmd_bad(pmd))
			break;

		ptep = pte_offset_map(pmdp, addr);
		pte = READ_ONCE(*ptep);
		pr_cont(", pte=%016llx", pte_val(pte));
		pte_unmap(ptep);
	} while(0);

	pr_cont("\n");
}
```

#  insmod  alloc_page_test3.ko 

```
[18493.851240] free the alloc page and leave the v2p!
[19423.276770] ****************print page relate macro:
[19423.281716] pgdir_SHIFT = 42
[19423.284583]  = 0xffff800000000000
[19423.288843] P4D_SHIFT = 42
[19423.291538] PUD_SHIFT = 42
[19423.294233] PMD_SHIFT = 29
[19423.296932] PAGE_SHIFT = 16
[19423.299713] PTRS_PER_PGD = 64
[19423.302666] PTRS_PER_P4D = 1
[19423.305533] PTRS_PER_PUD = 1
[19423.308405] PTRS_PER_PMD = 8192
[19423.311531] PTRS_PER_PTE = 8192
[19423.314657] PAGE_MASK = 0xffffffffffff0000
```

***PAGE_OFFSET = 0xffff800000000000***

```
[root@centos7 alloc_page]# dmesg | tail -n 30
[17550.953218] __get_free_page, alloc the free page vaddr=0xffffa05fd9610000
[17550.959978] ************** call vaddr2paddr_1 
[17550.964400] pgd_val=0x0, pdg_index=0x28
[17550.968222] p4d_val=0x0
[17550.970659] pud_val=0x0
[17550.973095] not mapped in pud
[17550.976049] ************** call vaddr2paddr 
[17550.980304] kernel virtual address 
[17550.990242] pgd_val=0x205ffff80003, pdg_index=0x28
[17550.995010] p4d_val=0x205ffff80003 
[17550.998486] pud_val=0x205ffff80003
[17551.001874] pmd_val=0xf8205fc0000f11, pmd_index=0x2fe
[17551.006901] pte_val=0x0, pte_index=0x1961
[17551.010898] not mapped in pte
[17551.013852] ************** call printk_pagetable
[17551.018451]   ------------------------------
[17551.022702]   virtual kernel addr: ffffa05fd9610000
[17551.027557]   page: ffff7fe817f65840
[17551.031120] kernel virtual address 
[17551.040917]   pgd: ffff000009910140 (0000205ffff80003) 
[17551.040918]   p4d: ffff000009910140 (0000205ffff80003) 
[17551.046119]   pud: ffff000009910140 (0000205ffff80003) 
[17551.051325]   pmd: ffffa05ffff817f0 (00f8205fc0000f11) 
[17551.056527] pmd_large(*pmd): 1, pmd_present(*pmd) : 1 
[17551.066845]   p4d_page: ffff7fe817fffe00
[17551.070756]   pud_page: ffff7fe817fffe00
[17551.074661]   pmd_page: ffff7fe817f00000
[17551.078570]   physical addr: 0000205fd9610000
[17551.082908]   page addr: 0000205fc0000000
[17551.086899]   ------------------------------
[root@centos7 alloc_page]# 
```
***not mapped in pud***
***not mapped in pte***

# compare  vaddr2paddr_1 and vaddr2paddr

 ***slove problem : not mapped in pud***
 
```
    if (vaddr > PAGE_OFFSET) {
                /* kernel virtual address */
        pr_info("kernel virtual address \n");
        __init_mm = (struct mm_struct *)kallsyms_lookup_name("init_mm");
        pgd = pgd_offset(__init_mm, vaddr);
   } else {
                /* user (process) virtual address */
        pr_info("user virtual address \n");
        pgd = pgd_offset(current->mm, vaddr);
    }
    printk("pgd_val=0x%lx, pdg_index=0x%lx\n", pgd_val(*pgd), pgd_index(vaddr));
    if (pgd_none(*pgd)) {
        printk("not mapped in pgd\n");
        return -1;
    }
```
# compare  vaddr2paddr  and printk_pagetable

 ***slove problem : not mapped in pte***

```
        pmd = pmd_offset(pud, addr);
        printk("  pmd: %016lx (%016lx) ", (unsigned long)pmd,
               (unsigned long)pmd_val(*pmd));
        //printk_prot(pmd_val(*pmd), PT_LEVEL_PMD);
        if (pmd_large(*pmd) || !pmd_present(*pmd)) {
                pr_info("pmd_large(*pmd): %d, pmd_present(*pmd) : %d \n", pmd_large(*pmd), pmd_present(*pmd));
                phys_addr = (unsigned long)pmd_pfn(*pmd) << PAGE_SHIFT;
                offset = addr & ~PMD_MASK;
                goto out;
        }
```

# PAGE_OFFSET

PHYS_OFFSET: RAM第一个BANK的物理地址地址。
PAGE_OFFSET: RAM第一个BANK的虚拟地址地址。

```
//mm/dump.c
static const struct addr_marker address_markers[] = {
#ifdef CONFIG_KASAN
        { KASAN_SHADOW_START,           "Kasan shadow start" },
        { KASAN_SHADOW_END,             "Kasan shadow end" },
#endif
        { MODULES_VADDR,                "Modules start" },
        { MODULES_END,                  "Modules end" },
        { VMALLOC_START,                "vmalloc() Area" },
        { VMALLOC_END,                  "vmalloc() End" },
        { FIXADDR_START,                "Fixmap start" },
        { FIXADDR_TOP,                  "Fixmap end" },
        { PCI_IO_START,                 "PCI I/O start" },
        { PCI_IO_END,                   "PCI I/O end" },
#ifdef CONFIG_SPARSEMEM_VMEMMAP
        { VMEMMAP_START,                "vmemmap start" },
        { VMEMMAP_START + VMEMMAP_SIZE, "vmemmap end" },
#endif
        { PAGE_OFFSET,                  "Linear Mapping" },
        { -1,                           NULL },
};

[root@centos7 boot]#  dmesg | grep -i "Virtual kernel memory" -A 20
[    0.000000] Virtual kernel memory layout:
[    0.000000]     modules : 0xffff000000000000 - 0xffff000008000000   (   128 MB)
[    0.000000]     vmalloc : 0xffff000008000000 - 0xffff7bdfffff0000   (126847 GB)
[    0.000000]       .text : 0xffff000008080000 - 0xffff0000088a0000   (  8320 KB)
[    0.000000]     .rodata : 0xffff0000088a0000 - 0xffff000008c00000   (  3456 KB)
[    0.000000]       .init : 0xffff000008c00000 - 0xffff000008d70000   (  1472 KB)
[    0.000000]       .data : 0xffff000008d70000 - 0xffff000008f47a00   (  1887 KB)
[    0.000000]        .bss : 0xffff000008f47a00 - 0xffff0000098d5d18   (  9785 KB)
[    0.000000]     fixed   : 0xffff7fdffe790000 - 0xffff7fdffec00000   (  4544 KB)
[    0.000000]     PCI I/O : 0xffff7fdffee00000 - 0xffff7fdfffe00000   (    16 MB)
[    0.000000]     vmemmap : 0xffff7fe000000000 - 0xffff800000000000   (   128 GB maximum)
[    0.000000]               0xffff7fe000000000 - 0xffff7fe818000000   ( 33152 MB actual)
[    0.000000]     memory  : 0xffff800000000000 - 0xffffa06000000000   (33947648 MB)
```

```
//include/asm/memory.h
/*
 * VMEMMAP_SIZE - allows the whole linear region to be covered by
 *                a struct page array
 */
#define VMEMMAP_SIZE (UL(1) << (VA_BITS - PAGE_SHIFT - 1 + STRUCT_PAGE_MAX_SHIFT))

/*
 * PAGE_OFFSET - the virtual address of the start of the linear map (top
 *               (VA_BITS - 1))
 * KIMAGE_VADDR - the virtual address of the start of the kernel image
 * VA_BITS - the maximum number of bits for virtual addresses.
 * VA_START - the first kernel virtual address.
 */
#define VA_BITS                 (CONFIG_ARM64_VA_BITS)
#define VA_START                (UL(0xffffffffffffffff) - \
        (UL(1) << VA_BITS) + 1)
#define PAGE_OFFSET             (UL(0xffffffffffffffff) - \
        (UL(1) << (VA_BITS - 1)) + 1)
#define KIMAGE_VADDR            (MODULES_END)
#define MODULES_END             (MODULES_VADDR + MODULES_VSIZE)
#define MODULES_VADDR           (VA_START + KASAN_SHADOW_SIZE)
#define MODULES_VSIZE           (SZ_128M)
#define VMEMMAP_START           (PAGE_OFFSET - VMEMMAP_SIZE)
#define PCI_IO_END              (VMEMMAP_START - SZ_2M)
#define PCI_IO_START            (PCI_IO_END - PCI_IO_SIZE)
#define FIXADDR_TOP             (PCI_IO_START - SZ_2M)

#define KERNEL_START      _text
#define KERNEL_END        _end
```
PAGE_OFFSET其实就是物理地址与线性地址之间的位移量。Linux的虚拟地址空间也为0～4G。
Linux内核将这4G字节的空间分为两部分。将最高的1G字节供内核使用，称为“内核空间”。
而将较低的3G字节，供各个进程使用，称为“用户空间）。因为每个进程可以通过系统调用进入内核，
因此，Linux内核由系统内的所有进程共享。于是，从具体进程的角度来看，每个进程可以拥有4G字节的虚拟空间。
在嵌入式系统中，PAGE_OFFSET也是可配置的，比如修改为CONFIG_PAGE_OFFSET=0x80000000，
那么在压缩内核的工具中就需要做相应的修改，比如是vmlinux，才有mkimage工具，
```
./mkimage -A ARM-O linux -T kernel -C gzip -a 0x80800000 -e 0x80801000 -n "Linux 2.6" -d vmlinux.bin.gz vmlinux.ub。
```
PAGE_OFFSET后便宜8M是留在其他用途，根据具体芯片设计的要求来改就可以了。

# pmd_XX

```

            if (pmd_none(*pmd)) {
                printk("   pmd = empty\n");
                return;
            }
            if (pmd_huge(*pmd) && vma->vm_flags & VM_HUGETLB) {
                entry = (pte_t*)pmd_val(*pmd);
                printk("   pmd = huge\n");
                return;
            }
            if (pmd_trans_huge(*pmd)) {
                entry = (pte_t*)pmd_val(*pmd);
                printk("   pmd = trans_huge\n");
                return;
            }
            if (!pmd_bad(*pmd)) {
                pte_t * pte = pte_offset_map(pmd, address);
```

# references


[hello-7.c](https://github.com/martinmullins/CVE-2016-8655_Android/blob/8d92eca69317f07bb0d532b60e0508d9cee30698/rabit_hole/mod/hello-7.c)

[内存管理源码分析-内核页表的创建以及索引方式(基于ARM64以及4级页表)](https://blog.csdn.net/u011649400/article/details/105984564?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-105984564-blog-105807230.pc_relevant_multi_platform_whitelistv3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1-105984564-blog-105807230.pc_relevant_multi_platform_whitelistv3&utm_relevant_index=2)