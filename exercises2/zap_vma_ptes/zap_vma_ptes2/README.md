



```
[root@centos7 handle_mm_fault]# dmesg | tail -n 50
[710963.878650] sample_open
[710963.881285] sample_write
[710963.883896] no page for vma addr  849805312 
[710963.888256]   ------------------------------
[710963.892598]   virtual user addr: 0000000032a70000
[710963.897369]   pgd: ffffa05fd30a4400 (0000005f37420003) 
[710963.897372]   p4d: ffffa05fd30a4400 (0000005f37420003) 
[710963.902672]   pud: ffffa05fd30a4400 (0000005f37420003) 
[710963.907971]   pmd: ffff805f37420008 (0000005fc7230003) 
[710963.913262]   pte: ffff805fc7239538 (00e8005690060f53) 
[710963.918566]   p4d_page: ffff7fe017cdd080
[710963.927859]   pud_page: ffff7fe017cdd080
[710963.931853]   pmd_page: ffff7fe017f1c8c0
[710963.935845]   pte_page: ffff7fe015a40180
[710963.939849]   physical addr: 0000005690060000
[710963.944274]   page addr: 0000005690060000
[710963.948361]   ------------------------------
[710963.952760] #########  after get_user_pages_remote ############# 
[710963.958930]   ------------------------------
[710963.963271]   virtual user addr: 0000000032a70000
[710963.968053]   pgd: ffffa05fd30a4400 (0000005f37420003) 
[710963.968055]   p4d: ffffa05fd30a4400 (0000005f37420003) 
[710963.973345]   pud: ffffa05fd30a4400 (0000005f37420003) 
[710963.978647]   pmd: ffff805f37420008 (0000005fc7230003) 
[710963.983937]   pte: ffff805fc7239538 (00e8005690060f53) 
[710963.989235]   p4d_page: ffff7fe017cdd080
[710963.998530]   pud_page: ffff7fe017cdd080
[710964.002524]   pmd_page: ffff7fe017f1c8c0
[710964.006518]   pte_page: ffff7fe015a40180
[710964.010523]   physical addr: 0000005690060000
[710964.014949]   page addr: 0000005690060000
[710964.019041]   ------------------------------
[710964.023381] Got mmaped.

[711675.260421] #########  after zap_vma_ptes( -1 ) ############# 
[711675.266326]   ------------------------------
[711675.270667]   virtual user addr: 000000003e2d0000
[711675.275438]   pgd: ffffa05fd057a000 (0000005fd5800003) 
[711675.275441]   p4d: ffffa05fd057a000 (0000005fd5800003) 
[711675.280743]   pud: ffffa05fd057a000 (0000005fd5800003) 
[711675.286044]   pmd: ffff805fd5800008 (0000005f04c40003) 
[711675.291335]   pte: ffff805f04c4f168 (00e8005683640f53) 
[711675.296637]   p4d_page: ffff7fe017f56000
[711675.305929]   pud_page: ffff7fe017f56000
[711675.309924]   pmd_page: ffff7fe017c13100
[711675.313918]   pte_page: ffff7fe015a0d900
[711675.317923]   physical addr: 0000005683640000
[711675.322350]   page addr: 0000005683640000
[711675.326440]   ------------------------------
[711675.330944] sample_release
```
+ after zap_vma_ptes物理地址仍然存在,返回值是-1       

+    pgd 、p4d、pud、pmd、pte_val(*pte) 的值都不是0    


```Text    
This function only unmaps ptes assigned to VM_PFNMAP vmas.

The entire address range must be fully contained within the vma.

Returns 0 if successful.
```

#  VM_PFNMAP   



```
#if 1
    if (follow_pfn(vma, arg, &pfn))
    {
        //vma->vm_flags |= VM_IO | VM_DONTCOPY | VM_DONTEXPAND | VM_NORESERVE |
        //                      VM_DONTDUMP | VM_PFNMAP;
        vma->vm_flags |= VM_PFNMAP;
        printk(KERN_INFO "no page for vma addr  %lu \n",arg);
        handle_mm_fault(vma, arg, FAULT_FLAG_WRITE);

    }
    printk_pagetable(arg);
#if 0
    res = get_user_pages_remote(current, current->mm,
                                arg , 1, 0,  pages, &vma,NULL);
    pr_info("#########  after get_user_pages_remote ############# \n");
    printk_pagetable(arg);

    page = pages[0];
    if (res < 1) {
        printk(KERN_INFO "GUP error: %d\n", res);
        free_page((unsigned long) page);
        return -EFAULT;
    }
#endif
#else
        res = get_user_pages(
                arg ,
                1,
                1,
                &page,
                NULL);
#endif
```
+ vma->vm_flags |= VM_PFNMAP; vma->vm_flags设置VM_PFNMAP   


+  vma->vm_flags设置VM_PFNMAP 后，不能调用get_user_pages_remote（返回GUP error: -14） 


```
[root@centos7 handle_mm_fault]# ./user_test 
data is 
```


```
[712385.994515] sample_open
[712385.997136] sample_write
[712385.999748] no page for vma addr  454033408 
[712386.004119]   ------------------------------
[712386.008462]   virtual user addr: 000000001b100000
[712386.013236]   pgd: ffff805f053c7800 (0000005f65a80003) 
[712386.013238]   p4d: ffff805f053c7800 (0000005f65a80003) 
[712386.018544]   pud: ffff805f053c7800 (0000005f65a80003) 
[712386.023836]   pmd: ffff805f65a80000 (0000005f36430003) 
[712386.029139]   pte: ffff805f3643d880 (00e800568f1a0f53) 
[712386.034440]   p4d_page: ffff7fe017d96a00
[712386.043724]   pud_page: ffff7fe017d96a00
[712386.047729]   pmd_page: ffff7fe017cd90c0
[712386.051723]   pte_page: ffff7fe015a3c680
[712386.055727]   physical addr: 000000568f1a0000
[712386.060155]   page addr: 000000568f1a0000
[712386.064250]   ------------------------------
[712386.068687] #########  after zap_vma_ptes( 0 ) ############# 
[712386.074507]   ------------------------------
[712386.078846]   virtual user addr: 000000001b100000
[712386.083617]   pgd: ffff805f053c7800 (0000005f65a80003) 
[712386.083619]   p4d: ffff805f053c7800 (0000005f65a80003) 
[712386.088923]   pud: ffff805f053c7800 (0000005f65a80003) 
[712386.094225]   pmd: ffff805f65a80000 (0000005f36430003) 
[712386.099515]   pte: ffff805f3643d880 (0000000000000000) 
[712386.104813]   p4d_page: ffff7fe017d96a00
[712386.114105]   pud_page: ffff7fe017d96a00
[712386.118100]   pmd_page: ffff7fe017cd90c0
[712386.122093]   pte_page: ffff7fe000000000
[712386.126096]   physical addr: 0000000000000000
[712386.130522]   page addr: 0000000000000000
[712386.134615]   ------------------------------
[712386.139086] sample_release
```
after zap_vma_ptes物理地址不存在,返回值是0 

+  pgd 、p4d、pud、pmd的值都不是0    

+ 但是 pte_val(*pte) 是 (0000000000000000) 

#  vm_normal_page

关于normal页面和special页面？    
vm_normal_page根据pte来返回normal paging页面的struct page结构。   

一些特殊映射的页面是不会返回struct page结构的，这些页面不希望被参与到内存管理的一些活动中，如页面回收、页迁移和KSM等。

内核尝试用pte_mkspecial()宏来设置PTE_SPECIAL软件定义的比特位，主要用途有：      

+ 内核的零页面zero page    
+   大量的驱动程序使用remap_pfn_range()函数来实现映射内核页面到用户空间。这些用户程序使用的VMA通常设置了(VM_IO|VM_PFNMAP|VM_DONTEXPAND|VM_DONTDUMP)    
+ vm_insert_page()/vm_insert_pfn()映射内核页面到用户空间    


vm_normal_page()函数把page页面分为两阵营，一个是normal page，另一个是special page。   

normal page通常指正常mapping的页面，例如匿名页面、page cache和共享内存页面等。   
special page通常指不正常mapping的页面，这些页面不希望参与内存管理的回收或者合并功能，比如：  
+ VM_IO：为IO设备映射   
+ VM_PFN_MAP：纯PFN映射  
+ VM_MIXEDMAP：固定映射   