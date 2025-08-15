
# os

```
[root@centos7 handle_mm_fault]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 handle_mm_fault]# 
```

#  get_user_pages_remote 之前已经分配内存

```
if (follow_pfn(vma, arg, &pfn))
    {
        printk(KERN_INFO "no page for vma addr  %lu \n",arg);
        handle_mm_fault(vma, arg, FAULT_FLAG_WRITE);
    }
    printk_pagetable(arg);
    res = get_user_pages_remote(current, current->mm,
                                arg , 1, 0,  pages, &vma,NULL);
    pr_info("#########  after get_user_pages_remote ############# \n");
    printk_pagetable(arg);
```
follow_pfn 查看页表是否存在    
+ 1  log1 说明handle_mm_fault被调用
```
no page for vma addr  125960192  
```

+ 2 log2  physical addr: 00000056bdb50000  

+ 3 log3   physical addr: 00000056bdb50000   

log2和log3说明 get_user_pages_remote没有改变虚拟地址和物理地址映射关系   

```
[618038.950138] sample_open
[618038.952808] sample_write
[618038.955422] no page for vma addr  125960192 
[618038.959763]   ------------------------------
[618038.964117]   virtual user addr: 0000000007820000
[618038.968891]   pgd: ffffa05fd4c46600 (0000005fda4d0003) 
[618038.968894]   p4d: ffffa05fd4c46600 (0000005fda4d0003) 
[618038.974203]   pud: ffffa05fd4c46600 (0000005fda4d0003) 
[618038.979495]   pmd: ffff805fda4d0000 (0000005fd2800003) 
[618038.984797]   pte: ffff805fd2803c10 (00e80056bdb50f53) 
[618038.990088]   p4d_page: ffff7fe017f69340
[618038.999384]   pud_page: ffff7fe017f69340
[618039.003389]   pmd_page: ffff7fe017f4a000
[618039.007386]   pte_page: ffff7fe015af6d40
[618039.011393]   physical addr: 00000056bdb50000
[618039.015820]   page addr: 00000056bdb50000
[618039.019899]   ------------------------------
[618039.024270] #########  after get_user_pages_remote ############# 
[618039.030424]   ------------------------------
[618039.034775]   virtual user addr: 0000000007820000
[618039.039546]   pgd: ffffa05fd4c46600 (0000005fda4d0003) 
[618039.039549]   p4d: ffffa05fd4c46600 (0000005fda4d0003) 
[618039.044851]   pud: ffffa05fd4c46600 (0000005fda4d0003) 
[618039.050142]   pmd: ffff805fda4d0000 (0000005fd2800003) 
[618039.055446]   pte: ffff805fd2803c10 (00e80056bdb50f53) 
[618039.060736]   p4d_page: ffff7fe017f69340
[618039.070025]   pud_page: ffff7fe017f69340
[618039.074027]   pmd_page: ffff7fe017f4a000
[618039.078021]   pte_page: ffff7fe015af6d40
[618039.082023]   physical addr: 00000056bdb50000
[618039.086449]   page addr: 00000056bdb50000
[618039.090528]   ------------------------------
[618039.094877] Got mmaped.

[618039.099126] sample_release
```