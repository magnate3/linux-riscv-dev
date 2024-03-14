

# insmod  alloc_page_test3.ko

```
root@ubuntux86:/work/kernel_learn# dmesg | tail -n 32
[20469.259502] ***************v2p moudle run:
[20469.259507] print page relate macro:
[20469.259509] cr0=0x80050033, cr3=0x1087cc000
[20469.259513] pgdir_SHIFT = 39
[20469.259515] P4D_SHIFT = 39
[20469.259516] PUD_SHIFT = 30
[20469.259518] PMD_SHIFT = 21
[20469.259519] PAGE_SHIFT = 12
[20469.259521] PTRS_PER_PGD = 512
[20469.259522] PTRS_PER_P4D = 1
[20469.259524] PTRS_PER_PUD = 512
[20469.259525] PTRS_PER_PMD = 512
[20469.259526] PTRS_PER_PTE = 512
[20469.259528] PAGE_MASK = 0xfffffffffffff000
[20469.259529] vaddr to phy addr entry!
[20469.259531] __get_free_page, alloc the free page vaddr=0xffff985a8d948000
[20469.259533] ***************before write, vaddr2paddr:
[20469.259535] pgd_val=0x5eb401067, pdg_index=0x130
[20469.259537] p4d_val=0x5eb401067, p4d_index=0x0
[20469.259539] pud_val=0x101534063, pud_index=0x16a
[20469.259541] pmd_val=0x10d82b063, pmd_index=0x6c
[20469.259543] pte_val=0x800000010d948163, pte_index=0x148
[20469.259545] page_offset=0x0, page_addr=0x800000010d948000
[20469.259547] vaddr=0xffff985a8d948000, paddr=0x800000010d948000
[20469.259549] ***************after write, vaddr2paddr :
[20469.259550] pgd_val=0x5eb401067, pdg_index=0x130
[20469.259552] p4d_val=0x5eb401067, p4d_index=0x0
[20469.259554] pud_val=0x101534063, pud_index=0x16a
[20469.259555] pmd_val=0x10d82b063, pmd_index=0x6c
[20469.259557] pte_val=0x800000010d948163, pte_index=0x148
[20469.259559] page_offset=0x0, page_addr=0x800000010d948000
[20469.259561] vaddr=0xffff985a8d948000, paddr=0x800000010d948000
```
root@ubuntux86:/work/kernel_learn# insmod   page_test.ko
```
[25167.144856] virtual kernel memory layout:
                   fixmap  : 0xffffffffff579000 - 0xffffffffff7ff000   (2584 kB)
                   vmalloc : 0xffffa9e540000000 - 0xffffc9e53fffffff   (33554431 MB)
                   lowmem  : 0xffff985980000000 - 0xffff98620d800000   (35032 MB)
[25168.190434] True!!
```