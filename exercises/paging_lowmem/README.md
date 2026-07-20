
# os

```
uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
```

# insmod  paging_lowmem.ko

```
[ 3962.586622] vaddr to paddr module is running..
[ 3962.586629] cr0 = 0x80050033, cr3 = 0x14e6ee000
[ 3962.586634] PGDIR_SHIFT = 39
[ 3962.586637] P4D_SHIFT = 39
[ 3962.586638] PUD_SHIFT = 30
[ 3962.586640] PMD_SHIFT = 21
[ 3962.586642] PAGE_SHIFT = 12
[ 3962.586643] PTRS_PER_PGD = 512
[ 3962.586646] PTRS_PER_P4D = 1
[ 3962.586647] PTRS_PER_PUD = 512
[ 3962.586649] PTRS_PER_PMD = 512
[ 3962.586651] PTRS_PER_PTE = 512
[ 3962.586652] PAGE_MASK = 0xfffffffffffff000

[ 3962.586657] get_page_vaddr=0xffff8cfc110c3000
[ 3962.586660] pgd_val = 0x161801067, pgd_index = 281
[ 3962.586663] p4d_val = 0x161801067, p4d_index = 0
[ 3962.586666] pud_val = 0x101537063, pud_index = 496
[ 3962.586668] pmd_val = 0x11113a063, pmd_index = 136
[ 3962.586670] pte_val = 0x80000001110c3163, ptd_index = 195
[ 3962.586673] page_addr = 80000001110c3000, page_offset = 0
[ 3962.586675] vaddr = ffff8cfc110c3000, paddr = 80000001110c3000
```

```
ubuntu@ubuntux86:~$ dmesg | tail -n 30
[ 4846.386637] cr0 = 0x80050033, cr3 = 0x14e676000
[ 4846.386642] PGDIR_SHIFT = 39
[ 4846.386645] P4D_SHIFT = 39
[ 4846.386647] PUD_SHIFT = 30
[ 4846.386649] PMD_SHIFT = 21
[ 4846.386650] PAGE_SHIFT = 12
[ 4846.386652] PTRS_PER_PGD = 512
[ 4846.386654] PTRS_PER_P4D = 1
[ 4846.386656] PTRS_PER_PUD = 512
[ 4846.386658] PTRS_PER_PMD = 512
[ 4846.386659] PTRS_PER_PTE = 512
[ 4846.386661] PAGE_MASK = 0xfffffffffffff000

[ 4846.386666] get_page_vaddr=0xffff8cfc6608a000
[ 4846.386669] pgd_val = 0x161801067, pgd_index = 281
[ 4846.386672] p4d_val = 0x161801067, p4d_index = 0
[ 4846.386675] pud_val = 0x1001db063, pud_index = 497
[ 4846.386677] pmd_val = 0x109b7e063, pmd_index = 304
[ 4846.386679] pte_val = 0x800000016608a163, ptd_index = 138
[ 4846.386682] page_addr = 800000016608a000, page_offset = 0
[ 4846.386685] vaddr = ffff8cfc6608a000, paddr = 800000016608a000
[ 4859.730262] vaddr to paddr module is leaving..
```