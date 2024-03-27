
/proc/pid/pagemap. This file lets a userspace process find out which physical frame each virtual page is mapped to. It contains one 64-bit value for each virtual page, containing the following data (from fs/proc/task_mmu.c, above pagemap_read):

Bits 0-54 page frame number (PFN) if present
Bits 0-4 swap type if swapped
Bits 5-54 swap offset if swapped
Bit 55 pte is soft-dirty (see Soft-Dirty PTEs)
Bit 56 page exclusively mapped (since 4.2)
Bit 57 pte is uffd-wp write-protected (since 5.13) (see Userfaultfd)
Bits 58-60 zero
Bit 61 page is file-page or shared-anon (since 3.5)
Bit 62 page swapped
Bit 63 page present