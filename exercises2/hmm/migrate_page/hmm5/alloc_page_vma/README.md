
# alloc_page_vma
```
static vm_fault_t vma_fault(struct vm_fault* vmf)
{
    struct vm_area_struct *vma = vmf->vma;
    //struct page* start_page = alloc_page(GFP_HIGHUSER_MOVABLE | __GFP_ZERO);
    struct page* start_page = alloc_page_vma(GFP_HIGHUSER_MOVABLE, vma, vma->vm_start);
    if(!start_page)
        return VM_FAULT_SIGBUS;
    unsigned long phy_addr = (page_to_pfn(start_page)<< PAGE_SHIFT) + vma->vm_pgoff;
    struct my_page * my_page = kmalloc(sizeof(struct my_page),GFP_KERNEL|GFP_ATOMIC);
    vmf->page = start_page;
    printk(KERN_INFO "alloc page frame struct is @ %p", start_page);
    //printk("fault, is_write = %d, vaddr = %lx, paddr = %lx\n", vmf->flags & FAULT_FLAG_WRITE, vmf->address, (size_t)virt_to_phys(page_address(page)));
    printk("fault, is_write = %d, vaddr = %lx, paddr = %lx\n", vmf->flags & FAULT_FLAG_WRITE, vmf->address, phy_addr);
    my_page->page = start_page;
    list_add_tail(&my_page->list, &page_list);
    return 0;
}
```

# test
```
root@ubuntux86:# insmod  hybridmem_test.ko 
root@ubuntux86:# mknod /dev/hybridmem c 224 0
root@ubuntux86:# ./mmap_test 
0
virt addr 7f3821010000, phy addr 113125000,phy addr 113125000
phy addr 13bb07000
root@ubuntux86:# dmesg | tail -n 10
[   11.992421] start_addr=(0x20000), end_addr=(0x40000), buffer_size=(0x20000), smp_number_max=(16384)
[   12.054647] IPv6: ADDRCONF(NETDEV_CHANGE): wlxe0e1a91deeb2: link becomes ready
[  154.327106] *********** moudle 'hybridmem' begin :
[  217.058708] mmap, start = 7f3821010000, end = 7f3821012000
[  217.058724] alloc page frame struct is @ 00000000e0d1a180
[  217.058729] fault, is_write = 1, vaddr = 7f3821010000, paddr = 113125000
[  217.058740] alloc page frame struct is @ 000000001cd83ed0
[  217.058743] fault, is_write = 0, vaddr = 7f3821011000, paddr = 13bb07000
[  217.059019] page frame struct is @ 00000000e0d1a180, and user vaddr 7f3821010000, paddr 113125000
[  217.059025] buffer phy addr 113125000, virt addr 7f3821010000
```