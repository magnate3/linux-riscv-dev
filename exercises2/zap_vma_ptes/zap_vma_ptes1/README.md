
# 内核zap_vma_ptes
my_write调用zap_vma_ptes   
```
static ssize_t  my_write(struct file *file, const char __user *buf, size_t count, loff_t *off)
{

     unsigned long vaddr = (unsigned long)buf;
     struct vm_area_struct *vma;
     vma = find_vma(current->mm, vaddr);
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
     down_read(&current->mm->mmap_sem);
#else
     down_read(&current->mm->context.ldt_usr_sem);
#endif
     if(vma)
     {
          zap_vma_ptes(vma, vma->vm_start, vma->vm_end - vma->vm_start);
     }
#if  LINUX_VERSION_CODE <= KERNEL_VERSION(5,13,0)
     up_read(&current->mm->mmap_sem);
#else
     up_read(&current->mm->context.ldt_usr_sem);
#endif
        return 0;
}
```

# 用户态
通过write调用内核态的zap_vma_ptes   

```
#if 1
        printf("phy addr of addr 0x%lx \n",mem_virt2phy(addr));
        write(fd, addr, len);
        printf("after zap_vma_ptes, phy addr of addr  0x%lx \n",mem_virt2phy(addr));
        ret = posix_memalign((void **)&ptr, pagesize, pagesize);
        if(!ret)
        {
            memcpy(ptr, "krishna", strlen("krishna"));
            printf("phy addr of ptr  0x%lx \n",mem_virt2phy(ptr));
            write(fd, ptr, len);
            printf("after zap_vma_ptes, phy addr of ptr 0x%lx \n",mem_virt2phy(addr));
            free(ptr);
        }
        else
        {
            fprintf(stderr, "posix_memalign: %s\n", strerror (ret));
        }
#endif
```

# 运行

```
root@ubuntux86:# ./mmap_test 
addr: 0x7fb08fb31000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
Zero page frame number
phy addr of addr 0x0 
Zero page frame number
after zap_vma_ptes, phy addr of addr  0x0 
phy addr of ptr  0x15a273000 
Zero page frame number
after zap_vma_ptes, phy addr of ptr 0x0 
root@ubuntux86:# 
```
phy addr of addr 0x0 这是因为内存是内核 kzalloc分配的的，mmap没有采用page fault实现       


```
phy addr of ptr  0x15a273000 
Zero page frame number
after zap_vma_ptes, phy addr of ptr 0x0 
```
执行zap_vma_ptes后，posix_memalign分配的内存释放，phy addr of ptr从0x15a273000 变成了0x0    

# tlb_flush_mmu --> __mmu_notifier_invalidate_range

+ 1 munmap触发   
```
[ 1584.284105]  dump_stack+0x7d/0x9c
[ 1584.284114]  my_mmu_invalidate_range+0xe/0x1c [mmu_test]
[ 1584.284120]  __mmu_notifier_invalidate_range_end+0x73/0xd0
[ 1584.284127]  unmap_vmas+0xd0/0xf0
[ 1584.284137]  unmap_region+0xbf/0x120
[ 1584.284143]  ? exit_to_user_mode_prepare+0x3d/0x1c0
[ 1584.284153]  ? ksys_write+0x67/0xe0
[ 1584.284158]  __do_munmap+0x26f/0x500
[ 1584.284165]  __vm_munmap+0x7f/0x130
[ 1584.284171]  __x64_sys_munmap+0x2d/0x40
[ 1584.284178]  do_syscall_64+0x61/0xb0
[ 1584.284182]  ? asm_exc_page_fault+0x8/0x30
[ 1584.284190]  entry_SYSCALL_64_after_hwframe+0x44/0xae
```
+ zap_vma_ptes 触发   
```
[ 1584.283836]  <TASK>
[ 1584.283838]  dump_stack+0x7d/0x9c
[ 1584.283846]  my_mmu_invalidate_range+0xe/0x1c [mmu_test]
[ 1584.283852]  __mmu_notifier_invalidate_range+0x58/0x90
[ 1584.283859]  tlb_flush_mmu+0x138/0x140
[ 1584.283867]  tlb_finish_mmu+0x42/0x80
[ 1584.283873]  zap_page_range_single+0x115/0x170
[ 1584.283882]  ? tty_write+0x11/0x20
[ 1584.283890]  ? common_file_perm+0x72/0x170
[ 1584.283900]  zap_vma_ptes+0x25/0x30
[ 1584.283908]  my_write+0x50/0x73 [mmu_test]
[ 1584.283915]  vfs_write+0xb9/0x250
[ 1584.283923]  ksys_write+0x67/0xe0
[ 1584.283927]  __x64_sys_write+0x1a/0x20
[ 1584.283931]  do_syscall_64+0x61/0xb0
[ 1584.283935]  ? __x64_sys_write+0x1a/0x20
[ 1584.283938]  ? do_syscall_64+0x6e/0xb0
[ 1584.283942]  ? do_syscall_64+0x6e/0xb0
[ 1584.283946]  ? exc_page_fault+0x8f/0x170
[ 1584.283951]  ? asm_exc_page_fault+0x8/0x30
[ 1584.283959]  entry_SYSCALL_64_after_hwframe+0x44/0xae
```
+ 3 handle_mm_fault触发 my_mmu_change_pte      

```
[ 1584.283301]  <TASK>
[ 1584.283303]  dump_stack+0x7d/0x9c
[ 1584.283311]  my_mmu_change_pte+0xe/0x1c [mmu_test]
[ 1584.283318]  __mmu_notifier_change_pte+0x58/0x90
[ 1584.283325]  wp_page_copy+0x484/0x590
[ 1584.283334]  do_wp_page+0xeb/0x2f0
[ 1584.283343]  __handle_mm_fault+0x8b5/0x8e0
[ 1584.283349]  handle_mm_fault+0xda/0x2b0
[ 1584.283354]  do_user_addr_fault+0x1bb/0x650
[ 1584.283359]  exc_page_fault+0x7d/0x170
[ 1584.283365]  ? asm_exc_page_fault+0x8/0x30
[ 1584.283373]  asm_exc_page_fault+0x1e/0x30
```