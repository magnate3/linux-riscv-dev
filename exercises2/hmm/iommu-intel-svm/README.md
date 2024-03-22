#  vfio

```
static const struct mmu_notifier_ops vfio_dma_mmu_notifier_ops = {
	.invalidate_page = vfio_dma_inval_page,
	.invalidate_range_start = vfio_dma_inval_range_start,
};
```

#  intel_mmuops  

```
static const struct mmu_notifier_ops intel_mmuops = {
	.release = intel_mm_release,
	.invalidate_range = intel_invalidate_range,
};

static void intel_invalidate_range(struct mmu_notifier *mn,
                                   struct mm_struct *mm,
                                   unsigned long start, unsigned long end)
{
        struct intel_svm *svm = container_of(mn, struct intel_svm, notifier);

        intel_flush_svm_range(svm, start,
                              (end - start + PAGE_SIZE - 1) >> VTD_PAGE_SHIFT, 0);
}
```

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

#  migrate_pages
Linux提供了migrate_pages系统调用，从old_nodes中获取原内存节点，从new_nodes中获取目的内存节点；然后将当前进程的mm_struct作为参数，调用do_migrate_pages进行迁移操作

```
migrate_pages-------------------------------------页面迁移核心函数
    unmap_and_move
        get_new_page------------------------------分配新页面
        __unmap_and_move--------------------------迁移页面到新页面
            move_to_new_page
                page_mapping----------------------找到页面对应的地址空间
                migrate_page----------------------将旧页面的相关信息迁移到新页面
                    migrate_page_copy
                remove_migration_ptes-------------利用方向映射找到映射旧页面的每个PTE
                    remove_migration_pte----------处理其中一个虚拟地址
```

 