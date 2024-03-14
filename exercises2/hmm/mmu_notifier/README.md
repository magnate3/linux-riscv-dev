
# os


```Shell
root@ubuntux86:# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:# 
```

# unmap_region -->……  __mmu_notifier_invalidate_range
```
[11501.184110] Call Trace:
[11501.184112]  <TASK>
[11501.184113]  dump_stack+0x7d/0x9c
[11501.184120]  my_mmu_invalidate_range+0xe/0x1c [mmu_test]
[11501.184125]  __mmu_notifier_invalidate_range+0x58/0x90
[11501.184131]  tlb_flush_mmu+0x138/0x140
[11501.184137]  tlb_finish_mmu+0x42/0x80
[11501.184142]  unmap_region+0xef/0x120
[11501.184148]  ? __wake_up+0x13/0x20
[11501.184154]  ? tty_write_unlock+0x31/0x40
[11501.184160]  __do_munmap+0x26f/0x500
[11501.184165]  __vm_munmap+0x7f/0x130
[11501.184170]  __x64_sys_munmap+0x2d/0x40
[11501.184175]  do_syscall_64+0x61/0xb0
[11501.184178]  ? vfs_write+0x1c3/0x250
[11501.184185]  ? exit_to_user_mode_prepare+0x3d/0x1c0
[11501.184192]  ? ksys_write+0x67/0xe0
[11501.184195]  ? syscall_exit_to_user_mode+0x27/0x50
[11501.184200]  ? __x64_sys_write+0x1a/0x20
[11501.184203]  ? do_syscall_64+0x6e/0xb0
[11501.184206]  ? syscall_exit_to_user_mode+0x27/0x50
[11501.184211]  ? __x64_sys_write+0x1a/0x20
[11501.184214]  ? do_syscall_64+0x6e/0xb0
[11501.184217]  ? exc_page_fault+0x8f/0x170
[11501.184221]  ? asm_exc_page_fault+0x8/0x30
[11501.184228]  entry_SYSCALL_64_after_hwframe+0x44/0xae
```
# insmod  mmu_test.ko 
```
root@ubuntux86:# insmod  mmu_test.ko 
root@ubuntux86:# ./mmap_test 
addr: 0x7f753bf19000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
root@ubuntux86:# dmesg | tail -n 10
[13773.061195] Hello modules on myMMU
[13776.319887] myMMU notifier: invalidate_range_start.
[13776.319894] myMMU notifier: invalidate_range.
[13776.319896] myMMU notifier: change_pte
[13776.319897] myMMU notifier: invalidate_range_end.
[13776.320050] myMMU notifier: invalidate_range_start.
[13776.320059] myMMU notifier: invalidate_range.
[13776.320060] myMMU notifier: invalidate_range_end.
[13776.320063] myMMU notifier: invalidate_range.
[13776.320067] myMMU notifier: release
root@ubuntux86:# 
```