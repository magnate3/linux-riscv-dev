
#  Heterogeneous Memory Management 
HMM(Heterogeneous MemoryManagement)提供一个统一和简单的API来映射进程地址空间到外设的MMU上，这样进程地址空间的改变就可以反映到外设的页表中。      
建立一个共享的地址空间，系统内存可以透明地迁移到外设内存中。HMM新定义一个名为ZONE_DEVICE的zone类型，外设内存被标记为ZONE_DEVICE，系统内存可以迁移到这个zone中。   
从CPU角度看，就想把系统内存swapping到ZONE_DEVICE中，当CPU需要访问这些内存时会触发一个缺页中断，然后再把这些内存从外设中迁移回到系统内存。   

Address space mirroring implementation and API
Address space mirroring’s main objective is to allow duplication of a range of CPU page table into a device page table; HMM helps keep both synchronized. A device driver that wants to mirror a process address space must start with the registration of a mmu_interval_notifier:
```
int mmu_interval_notifier_insert(struct mmu_interval_notifier *interval_sub,
                                 struct mm_struct *mm, unsigned long start,
                                 unsigned long length,
                                 const struct mmu_interval_notifier_ops *ops);
```								 ·
During the ops->invalidate() callback the device driver must perform the update action to the range (mark range read only, or fully unmap, etc.). The device must complete the update before the driver callback returns.   

When the device driver wants to populate a range of virtual addresses, it can use:   
```
int hmm_range_fault(struct hmm_range *range);
```
It will trigger a page fault on missing or read-only entries if write access is requested (see below). Page faults use the generic mm page fault code path just like a CPU page fault.   

Both functions copy CPU page table entries into their pfns array argument. Each entry in that array corresponds to an address in the virtual range. HMM provides a set of flags to help the driver identify special CPU page table entries.   

Locking within the sync_cpu_device_pagetables() callback is the most important aspect the driver must respect in order to keep things properly synchronized. The usage pattern is:   
```
int driver_populate_range(...)
{
     struct hmm_range range;
     ...

     range.notifier = &interval_sub;
     range.start = ...;
     range.end = ...;
     range.hmm_pfns = ...;

     if (!mmget_not_zero(interval_sub->notifier.mm))
         return -EFAULT;

again:
     range.notifier_seq = mmu_interval_read_begin(&interval_sub);
     mmap_read_lock(mm);
     ret = hmm_range_fault(&range);
     if (ret) {
         mmap_read_unlock(mm);
         if (ret == -EBUSY)
                goto again;
         return ret;
     }
     mmap_read_unlock(mm);

     take_lock(driver->update);
     if (mmu_interval_read_retry(&ni, range.notifier_seq) {
         release_lock(driver->update);
         goto again;
     }

     /* Use pfns array content to update device page table,
      * under the update lock */

     release_lock(driver->update);
     return 0;
}
```
The driver->update lock is the same lock that the driver takes inside its invalidate() callback. That lock must be held before calling mmu_interval_read_retry() to avoid any race with a concurrent CPU page table update.  

# references
[Heterogeneous Memory Management (HMM)](https://www.infradead.org/~mchehab/kernel_docs/vm/hmm.html)   