
#  struct file_operations nvfs_dev_fops


```
struct file_operations nvfs_dev_fops = {
	.compat_ioctl = nvfs_ioctl,
	.unlocked_ioctl = nvfs_ioctl,
	.open = nvfs_open,
	.release = nvfs_close,
        .mmap = nvfs_mgroup_mmap,
        .owner = THIS_MODULE,
};
```
#  const struct vm_operations_struct nvfs_mmap_ops

```
static const struct vm_operations_struct nvfs_mmap_ops = {
	.open = nvfs_vma_open,
#ifdef HAVE_VM_OPS_SPLIT
	.split = nvfs_vma_split,
#else
	.may_split = nvfs_vma_split,
#endif
	.mremap = nvfs_vma_mremap,
	.close = nvfs_vma_close,
        .fault = nvfs_vma_fault,
        .pfn_mkwrite = nvfs_pfn_mkwrite,
        .page_mkwrite = nvfs_page_mkwrite,
};
```