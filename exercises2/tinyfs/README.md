# linux 系统调用read 剖析

[linux 系统调用read 剖析](https://blog.csdn.net/u011159820/article/details/112439457)   


> ##  inode->i_op pk  inode->i_mapping->a_ops
```
void ext2_read_inode (struct inode * inode)
{
……
if (S_ISREG(inode->i_mode)) {
inode->i_op = &ext2_file_inode_operations;
inode->i_fop = &ext2_file_operations;
if (test_opt(inode->i_sb, NOBH))
inode->i_mapping->a_ops = &ext2_nobh_aops;
else
inode->i_mapping->a_ops = &ext2_aops;
}
……
}
```
从代码中可以看出，如果该 inode 所关联的文件是普通文件，则将变量 ext2_file_operations 的地址赋予 inode 对象的 i_fop 成员。所以可以知道： inode->i_fop.read 函数指针所指向的函数为 ext2_file_operations 变量的成员 read 所指向的函数。下面来看一下 ext2_file_operations 变量的初始化过程，ext2_file_operations 的初始化代码如下：
```
struct file_operations ext2_file_operations = {
.llseek = generic_file_llseek,
.read = generic_file_read,
.write = generic_file_write,
.aio_read = generic_file_aio_read,
.aio_write = generic_file_aio_write,
.ioctl = ext2_ioctl,
.mmap = generic_file_mmap,
.open = generic_file_open,
.release = ext2_release_file,
.fsync = ext2_sync_file,
.readv = generic_file_readv,
.writev = generic_file_writev,
.sendfile = generic_file_sendfile,

};
```
该成员 read 指向函数 generic_file_read 。所以， inode->i_fop.read 指向 generic_file_read 函数，进而 file->f_op.read 指向 generic_file_read 函数。最终得出结论： generic_file_read 函数才是 ext2 层的真实入口。   

**Ext2 文件系统层的处理**   

![images](fs1.jpg)

由图 4 可知，该层入口函数 generic_file_read 调用函数 __generic_file_aio_read ，后者判断本次读请求的访问方式，如果是直接 io （filp->f_flags 被设置了 O_DIRECT 标志，即不经过 cache）的方式，则调用 generic_file_direct_IO 函数；如果是 page cache 的方式，则调用 do_generic_file_read 函数。函数 do_generic_file_read 仅仅是一个包装函数，它又调用 do_generic_mapping_read 函数。

在讲解 do_generic_mapping_read 函数都作了哪些工作之前，我们再来看一下文件在内存中的缓存区域是被怎么组织起来的。   

**文件的 page cache 结构**   
 
![images](fs2.jpg)
 图5显示了一个文件的 page cache 结构。文件被分割为一个个以 page 大小为单元的数据块,这些数据块（页）被组织成一个多叉树（称为 radix 树）。树中所有叶子节点为一个个页帧结构（struct page），表示了用于缓存该文件的每一个页。在叶子层最左端的第一个页保存着该文件的前4096个字节（如果页的大小为4096字节），接下来的页保存 着文件第二个4096个字节，依次类推。树中的所有中间节点为组织节点，指示某一地址上的数据所在的页。此树的层次可以从0层到6层，所支持的文件大小从 0字节到16 T 个字节。树的根节点指针可以从和文件相关的 address_space 对象（该对象保存在和文件关联的 inode 对象中）中取得（更多关于 page cache 的结构内容请参见参考资料）。
现在，我们来看看函数 do_generic_mapping_read 都作了哪些工作， do_generic_mapping_read 函数代码较长，本文简要介绍下它的主要流程：

根据文件当前的读写位置，在 page cache 中找到缓存请求数据的 page；如果该页已经最新，将请求的数据拷贝到用户空间；否则， Lock 该页，调用 readpage 函数向磁盘发出添页请求（当下层完成该 IO 操作时会解锁该页），代码：error = mapping->a_ops->readpage(filp, page);再一次 lock 该页，操作成功时，说明数据已经在 page cache 中了，因为只有 IO 操作完成后才可能解锁该页。此处是一个同步点，用于同步数据从磁盘到内存的过程。解锁该页。
到此为止数据已经在 page cache 中了，再将其拷贝到用户空间中（之后 read 调用可以在用户空间返回了）
到此，我们知道：当页上的数据不是最新的时候，该函数调用 mapping->a_ops->readpage 所指向的函数（变量 mapping 为 inode 对象中的 address_space 对象），那么这个函数到底是什么呢？

Readpage 函数的由来

address_space 对象是嵌入在 inode 对象之中的，那么不难想象： address_space 对象成员 a_ops 的初始化工作将会在初始化 inode 对象时进行。

if (test_opt(inode->i_sb, NOBH))
inode->i_mapping->a_ops = &ext2_nobh_aops;
else
inode->i_mapping->a_ops = &ext2_aops;
可以知道 address_space 对象的成员 a_ops 指向变量 ext2_aops 或者变量 ext2_nobh_aops 。这两个变量的初始化如清单5所示。

变量 ext2_aops 和变量 ext2_nobh_aops 的初始化，代码如下：
```
struct address_space_operations ext2_aops = {
.readpage =
ext2_readpage,
.readpages = ext2_readpages,
.writepage = ext2_writepage,
.sync_page = block_sync_page,
.prepare_write = ext2_prepare_write,
.commit_write = generic_commit_write,
.bmap = ext2_bmap,
.direct_IO = ext2_direct_IO,
.writepages = ext2_writepages,
};

struct address_space_operations ext2_nobh_aops = {
.readpage
= ext2_readpage,
.readpages = ext2_readpages,
.writepage = ext2_writepage,
.sync_page = block_sync_page,
.prepare_write = ext2_nobh_prepare_write,
.commit_write = nobh_commit_write,
.bmap = ext2_bmap,
.direct_IO = ext2_direct_IO,
.writepages = ext2_writepages,
};
```
从上述代码中可以看出，不论是哪个变量，其中的 readpage 成员都指向函数 ext2_readpage 。所以可以断定：函数do_generic_mapping_read 最终调用 ext2_readpage 函数处理读数据请求。
到此为止， ext2 文件系统层的工作结束。
 
 

# address_space_operations def_blk_aops


```
const struct address_space_operations def_blk_aops = {
        .dirty_folio    = block_dirty_folio,
        .invalidate_folio = block_invalidate_folio,
        .read_folio     = blkdev_read_folio,
        .readahead      = blkdev_readahead,
        .writepage      = blkdev_writepage,
        .write_begin    = blkdev_write_begin,
        .write_end      = blkdev_write_end,
        .writepages     = blkdev_writepages,
        .direct_IO      = blkdev_direct_IO,
        .migrate_folio  = buffer_migrate_folio_norefs,
        .is_dirty_writeback = buffer_check_dirty_writeback,
};
struct block_device *bdev_alloc(struct gendisk *disk, u8 partno)
{
        struct block_device *bdev;
        struct inode *inode;

        inode = new_inode(blockdev_superblock);
        if (!inode)
                return NULL;
        inode->i_mode = S_IFBLK;
        inode->i_rdev = 0;
        inode->i_data.a_ops = &def_blk_aops;
        mapping_set_gfp_mask(&inode->i_data, GFP_USER);

        bdev = I_BDEV(inode);
        mutex_init(&bdev->bd_fsfreeze_mutex);
        spin_lock_init(&bdev->bd_size_lock);
        bdev->bd_partno = partno;
        bdev->bd_inode = inode;
        bdev->bd_queue = disk->queue;
        bdev->bd_stats = alloc_percpu(struct disk_stats);
        if (!bdev->bd_stats) {
                iput(inode);
                return NULL;
        }
        bdev->bd_disk = disk;
        return bdev;
}

```

> ##  do_read_cache_folio --> blkdev_read_folio

```
> [   20.878128]  memcg_slab_post_alloc_hook+0xa8/0x1c8
> [   20.878132]  kmem_cache_alloc+0x18c/0x338
> [   20.878135]  alloc_buffer_head+0x28/0xa0
> [   20.878138]  folio_alloc_buffers+0xe8/0x1c0
> [   20.878141]  folio_create_empty_buffers+0x2c/0x1e8
> [   20.878143]  folio_create_buffers+0x58/0x80
> [   20.878145]  block_read_full_folio+0x80/0x450
> [   20.878148]  blkdev_read_folio+0x24/0x38
> [   20.956921]  filemap_read_folio+0x60/0x138
> [   20.956925]  do_read_cache_folio+0x180/0x298
> [   20.965270]  read_cache_page+0x24/0x90
> [   20.965273]  __arm64_sys_swapon+0x2e0/0x1208
> [   20.965277]  invoke_syscall+0x78/0x108
> [   20.965282]  el0_svc_common.constprop.0+0x48/0xf0
> [   20.981702]  do_el0_svc+0x24/0x38
> [   20.993773]  el0t_64_sync_handler+0x100/0x130
> [   20.993776]  el0t_64_sync+0x190/0x198
> [   20.993779] ---[ end trace 0000000000000000 ]---
> [   20.999972] Adding 999420k swap on /dev/mapper/eng07sys--r113--vg-swap_1.
> Priority:-2 extents:1 across:999420k SS
```

# filemap_map_pages 

```
static const struct vm_operations_struct blkdev_dax_vm_ops = {
	.open		= blkdev_vm_open,
	.close		= blkdev_vm_close,
	.fault		= blkdev_dax_fault,
	.pmd_fault	= blkdev_dax_pmd_fault,
	.pfn_mkwrite	= blkdev_dax_fault,
};

static const struct vm_operations_struct blkdev_default_vm_ops = {
	.open		= blkdev_vm_open,
	.close		= blkdev_vm_close,
	.fault		= filemap_fault,
	.map_pages	= filemap_map_pages,
};
```

> ## Page Cache 的插入

我们在Linux内核源码分析-内存请页机制中分析了缺页中断时，当访问的 Page Table 尚未分配，即vma对应磁盘上的某一个文件时，会调用vma->vm_ops->fault(vmf)对应的文件系统的缺页处理函数。

```
page = page_cache_alloc();
/* ... */
__add_to_page_cache(page, mapping, index, hash);
```
以ext4为例，ext4_filemap_fault()为缺页处理函数，具体调用了内存管理模块的filemap_fault()来完成:


```
vm_fault_t filemap_fault(struct vm_fault *vmf)
{
        /* 查找缺页是否存在于 Page Cache.
           mapping 为该文件的 adress_space,
           offset 为该页的偏移量.
         */
        page = find_get_page(mapping, offset);
        if (likely(page) && !(vmf->flags & FAULT_FLAG_TRIED)) {
                /* 假如存在，进行预读 */
                do_async_mmap_readahead(vmf->vma, ra, file, page, offset);
        } else if (!page) {
                /* 假如不存在，则进行预读，之后立即尝试 Page Cache 查找，
                   假如仍然不存在，则跳转 no_cached_page.
                 */
                do_sync_mmap_readahead(vmf->vma, ra, file, offset);
                count_vm_event(PGMAJFAULT);
                count_memcg_event_mm(vmf->vma->vm_mm, PGMAJFAULT);
                ret = VM_FAULT_MAJOR;
retry_find:
                page = find_get_page(mapping, offset);
                if (!page)
                        goto no_cached_page;
        }

        /* ... */

        vmf->page = page;
        return ret | VM_FAULT_LOCKED;

no_cached_page:
        /* 1. 申请分配一个 Page
           2. 将该 Page 添加至Page Cache
           3. 调用 address_space 的 readpage() 函数完成该 Page 内容的读取
        */
        error = page_cache_read(file, offset, vmf->gfp_mask);
				/* ... */
}
EXPORT_SYMBOL(filemap_fault);


```

Page Cache 的插入主要流程如下:

判断查找的 Page 是否存在于 Page Cache，存在即直接返回   
否则通过 Linux 内核物理内存分配介绍的伙伴系统分配一个空闲的 Page.   
将 Page 插入 Page Cache，即插入address_space的i_pages.   
调用address_space的readpage()来读取指定 offset 的 Page.  


> ##  find_get_page  -->  pagecache_get_page

```
/**
 * find_get_page - find and get a page reference
 * @mapping: the address_space to search
 * @offset: the page index
 *
 * Looks up the page cache slot at @mapping & @offset.  If there is a
 * page cache page, it is returned with an increased refcount.
 *
 * Otherwise, %NULL is returned.
 */
static inline struct page *find_get_page(struct address_space *mapping,
                                        pgoff_t offset)
{
        return pagecache_get_page(mapping, offset, 0, 0);
}
```

# references

[300行代码带你实现一个Linux文件系统](https://zhuanlan.zhihu.com/p/579011810)     