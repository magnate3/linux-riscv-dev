

#  create_shared_memory
```
/*
 * Uses mmap to create a shared memory area for storage of data
 * Used in this file to store the hugepage file map on disk
 */
static void *
create_shared_memory(const char *filename, const size_t mem_size)
{
        void *retval;
        int fd;

        /* if no shared files mode is used, create anonymous memory instead */
        if (internal_config.no_shconf) {
                retval = mmap(NULL, mem_size, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (retval == MAP_FAILED)
                        return NULL;
                return retval;
        }

        fd = open(filename, O_CREAT | O_RDWR, 0600);
        if (fd < 0)
                return NULL;
        if (ftruncate(fd, mem_size) < 0) {
                close(fd);
                return NULL;
        }
        retval = mmap(NULL, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (retval == MAP_FAILED)
                return NULL;
        return retval;
}
```

#   map the segment
# 物理巨页的获取 - 第一次mmap()
在用户空间，我们如何申请利用巨页呢？ 通过在 hugetblfs文件系统类型的 /dev/hugepages 先创建文件/dev/hugepages/rte_hugepage_%d，然后 mmap()系统调用 获取1G的巨页。有别于普通文件系统的内存申请，这里无需设置文件大小，如ftruncate(fd)。
但此时的物理巨页的位置 是用户无法控制的，需后续进一步的排序处理   
 
```
// lib/eal/linux/eal_memory.c
map_all_hugepages()
	// 拼接文件名 /dev/hugepages/rte_hugepage_%s
	eal_get_hugefile_path(hf->filepath, sizeof(hf->filepath),
				hpi->hugedir, hf->file_id);
	// 在/dev/hugepages/ 目录下 新建文件
	fd = open(hf->filepath, O_CREAT | O_RDWR, 0600);
	// 获取物理巨页。注意：多次mmap()，该文件仍 只有1g大小
	virtaddr = mmap(NULL, hugepage_sz, PROT_READ | PROT_WRITE,
			MAP_SHARED | MAP_POPULATE, fd, 0);
``` 
 
```
   /* map the segment, and populate page tables,
                 * the kernel fills this segment with zeros. we don't care where
                 * this gets mapped - we already have contiguous memory areas
                 * ready for us to map into.
                 */
                virtaddr = mmap(NULL, hugepage_sz, PROT_READ | PROT_WRITE,
                                MAP_SHARED | MAP_POPULATE, fd, 0);
                if (virtaddr == MAP_FAILED) {
                        RTE_LOG(DEBUG, EAL, "%s(): mmap failed: %s\n", __func__,
                                        strerror(errno));
                        close(fd);
                        goto out;
                }
```
***MAP_POPULATE***
用MAP_POPULATE强制在mmap系统调用里面把所有的页面都fault in,这样后面访问就不会产生page fault而耽误时间了   
#  attach_segment
```
static int
attach_segment(const struct rte_memseg_list *msl, const struct rte_memseg *ms,
                void *arg)
{
        struct attach_walk_args *wa = arg;
        void *addr;

        if (msl->external)
                return 0;

        addr = mmap(ms->addr, ms->len, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, wa->fd_hugepage,
                        wa->seg_idx * EAL_PAGE_SIZE);
        if (addr == MAP_FAILED || addr != ms->addr)
                return -1;
        wa->seg_idx++;

        return 0;
}
```