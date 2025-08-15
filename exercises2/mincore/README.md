
# vmtouch 

[vmtouch ](https://hoytech.com/vmtouch/)  

# pagecache的作用

下面的例子说明pagecache的效果   

先用dd产生一个100MB的rand_data.bin   
dd if=/dev/random of=./rand_data.bin bs=1048576 count=100   
把pagecache不需要的部分清空,确保rand_data.bin不在pagecache里面    
```
echo 3 >/proc/sys/vm/drop_caches
```
第一次copy,时间是0.807s
```
time cp -f ./rand_data.bin /tmp/

real	0m0.807s
user	0m0.000s
sys	0m0.229s
```
4.第二次copy,经过前一次copy,这一次rand_data.bin已经在pagecache里面了.

读rand_data.bin的时候会直接从pagecache读，而不需要重新从外设存储设备读，所以速度会快很多.
```
time cp -f rand_data.bin /tmp/

real	0m0.362s
user	0m0.013s
sys	0m0.248s
```

# 如何知道某个文件是否在pagecache中


```
[root@centos7 kernel]# du -sh  twotermex.zip
99M     twotermex.zip
```

+ 1 没有读取twotermex.zip的时候,pagecache里面没有twotermex.zip
```
[root@centos7 kernel]# vmtouch  twotermex.zip
           Files: 1
     Directories: 0
  Resident Pages: 0/1574  0/98M  0%
         Elapsed: 8.7e-05 seconds
[root@centos7 kernel]# 
``` 

+ 2.读取twotermex.zip之后再来检测，通过Resident Pages:这一栏看，基本上都已经在pagecache里面了.    

```
[root@centos7 kernel]# cp -f twotermex.zip /tmp/
[root@centos7 kernel]# vmtouch  twotermex.zip
           Files: 1
     Directories: 0
  Resident Pages: 1574/1574  98M/98M  100%
         Elapsed: 0.000247 seconds
[root@centos7 kernel]# 
```
+ 3  vmtouch还可以看指定的目录有多少文件进pagecache了.    

```
[root@centos7 kernel]# vmtouch hinic-4.19/ 2>/dev/null 
           Files: 72
     Directories: 2
  Resident Pages: 0/208  0/13M  0%
         Elapsed: 0.000518 seconds
[root@centos7 kernel]# 
```

# vmtouch的原理
vmtouch检测某个文件是否在pagecache里面主要分如下步骤:    

open file
```
  fd = open(path, open_flags, 0);
```
mmap(MAP_SHARED) file,获得指向该文件的共享内存地址    
```
  mem = mmap(NULL, len_of_range, PROT_READ, MAP_SHARED, fd, offset);
```
使用fadvise来告知kernel不要主动把文件放进pagecache    
```
posix_fadvise(fd, offset, len_of_range, POSIX_FADV_DONTNEED)
```
使用mincore来检测该共享地址是否在内存中(pagecache)
```
mincore(mem, len_of_range, (void*)mincore_array)
```
mincore这个API是核心，它的用法如下:    
mincore - determine whether pages are resident in memory     


```
#include <unistd.h>
#include <sys/mman.h>

int mincore(void *addr, size_t length, unsigned char *vec);
```





mincore()返回起始地址为addr长度为length字节的虚拟地址访问中分页的内存驻留信息。addr 中的地址必须是分页对齐的，并且由于返回的信息是有关整个分页的，因此length 实际上会被向上舍入到系统分页大小的下一个整数倍。    

内存驻留相关的信息会通过 vec 返回，它是一个数组，其大小为(length + PAGE_SIZE – 1) / PAGE_SIZE 字节。每个字节的最低有效位在相应分页驻留在内存中时会被设置，而其他位的设置在一些 UNIX 实现上是未定义的，因此可移植的应用程序应该只测试最低有效位。    

mincore()返回的信息在执行调用的时刻与检查 vec 中的元素的时刻期间可能会发生变化。`唯一能够确保保持驻留在内存中的分页是那些通过 mlock()或 mlockall()锁住的分页`。    

下面演示了如何使用 mlock()和 mincore()。这个程序首先分配并使用 mmap()
映射了一块内存区域，然后以固定的时间间隔使用 mlock()将整个区域或一组分页锁进内存。（传给这个程序的所有命令行参数的单位是分页，程序会将这些参数转换成字节，因为 mmap()、mlock()以及 mincore()使用的是字节。）在调用 mlock()之前和之后，程序使用mincore()来获取这个区域中分页的内存驻留信息并图形化地将这些信息展现了出来。   
> ## test2

下面中分配了 32 个分页，每组为 8 个分页，并给三个连续分页加锁。
```
[root@centos7 minicore]# ./minicore_test  32  8 3
Allocated 2097152 (0x200000) bytes starting at 0xffffb88e0000
Before mlock:
0xffffb88e0000: ................................
After mlock:
0xffffb88e0000: ***.....***.....***.....***.....
[root@centos7 minicore]# 
``` 

在程序输出中，点表示分页不在内存中，星号表示分页驻留在内存中。从最后一行输出中可以看出，每组 8 个分页中有 3 个分页是驻留在内存中的。


> ## do_generic_file_read


+ 高版本 generic_file_read_iter  -->    generic_file_buffered_read(
```
static void do_generic_file_read(struct file *filp, loff_t *ppos,
		read_descriptor_t *desc, read_actor_t actor)
{
	struct address_space *mapping = filp->f_mapping;
	struct inode *inode = mapping->host;
	struct file_ra_state *ra = &filp->f_ra;
	pgoff_t index;
	pgoff_t last_index;
	pgoff_t prev_index;
	unsigned long offset;      /* offset into pagecache page */
	unsigned int prev_offset;
	int error;

	index = *ppos >> PAGE_CACHE_SHIFT;
	prev_index = ra->prev_pos >> PAGE_CACHE_SHIFT;
	prev_offset = ra->prev_pos & (PAGE_CACHE_SIZE-1);
	last_index = (*ppos + desc->count + PAGE_CACHE_SIZE-1) >> PAGE_CACHE_SHIFT;
	offset = *ppos & ~PAGE_CACHE_MASK;

	for (;;) {
		struct page *page;
		pgoff_t end_index;
		loff_t isize;
		unsigned long nr, ret;

		cond_resched();
find_page:
		page = find_get_page(mapping, index);
```

> ### generic_file_buffered_read 
```
static ssize_t generic_file_buffered_read(struct kiocb *iocb,
                struct iov_iter *iter, ssize_t written)
{
        struct file *filp = iocb->ki_filp;
        struct address_space *mapping = filp->f_mapping;
        struct inode *inode = mapping->host;
        struct file_ra_state *ra = &filp->f_ra;
        loff_t *ppos = &iocb->ki_pos;
        pgoff_t index;
        pgoff_t last_index;
        pgoff_t prev_index;
        unsigned long offset;      /* offset into pagecache page */
        unsigned int prev_offset;
        int error = 0;

        if (unlikely(*ppos >= inode->i_sb->s_maxbytes))
                return 0;
        iov_iter_truncate(iter, inode->i_sb->s_maxbytes);

        index = *ppos >> PAGE_SHIFT;
        prev_index = ra->prev_pos >> PAGE_SHIFT;
        prev_offset = ra->prev_pos & (PAGE_SIZE-1);
        last_index = (*ppos + iter->count + PAGE_SIZE-1) >> PAGE_SHIFT;
        offset = *ppos & ~PAGE_MASK;

        for (;;) {
                struct page *page;
                pgoff_t end_index;
                loff_t isize;
                unsigned long nr, ret;

                cond_resched();
find_page:
                if (fatal_signal_pending(current)) {
                        error = -EINTR;
                        goto out;
                }

                page = find_get_page(mapping, index);
```


> ## test3


```
gcc user_test.c  -o user_test
```





删除rand_data.bin，重新建立    
```
[root@centos7 minicore]# rm rand_data.bin 
rm: remove regular file ‘rand_data.bin’? y
[root@centos7 minicore]# dd if=/dev/zero of=./rand_data.bin bs=1k count=128 oflag=direct
128+0 records in
128+0 records out
131072 bytes (131 kB) copied, 0.00465015 s, 28.2 MB/s
[root@centos7 minicore]# gcc user_test.c  -o user_test
[root@centos7 minicore]# ./user_test 
page size is 65536 
./rand_data.bin exists.
pagecahe exist 0 
pagecahe exist 0 
[root@centos7 minicore]# 
```

```
[root@centos7 minicore]# vmtouch rand_data.bin 
           Files: 1
     Directories: 0
  Resident Pages: 0/2  0/128K  0%
         Elapsed: 4.7e-05 seconds
[root@centos7 minicore]# 
```

执行cp -f ./rand_data.bin /tmp/后      

```
[root@centos7 minicore]# ./user_test 
page size is 65536 
./rand_data.bin exists.
pagecahe exist 1 
pagecahe exist 1 
[root@centos7 minicore]# dmesg | tail -n 3
[ 4851.008709] sample_open
[ 4851.011160] pagecache index 0 , value 1 , page @ffff7fe817f99fc0 
[ 4851.017244] pagecache index 1 , value 1 , page @ffff7fe817e64dc0 
[root@centos7 minicore]
```