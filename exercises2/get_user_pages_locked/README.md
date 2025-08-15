
```
long get_user_pages(unsigned long start, unsigned long nr_pages,
                unsigned int gup_flags, struct page **pages,
                struct vm_area_struct **vmas)
{
        int locked = 1;

        if (!is_valid_gup_args(pages, vmas, NULL, &gup_flags, FOLL_TOUCH))
                return -EINVAL;

        return __get_user_pages_locked(current->mm, start, nr_pages, pages,
                                       vmas, &locked, gup_flags);
}
EXPORT_SYMBOL(get_user_pages);
```

# get_user_pages_locked
```
long get_user_pages_locked(unsigned long start, unsigned long nr_pages,
                           unsigned int gup_flags, struct page **pages,
                           int *locked)
{
        /*
         * FIXME: Current FOLL_LONGTERM behavior is incompatible with
         * FAULT_FLAG_ALLOW_RETRY because of the FS DAX check requirement on
         * vmas.  As there are no users of this flag in this call we simply
         * disallow this option for now.
         */
        if (WARN_ON_ONCE(gup_flags & FOLL_LONGTERM))
                return -EINVAL;
        /*
         * FOLL_PIN must only be set internally by the pin_user_pages*() APIs,
         * never directly by the caller, so enforce that:
         */
        if (WARN_ON_ONCE(gup_flags & FOLL_PIN))
                return -EINVAL;

        return __get_user_pages_locked(current->mm, start, nr_pages,
                                       pages, NULL, locked,
                                       gup_flags | FOLL_TOUCH);
}
EXPORT_SYMBOL(get_user_pages_locked);
```

```
int nvidia_p2p_get_pages(u64 vaddr, u64 size,
		struct nvidia_p2p_page_table **page_table,
		void (*free_callback)(void *data), void *data)
{
    down_read(&current->mm->mmap_sem);
	locked = 1;
	user_pages = get_user_pages_locked(vaddr & PAGE_MASK, nr_pages,
			FOLL_WRITE | FOLL_FORCE,
			pages, &locked);
	up_read(&current->mm->mmap_sem);
	if (user_pages != nr_pages) {
		ret = user_pages < 0 ? user_pages : -ENOMEM;
		goto free_pages;
	}

}
```


# test1

+ 1  用户态程序   
```
int
main()
{
        int fd;
        int pagesize = getpagesize();
        char *ptr;
        printf("page size is %d \n", pagesize);
        fd = open("/dev/Sample", O_RDWR);
        if (fd < 0) {
                perror("error");
        }
        posix_memalign((void **)&ptr, pagesize, 4096);
#if 0
        memcpy(ptr, "krishna", strlen("krishna"));  //Write String to Driver
        printf("phy addr 0x%lx \n",mem_virt2phy(ptr));
#endif
        write(fd, ptr, 4096);
        printf("data is %s\n", ptr);   //Read Data from Driver
        free(ptr);
        close(fd);
}
```

+ 2 内核


```
#if 1
       user_pages = pin_user_pages(vaddr & PAGE_MASK, nr_pages, flags, pages, NULL);
#else
       locked = 1;
       user_pages = get_user_pages_locked(vaddr , nr_pages, flags,
                        pages, &locked);
#endif
```

+ 3 执行  
```
root@ubuntux86:# insmod  get_user_pages_test.ko 
root@ubuntux86:# ./test 
page size is 4096 
data is Mohan from kernel
root@ubuntux86:# dmesg | tail -n 10
[ 4107.295794] sample_open
[ 4107.295809] sample_write
[ 4107.295820] get_user_pages_locked -22 pages,need 1 pages 
[ 4107.295837] sample_release
[ 4161.576530] sample_open
[ 4161.576547] sample_write
[ 4161.576555] Got mmaped.
[ 4161.576556] kernel phy addr 10eece000

[ 4161.576574] sample_release
root@ubuntux86:# 
```