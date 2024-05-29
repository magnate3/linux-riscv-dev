


#  get_user_pages 之前分配内存


```
 cat user_test.c 
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

int
main()
{
        int fd;
        char *ptr;
        int pagesize = getpagesize();
        fd = open("/dev/Sample", O_RDWR);
        if (fd < 0) {
                perror("error");
        }
        posix_memalign((void **)&ptr,pagesize, 4096);
        memcpy(ptr, "krishna", strlen("krishna"));  //Write String to Driver
        write(fd, ptr, 4096);
        printf("data is %s\n", ptr);   //Read Data from Driver
        close(fd);
}
```

```
[root@centos7 get_user_pages_remote]# ./user_test 
data is Mohan
```


```
[596780.431385] sample_release
[596849.446102] sample_open
[596849.448775] sample_write
[596849.451389] Got mmaped.
[596849.453913] krishna
[596849.456136] sample_release
```

# #  get_user_pages 之前 没有分配内存


```
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

int
main()
{
        int fd;
        char *ptr;
        int pagesize = getpagesize();
        fd = open("/dev/Sample", O_RDWR);
        if (fd < 0) {
                perror("error");
        }
        posix_memalign((void **)&ptr,pagesize, 4096);
        //memcpy(ptr, "krishna", strlen("krishna"));  //Write String to Driver
        write(fd, ptr, 4096);
        printf("data is %s\n", ptr);   //Read Data from Driver
        close(fd);
}
```

```
[root@centos7 get_user_pages_remote]# gcc user_test.c  -o user_test
[root@centos7 get_user_pages_remote]# ./user_test 
data is Mohan
```


```
[597012.816025] sample_open
[597012.818649] sample_write
[597012.821262] Got mmaped.

[597012.825466] sample_release
```

# vfio_iommu_type1.c

```
static int vaddr_get_pfn(struct mm_struct *mm, unsigned long vaddr,
			 int prot, unsigned long *pfn)
{
	struct page *page[1];
	struct vm_area_struct *vma;
	struct vm_area_struct *vmas[1];
	unsigned int flags = 0;
	int ret;

	if (prot & IOMMU_WRITE)
		flags |= FOLL_WRITE;

	down_read(&mm->mmap_sem);
	if (mm == current->mm) {
		ret = get_user_pages_longterm(vaddr, 1, flags, page, vmas);
	} else {
		ret = get_user_pages_remote(NULL, mm, vaddr, 1, flags, page,
					    vmas, NULL);
		/*
		 * The lifetime of a vaddr_get_pfn() page pin is
		 * userspace-controlled. In the fs-dax case this could
		 * lead to indefinite stalls in filesystem operations.
		 * Disallow attempts to pin fs-dax pages via this
		 * interface.
		 */
		if (ret > 0 && vma_is_fsdax(vmas[0])) {
			ret = -EOPNOTSUPP;
			put_page(page[0]);
		}
	}
	up_read(&mm->mmap_sem);

	if (ret == 1) {
		*pfn = page_to_pfn(page[0]);
		return 0;
	}

	down_read(&mm->mmap_sem);

retry:
	vma = find_vma_intersection(mm, vaddr, vaddr + 1);

	if (vma && vma->vm_flags & VM_PFNMAP) {
		ret = follow_fault_pfn(vma, mm, vaddr, pfn, prot & IOMMU_WRITE);
		if (ret == -EAGAIN)
			goto retry;

		if (!ret && !is_invalid_reserved_pfn(*pfn))
			ret = -EFAULT;
	}

	up_read(&mm->mmap_sem);
	return ret;
}

```