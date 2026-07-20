#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#define BAD_PHYS_ADDR 0
#define PFN_MASK_SIZE 8
/* PAGE_SIZE = getpagesize() */
uint64_t mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long long virt_pfn;	// virtual page frame number
	int page_size;
	off_t offset;

	/* standard page size */
	page_size = getpagesize();

	fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0) {
		fprintf(stderr, "%s(): cannot open /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	}

	virt_pfn = (unsigned long)virtaddr / page_size;
	//printf("Virtual page frame number is %llu\n", virt_pfn);

	offset = sizeof(uint64_t) * virt_pfn;
	if (lseek(fd, offset, SEEK_SET) == (off_t) - 1) {
		fprintf(stderr, "%s(): seek error in /proc/self/pagemap: %s\n", __func__, strerror(errno));
		close(fd);
		return BAD_PHYS_ADDR;
	}

	retval = read(fd, &page, PFN_MASK_SIZE);
	close(fd);
	if (retval < 0) {
		fprintf(stderr, "%s(): cannot read /proc/self/pagemap: %s\n", __func__, strerror(errno));
		return BAD_PHYS_ADDR;
	} else if (retval != PFN_MASK_SIZE) {
		fprintf(stderr, "%s(): read %d bytes from /proc/self/pagemap but expected %d\n", 
		        __func__, retval, PFN_MASK_SIZE);
		return BAD_PHYS_ADDR;
	}

	/*
	 * the pfn (page frame number) are bits 0-54 (see
	 * pagemap.txt in linux Documentation)
	 */
	if ((page & 0x7fffffffffffffULL) == 0) {
		fprintf(stderr, "Zero page frame number\n");
		return BAD_PHYS_ADDR;
	}

	physaddr = ((page & 0x7fffffffffffffULL) * page_size) + ((unsigned long)virtaddr % page_size);

	return physaddr;
}
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
