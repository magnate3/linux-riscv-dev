/*
 * SCE394 - Memory Mapping Lab (#14)
 *
 * memory mapping between user-space and kernel-space
 *
 * test case
 */

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include"simple.h"

#define NPAGES		1
#define MMAP_DEV	"/dev/simpler"

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
/* PAGE_SIZE = getpagesize() */

int test_write_read(int fd, unsigned char *mmap_addr)
{
	int i;
	printf("\nWrite/Read test ...\n");

	/* TODO: write to device mmap'ed address */
	for(i = 0 ; i < NPAGES * getpagesize() ; i += getpagesize()){
		mmap_addr[i] = 'h';
		mmap_addr[i+1] = 'l';
		mmap_addr[i+2] = 'l';
		mmap_addr[i+3] = 'o';
		mmap_addr[i+4] = ' ';
		mmap_addr[i+5] = 'w';
		mmap_addr[i+6] = 'o';
		mmap_addr[i+7] = 'o';
		mmap_addr[i+7] = 'l';
		mmap_addr[i+8] = '\0';
	}

	return 0;
}

int main(int argc, const char **argv)
{
	int fd;
	unsigned char *src, *dst;
	int pagesize = getpagesize();
	int len = NPAGES*pagesize;
        char *str = "hello world"; 
        char buf[64] = {0};
        struct address addr;
        unsigned long  dst2;
	fd = open(MMAP_DEV, O_RDWR | O_SYNC);
	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}

        src = (char*)mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0); 
	if(NULL == src)
	{
            printf("src: is NULL  \n");
	    goto err1;
	}
        memcpy(src, str, strlen(str) +1); 
        dst  = (char*)mmap(NULL, len, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0); 
	if(NULL == dst)
	{
            printf("dst: is NULL  \n");
	    goto err2;
	}
       dst2 = (unsigned long)dst + 2*pagesize;
       addr.from_addr = (unsigned long)src; 
       addr.to_addr = dst2; 
       munmap((void *)dst, len); //making a hole at dest address
       if(ioctl(fd, IOCTL_MVE_VMA_TO, &addr) < 0){
           printf("Testcase Failed\n");
           goto err2;
       }
       memcpy(buf, (char*)dst2, strlen(str) +1); 
       printf("buf is %s \n", (char*)dst2); 
err3:
	//free(dst);
        munmap((char*)dst2, len);
err2:
	//free(src);

        munmap(src, len);

err1:
	close(fd);

	return 0;
}
