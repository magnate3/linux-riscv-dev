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

#define NPAGES		3
#define MMAP_DEV	"/dev/my_dev"

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
		mmap_addr[i] = 'f';
		mmap_addr[i+1] = 'a';
		mmap_addr[i+2] = 'c';
		mmap_addr[i+3] = 'e';
		mmap_addr[i+4] = 'b';
		mmap_addr[i+5] = 'o';
		mmap_addr[i+6] = 'o';
		mmap_addr[i+7] = 'k';
		mmap_addr[i+8] = '\0';
	}

	return 0;
}

int main(int argc, const char **argv)
{
	int fd;
	unsigned char *addr;
	char *ptr;
	int pagesize = getpagesize();
	int len = NPAGES * pagesize;
	int i;
        int ret = 0;
	fd = open(MMAP_DEV, O_RDWR | O_SYNC);
	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}

	/* TODO: call mmap() system call with R/W permission to address */
	addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        printf("addr: %p \n", addr);
	/* TODO: call test_write_read() */
	test_write_read(fd, addr);
	/* TODO: check the values by module init */
	for(i = 0 ; i < len ; i += getpagesize()){
		printf("0x%02x%02x%02x%02x\n",
				addr[i],
				addr[i+1], 
				addr[i+2], 
				addr[i+3]);
	}
#if 1
	ret = posix_memalign((void **)&ptr, pagesize, pagesize);
	if(!ret)
	{
            memcpy(ptr, "krishna", strlen("krishna"));
	    printf("phy addr of ptr  0x%lx \n",mem_virt2phy(ptr));
	    //read(fd, ptr, pagesize);

	    write(fd, ptr, pagesize);
	    printf("after migrate, phy addr of ptr 0x%lx \n",mem_virt2phy(addr));
	    //read(fd, ptr, pagesize);
#if 1
            memcpy(ptr, "krishna2", strlen("krishna"));
	    printf("after migrate and memecpy again , phy addr of ptr 0x%lx \n", mem_virt2phy(addr));
	    read(fd, ptr, pagesize);
#endif
	    free(ptr);
	}
	else
	{
	    fprintf(stderr, "posix_memalign: %s\n", strerror (ret));
	}
#endif
        munmap(addr, len);

	close(fd);

	return 0;
}
