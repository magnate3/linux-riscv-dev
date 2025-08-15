#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>

#define HUGEPAGE_FILE "/dev/hugepages1G/random"
#define LENGTH (1024UL * 1024 * 1)   // 1M
#define PROTECTION (PROT_READ | PROT_WRITE)
#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_SHARED | MAP_POPULATE)
#define BAD_PHYS_ADDR 0
#define PFN_MASK_SIZE 8

static void write_byt(char *addr, char c);
static void print_byt(char *addr);

// Get physical address of any mapped virtual address in the current process
uint64_t mem_virt2phy(const void *virtaddr);

int main()
{
	void *addr;
	int hugepage_fd, ret;
	uint64_t paddr;

	hugepage_fd = open(HUGEPAGE_FILE, O_CREAT | O_RDWR, 0755);
	if (hugepage_fd < 0) {
		perror("Fail to open hugepage memory file");
		exit(1);
	}

	addr = mmap(ADDR, LENGTH, PROTECTION, FLAGS, hugepage_fd, 0);
	if (addr == MAP_FAILED) {
		perror("mmap");
		unlink(HUGEPAGE_FILE);
		exit(1);
	} 

	paddr = mem_virt2phy(addr);
	printf("Virtual address is %p\n", addr);
	printf("Physical address is %llu\n", paddr);

	munmap(addr, LENGTH);
	close(hugepage_fd);
	unlink(HUGEPAGE_FILE);

	return 0;
}

static void write_byt(char *addr, char c)
{	
	if (addr) {
		*addr = c;
	} else {
		fprintf(stderr, "%s(): empty pointer\n", __func__);		
	}	
}

static void print_byt(char *addr)
{
	if (addr) {
		printf("%d\n", (int)(*addr));
	} else {
		fprintf(stderr, "%s(): empty pointer\n", __func__);	
	}
}

uint64_t mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long virt_pfn;	// virtual page frame number
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
		fprintf(stderr, "%s(): read %d bytes from /proc/self/pagemap but expected %d:\n", 
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
