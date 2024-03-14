/*
 * Hugetlb: Shared File hugepage for migration
 *
 * 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <stdint.h>
/* NUMA header with -lnuma */
#include <numa.h>
#include <numaif.h>
/* errno */
#include <errno.h>
#define BAD_PHYS_ADDR 0
#define PFN_MASK_SIZE 8

#define NUMA_NODE0	0
#define NUMA_NODE1	1
#define NUMA_AREA	2

#define HPAGE_SIZE		(512UL * 1024 * 1024)
#define HPAGE_MASK		(~(HPAGE_SIZE - 1))
#define TEST_MAP_SIZE	(2 * HPAGE_SIZE)
#define TEST_HUGEPAGE_PATH	"/mnt/huge/test"
//#define TEST_HUGEPAGE_PATH	"/dev/hugepages"
//gcc migrate_test.c  -o migrate_test -lnuma
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
int main()
{
	struct bitmask *old_nodes;
	struct bitmask *new_nodes;
	void *addr[NUMA_AREA];
	char *page_base;
	int nr_nodes;
	char *pages;
	int status;
	int nodes;
	int rc;
	int fd;
        uint64_t paddr;
        const char * str = "hello";
	/* NUMA NODE */
	nr_nodes = numa_max_node() + 1;

	/* NODEMASK alloc */
	old_nodes = numa_bitmask_alloc(nr_nodes);
	new_nodes = numa_bitmask_alloc(nr_nodes);

	if (nr_nodes < 2) {
		printf("Request minimum of 2 nodes!\n");
		rc = -EINVAL;
		goto err_node;
	}

	/* Open hugepge file */
	fd = open(TEST_HUGEPAGE_PATH, O_RDWR | O_CREAT);
	if (fd < 0) {
		printf("ERROR: failed open %s\n", TEST_HUGEPAGE_PATH);
		return -EINVAL;
	}

#if 1
	page_base = (char *)mmap(NULL,
				TEST_MAP_SIZE,
				PROT_READ | PROT_WRITE,
				MAP_SHARED,
				fd,
				0);
#else
	page_base = (char *)mmap(NULL,
				TEST_MAP_SIZE,
				PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS,
				-1,
				0);
#endif
	if (page_base == MAP_FAILED) {
		printf("ERROR: mmap failed.\n");
		return -ENOMEM;
	}

        memcpy(page_base,str, strlen(str));
	/* Page Alignment: Bound to HugePage */
	pages = (void *)((((long)page_base) & HPAGE_MASK) + HPAGE_SIZE);

	/* Prepare: Bind to Physical Page */
	pages[0] = 0;
	addr[0] = pages;

	/* Verify correct startup locations */
	printf("******** Before Migration\n");
        paddr = mem_virt2phy(page_base);
        printf("before migrate, Physical address is %llu\n", (unsigned long long)paddr);
	numa_move_pages(0, 1, addr, NULL, &status, 0);
	printf("  Page vaddr: %p node: %d\n", pages, status);

	/* Move to another NUMA NODE */
	nodes = status == NUMA_NODE0 ? NUMA_NODE1 : NUMA_NODE0;
	numa_bitmask_setbit(old_nodes, status);
	numa_bitmask_setbit(new_nodes, nodes);
	status = nodes;

	/* Move to another NUMA NODE */
	numa_move_pages(0, 1, addr, &nodes, &status, 0);

	/* Migration */
	printf("Migrating the current processes pages ...\n");
	rc = numa_migrate_pages(0, old_nodes, new_nodes);
	if (rc < 0)
		printf("ERROR: numa_migrate_pages failed\n");

	/* Get page state after migration */
	numa_move_pages(0, 1, addr, NULL, &status, 0);
	printf("  Page vaddr: %p node: %d\n", pages, status);
	
	/* sleep just for debug */
	sleep(2);
        memcpy(page_base,str, strlen(str));
        paddr = mem_virt2phy(page_base);
        printf("after migrate, Physical address is %llu\n", (unsigned long long)paddr);
	/* unmap */
	munmap(page_base, TEST_MAP_SIZE);
	close(fd);
err_node:
	numa_bitmask_free(new_nodes);
	numa_bitmask_free(old_nodes);
	return rc;
}
