/*
 * Example of using hugepage memory in a user application using the mmap
 * system call with MAP_HUGETLB flag.  Before running this program make
 * sure the administrator has allocated enough default sized huge pages
 * to cover the 256 MB allocation.
 *
 * For ia64 architecture, Linux kernel reserves Region number 4 for hugepages.
 * That means the addresses starting with 0x800000... will need to be
 * specified.  Specifying a fixed address is not required on ppc64, i386
 * or x86_64.
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#define LENGTH (6UL*1024*1024*1024)
#define PROTECTION (PROT_READ | PROT_WRITE)

#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40
#endif

/* Only ia64 requires this */
#ifdef __ia64__
#define ADDR (void *)(0x8000000000000000UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_FIXED)
#else
#define ADDR (void *)(0x0UL)
#define FLAGS (MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB)
#endif

void check_bytes(char *addr)
{
	printf("First hex is %x\n", *((unsigned int *)addr));
}

void write_bytes(char *addr)
{
	unsigned long i;

	for (i = 0; i < LENGTH; i++)
		*(addr + i) = (char)i;
}

void read_bytes(char *addr)
{
	unsigned long i;

	check_bytes(addr);
	for (i = 0; i < LENGTH; i++)
		if (*(addr + i) != (char)i) {
			printf("Mismatch at %lu\n", i);
			break;
		}
}
void * mmap_test()
{
        void * addr;
	addr = mmap(ADDR, LENGTH, PROTECTION, FLAGS, 0, 0);
	if (addr == MAP_FAILED) {
		perror("mmap");
                return NULL;
	}

	printf("Returned address is %p\n", addr);
	check_bytes(addr);
	write_bytes(addr);
	read_bytes(addr);

       return addr;
}
int main(void)
{
	void *addr1, *addr2;
        int stack = 99;
        addr1 = mmap_test();
        addr2 = mmap_test();
        printf("munmap addr1 %p, addr2 %p , stack %p \n", addr1, addr2, &stack);
        printf("input char \n");
        getchar();
        getchar();
        if(addr1)
        {
           munmap(addr1, LENGTH);
        }
        if(addr2)
        {
           munmap(addr2, LENGTH);
        }
	return 0;
}
