/*
 * SCE394 - Memory Mapping Lab (#14)
 *
 * memory mapping between user-space and kernel-space
 *
 * test case
 */

#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#define NPAGES		16
#define MMAP_DEV	"/dev/mymmap"

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
	/* TODO: call mymap_read() */
	read(fd, NULL, 0);

	return 0;
}

int main(int argc, const char **argv)
{
	int fd;
	unsigned char *addr;
	int len = NPAGES * getpagesize();
	int i;

	fd = open(MMAP_DEV, O_RDWR | O_SYNC);
	if (fd < 0) {
		perror("open");
		exit(EXIT_FAILURE);
	}

	/* TODO: call mmap() system call with R/W permission to address */
	addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	/* TODO: check the values by module init */
	for(i = 0 ; i < len ; i += getpagesize()){
		printf("0x%02x%02x%02x%02x\n",
				addr[i],
				addr[i+1], 
				addr[i+2], 
				addr[i+3]);
	}
	/* TODO: call test_write_read() */
	test_write_read(fd, addr);


	close(fd);

	return 0;
}
