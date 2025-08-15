#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#define  MAP_SIZE	(4096)
#define  PATH		"/dev/test"

int main()
{
	void *addr;
	int fd;

	/* open */
	fd = open(PATH, O_RDWR);
	if (fd < 0) {
		printf("ERROR: open %s failed.\n", PATH);
		return -1;
	}

	/* mmap */
	addr = mmap(NULL, MAP_SIZE,
			PROT_READ | PROT_WRITE,
			MAP_SHARED,
			fd, 
			0);
	if (!addr) {
		printf("ERROR: mmap failed.\n");
		close(fd);
		return -1;
	}

	/* use */
	*(char *)addr = 'B';
	printf("%#lx => %c\n", (unsigned long)addr, *(char *)addr);

	/* unmap */
	munmap(addr, MAP_SIZE);
	close(fd);

	return 0;
}
