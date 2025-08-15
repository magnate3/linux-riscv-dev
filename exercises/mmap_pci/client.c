#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#define E1000_FLASH_BASE_ADDR 0xE000 
unsigned char* iomem;

void die(const char* msg)
{
	perror(msg);
	exit(-1);
}

void iowrite(uint64_t addr, uint64_t value)
{
	*((uint64_t*)(iomem + addr)) = value;
}

uint64_t ioread(uint64_t addr)
{
	return *((uint64_t*)(iomem + addr));
}

void iowrite32(uint64_t addr, uint32_t value)
{
	*((uint32_t*)(iomem + addr)) = value;
}

uint32_t ioread32(uint64_t addr)
{
	return *((uint32_t*)(iomem + addr));
}

int main(int argc, char *argv[])
{
	// Open and map I/O memory
	int fd = open("/sys/devices/pci0000:00/0000:00:0c.0/0000:03:00.0/0000:04:00.0/0000:05:00.0/resource0", O_RDWR | O_SYNC);
	if (fd == -1)
		die("open");

	iomem = mmap(0, 0x10000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (iomem == MAP_FAILED)
		die("mmap");

	printf("iomem @ %p\n", iomem);
    // Do something
	return 0;
}
