#include <stdio.h>
#include <sys/mman.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

int physmem_fd;
int devmem_fd;

void hexdump(void *data, int size) {
	
	for (int i = 0; i < size; i++) {
		printf("%02x ", *(unsigned char *)(data + i));

		if ((i+1) % 8 == 0 && i != 0) {
			printf("  ");
		}
		if ((i+1) % 16 == 0 && i != 0 ) {
			printf("\n");
		}
	}

}

void map_physmem(off_t offset) {

	void *physmem_addr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, physmem_fd, offset);
	void  *devmem_addr = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, devmem_fd, offset);

	if (physmem_addr != MAP_FAILED) {
		printf("/dev/physmem offset: %lx\n", offset);
		hexdump(physmem_addr, 64);
	} else {
		perror("/dev/physmem mmap");
	}

	if (devmem_addr != MAP_FAILED) {
		printf("/dev/mem offset: %lx\n", offset);
		hexdump(devmem_addr, 64);
	} else {
		perror("/dev/mem mmap");
	}
}


int main(int argc, char *argv[]) {
	physmem_fd = open("/dev/physmem", O_RDWR);
	devmem_fd = open("/dev/mem", O_RDWR);

	if (devmem_fd == -1) {
		perror("open /dev/mem");
		exit(-1);
	}
	
	if (physmem_fd == -1) {
		perror("open /dev/physmem");
		exit(-1);
	}

	printf("Trying at offset 0x00\n\n");

        map_physmem(0x00);	

	printf("\nTrying beyond 1mb...\n\n");
        
	map_physmem(0x890000000);	
	//map_physmem(0x100000);	
	
	close(physmem_fd);
	close(devmem_fd);

	return 0;
}
