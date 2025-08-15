#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include<sys/ioctl.h>
 
#define IOCTL_PA_ALLOC _IOWR('p', 0, size_t *)
#define IOCTL_PA_FREE _IOR('p', 1, void *)
 
int main()
{
        int fd;
	int ret;
        unsigned long size;
	printf("\nOpening Driver\n");
        fd = open("/dev/physmem", O_RDWR);
        if(fd < 0) {
                printf("Cannot open device file...\n");
                return 0;
        }

        size = 4096;
	printf("Informing driver to allocate\n");
        ret = ioctl(fd, IOCTL_PA_ALLOC, (size_t*) &size); 
	printf("ioctl ret: %x\n", ret);
	printf("addr: %p\n", (void *) size);

        printf("Informing driver to free allocation\n");
        ret = ioctl(fd, IOCTL_PA_FREE, (void *) &size);
 
	printf("ioctl ret: %x\n", ret);
        printf("Closing Driver\n");
        close(fd);
}

