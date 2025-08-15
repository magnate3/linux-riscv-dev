#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

int
main()
{
        int fd;
        int pagesize = getpagesize();
        char *ptr;
        printf("page size is %d \n", pagesize);
        fd = open("/dev/Sample", O_RDWR);
        if (fd < 0) {
                perror("error");
        }
        posix_memalign((void **)&ptr, 4096, 4096);
        memcpy(ptr, "krishna", strlen("krishna"));  //Write String to Driver
        write(fd, ptr, 4096);
        printf("data is %s\n", ptr);   //Read Data from Driver
        close(fd);
}
