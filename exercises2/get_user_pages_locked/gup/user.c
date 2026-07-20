#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

const char *init_str = "INIT_DATA";
int
main()
{
        int fd;
        char *ptr;
        fd = open("/dev/sample", O_RDWR);
        if (fd < 0) {
                perror("error");
        }
        // Alloc a aligned page
        // Device driver will mapp this page, and write to it
        posix_memalign((void **)&ptr, getpagesize(), getpagesize());

        // Init the page with data
        memcpy(ptr, init_str, strlen(init_str));

        // data count is not relevant
        read(fd, ptr, 0);
        printf("Data after read is \"%s\"\n", ptr);   //Read Data from Driver

        write(fd, ptr, 0);
        printf("Data after write is \"%s\"\n", ptr);   //Read Data from Driver

        // Finalize
        close(fd);
        free(ptr);
        return 0;
}
