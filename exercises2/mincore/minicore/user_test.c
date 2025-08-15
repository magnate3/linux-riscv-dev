#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <errno.h>
#define PAGE_SIZE (64*1024)
#define FILE_SIZE  (PAGE_SIZE*2)
int main()
{
        int fd;
        int pagesize = getpagesize();
        char *  name  = "./rand_data.bin";
        uint8_t v;
        int ret = 0;
        size_t i = 0;
        printf("page size is %d \n", pagesize);
        if (access(name, F_OK) == 0)
        {
            printf("%s exists.\n",name);
        }
        else
        {
            printf("%s not exists.\n",name);
            return -1;
        }
        fd = open("/dev/Sample", O_RDWR);
        if (fd < 0) {
                perror("error");
                return -1;
        }
        write(fd, name, strlen(name));
        for (i = 0; i < FILE_SIZE; i += PAGE_SIZE) {
           ret = pread(fd, &v, 1, i);
           printf("pagecahe exist %u \n", v);
        }
        close(fd);
        return 0;
}
