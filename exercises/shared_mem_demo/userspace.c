#include <unistd.h>  
#include <stdio.h>  
#include <stdlib.h>  
#include <string.h>  
#include <fcntl.h>  
#include <linux/fb.h>  
#include <sys/mman.h>
#include <errno.h>  
#include <sys/ioctl.h>  

#define PAGE_SIZE 4096 

int main(void)
{
    unsigned char *p_map;
    int i;
    int fd = open("/dev/evt_map", O_RDWR|O_SYNC);
    if (fd < 0) {
        printf("Fail to open %s. Error:%s\n", "/dev/evt_map", strerror(errno));
        exit(-1);
    }
    p_map = (unsigned char *)mmap(NULL, PAGE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED,fd, 0);  
    if(p_map == MAP_FAILED)  
    {  
        printf("mmap fail\n");  
        goto here;  
    }  
    for(i=0;i<256;i++)  
        printf("%c",p_map[i]);  
    printf("\n");
  
here:  
    munmap(p_map, PAGE_SIZE);  
    close(fd);
    return 0;  
}