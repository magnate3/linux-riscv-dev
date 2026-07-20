#include<stdio.h>
#include<stdlib.h>
#include <unistd.h>

#include<string.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <sys/mman.h>
int main()
{
    const char *FileName="/dev/sharemem";
    int flag=O_RDWR;
    int fd=open(FileName,flag);
    if(fd == -1){
        printf("Open Error\n");
        return fd;
    }
    


    size_t length=sysconf(_SC_PAGE_SIZE);
    int* addr = mmap(NULL,length , PROT_READ|PROT_WRITE,MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        printf("mmap Failed\n");
        return -1;
    }
    printf("addr[%p]=0x%x\n",addr,addr[0]);
    addr[0]=0x12345678;
    printf("addr[%p]=0x%x\n",addr,addr[0]);
    
    char buf[100];
    memset(buf,0,sizeof(buf));

    int ret=read(fd,buf,sizeof(buf));
    if(ret == -1){
        printf("Read Error\n");
        return ret;
    }else{
        printf("%s\n",buf);
    }
    printf("addr[%p]=0x%x\n",addr,addr[0]);
}
