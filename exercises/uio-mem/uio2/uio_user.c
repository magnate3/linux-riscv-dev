#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>

#define UIO_DEV "/dev/uio0"
#define UIO_ADDR "/sys/class/uio/uio0/maps/map0/addr"
#define UIO_SIZE "/sys/class/uio/uio0/maps/map0/size"

static char uio_addr_buf[16]={0};
static char uio_size_buf[16]={0};

int main(void)
{
    int uio_fd,addr_fd,size_fd;
    int uio_size;
    void *uio_addr, *access_address;
    int n=0;
    uio_fd = open(UIO_DEV,O_RDWR);
    addr_fd = open(UIO_ADDR,O_RDONLY);
    size_fd = open(UIO_SIZE,O_RDONLY);
    if(addr_fd < 0 || size_fd < 0 || uio_fd < 0){
        fprintf(stderr,"mmap:%s\n",strerror(errno));
        exit(-1);
    }

    n=read(addr_fd,uio_addr_buf,sizeof(uio_addr_buf));
    if(n<0){
        fprintf(stderr, "%s\n", strerror(errno));
        exit(-1);
    }
    n=read(size_fd,uio_size_buf,sizeof(uio_size_buf));
    if(n<0){
        fprintf(stderr, "%s\n", strerror(errno));
        exit(-1);
    }
    uio_addr = (void*)strtoul(uio_addr_buf,NULL,0);
    uio_size = (int)strtol(uio_size_buf,NULL,0);

    access_address = mmap(NULL,uio_size,PROT_READ | PROT_WRITE,
                            MAP_SHARED,uio_fd,0);
    if(access_address == (void*)-1){
        fprintf(stderr,"mmap:%s\n",strerror(errno));
        exit(-1);
    }

    printf("The device address %p (lenth %d)\n"
        "can be accessed over\n"
        "logical address %p\n",uio_addr,uio_size,access_address);
/*
    access_address = (void*)(long)mremap(access_address, getpagesize(),uio_size + getpagesize()+ 11111, MAP_SHARED);
    if(access_address == (void*)-1){
        fprintf(stderr,"mremap: %s\n",strerror(errno));
        exit(-1);
    }
    printf(">>>AFTER REMAP:""logical address %p\n",access_address);
*/
    return 0;
}