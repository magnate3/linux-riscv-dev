#include <stdio.h>      // printf
#include <fcntl.h>      // open
#include <unistd.h>     // read, close, getpagesize
#include <sys/mman.h>   // mmap
#include <string.h>     // memcmp, strlen
#include <assert.h>     // assert

#define DEMO_DEV_NAME   "/dev/demo_dev"

int main()
{
    char buf[64];
    int fd;
    char *addr = NULL;
    off_t offset;
    int ret;
    char *message = "Hello World\n";
    char *message2 = "I'm superman\n";

    fd = open(DEMO_DEV_NAME, O_RDWR);
    if (fd < 0) {
        printf("open device %s failed\n", DEMO_DEV_NAME);
        return -1;
    }
    /* 映射2页，offset 3页 */
    addr = mmap(NULL, getpagesize() * 2, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_LOCKED, fd, getpagesize() * 3);

    /* 写到mmap映射的虚拟地址中，通过read读取设备文件 */
    ret =sprintf(addr, "%s", message);
    assert(ret == strlen(message));

    offset = lseek(fd, getpagesize() * 3, SEEK_SET);
    ret = read(fd, buf, sizeof(buf));
    assert(ret == sizeof(buf));
    assert(!memcmp(buf, message, strlen(message)));

    /* 写到一页之后，超出实际物理内存范围 */
    printf("expect bus error\n");
    ret = sprintf(addr + getpagesize(), "something");
    printf("never reach here\n");

    munmap(addr, getpagesize() * 2);
    close(fd);
    return 0;
}

