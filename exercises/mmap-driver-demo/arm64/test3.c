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
    /* 映射1个字节 */
    addr = mmap(NULL, 1, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_LOCKED, fd, 0);

    /* 写到mmap映射的虚拟地址中，通过read读取设备文件 */
    ret =sprintf(addr, "%s", message);
    assert(ret == strlen(message));

    ret = read(fd, buf, sizeof(buf));
    assert(ret == sizeof(buf));
    assert(!memcmp(buf, message, strlen(message)));

    /* 写到一页的尾部 */
    ret = sprintf(addr + getpagesize() - sizeof(buf), "%s", message2);
    assert(ret == strlen(message2));

    offset = lseek(fd, getpagesize() - sizeof(buf), SEEK_SET);
    assert(offset == getpagesize() - sizeof(buf));

    ret = read(fd, buf, sizeof(buf));
    assert(ret == sizeof(buf));
    assert(!memcmp(buf, message2, strlen(message2)));

    /* 写到一页之后，超出映射范围 */
    printf("expect segment error\n");
    ret = sprintf(addr + getpagesize(), "something");
    printf("never reach here\n");

    munmap(addr, 1);
    close(fd);
    return 0;
}

