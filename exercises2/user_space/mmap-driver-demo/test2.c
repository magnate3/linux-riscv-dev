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
    int ret;
    char *message = "Hello World\n";
    char *message2 = "I'm superman\n";

    /* 另一进程打开同一设备文件，然后用mmap映射 */
    fd = open(DEMO_DEV_NAME, O_RDWR);
    if (fd < 0) {
        printf("open device %s failed\n", DEMO_DEV_NAME);
        return -1;
    }
    addr = mmap(NULL, (size_t)getpagesize(), PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_LOCKED, fd, 0);

    /* 通过read读取设备文件 */
    ret = read(fd, buf, sizeof(buf));

    assert(ret == sizeof(buf));
    assert(!memcmp(buf, message, strlen(message)));

    /* 通过mmap映射的虚拟地址读取 */
    assert(!memcmp(addr + sizeof(buf), message2, strlen(message2)));

    munmap(addr, (size_t)getpagesize());
    close(fd);
    return 0;
}

