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

    fd = open(DEMO_DEV_NAME, O_RDWR);
    if (fd < 0) {
        printf("open device %s failed\n", DEMO_DEV_NAME);
        return -1;
    }
    sleep(5);
    addr = mmap(NULL, (size_t)getpagesize(), PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_LOCKED, fd, 0);
    sleep(5);
    /* 测试映射正确 */
    /* 写到mmap映射的虚拟地址中，通过read读取设备文件 */
    ret = sprintf(addr, "%s", message);
    assert(ret == strlen(message));

    ret = read(fd, buf, 64);
    assert(ret == 64);
    assert(!memcmp(buf, message, strlen(message)));

    /* 通过write写入设备文件，修改体现在mmap映射的虚拟地址 */
    ret = write(fd, message2, strlen(message2));

    assert(ret == strlen(message2));
    assert(!memcmp(addr + 64, message2, strlen(message2)));
    getchar();
    munmap(addr, (size_t)getpagesize());
    close(fd);
    return 0;
}

