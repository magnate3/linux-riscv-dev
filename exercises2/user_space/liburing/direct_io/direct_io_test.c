#define _GNU_SOURCE
#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int main() {
  int fd = open("/tmp/direct_io_test", O_CREAT | O_RDWR | O_DIRECT);
  if (fd < 0) {
    perror("open");
    return -1;
  }

  const int kBlockSize = 512;

  // buffer 地址不对齐
  {
    char* p = NULL;
    int ret = posix_memalign(
        (void**)&p, kBlockSize / 2,
        kBlockSize);  // 这里有一定概率可以分配出与 kBlockSize 对齐的内存
    assert(ret == 0);
    int n = write(fd, p, kBlockSize);
    assert(n < 0);
    perror("write: not align buffer");
    free(p);
  }

  // buffer 大小不对齐
  {
    char* p = NULL;
    int ret = posix_memalign((void**)&p, kBlockSize, kBlockSize / 2);
    assert(ret == 0);
    int n = write(fd, p, kBlockSize / 2);
    assert(n < 0);
    perror("write: not align buffer size");
    free(p);
  }

  // 文件 offset 不对齐
  {
    char* p = NULL;
    int ret = posix_memalign((void**)&p, kBlockSize, kBlockSize);
    assert(ret == 0);
    off_t offset = lseek(fd, kBlockSize / 2, SEEK_SET);
    assert(offset == kBlockSize / 2);
    int n = write(fd, p, kBlockSize);
    assert(n < 0);
    perror("write: not align buffer offset");
    free(p);
  }

  // 三者对齐
  {
    char* p = NULL;
    int ret = posix_memalign((void**)&p, kBlockSize, kBlockSize);
    assert(ret == 0);
    off_t offset = lseek(fd, 0, SEEK_SET);
    assert(offset == 0);
    int n = write(fd, p, kBlockSize);
    assert(n == kBlockSize);
    printf("write ok\n");
    free(p);
  }

  return 0;
}
