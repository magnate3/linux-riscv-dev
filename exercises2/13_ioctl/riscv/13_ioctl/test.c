#define _GNU_SOURCE
#include <sched.h>
#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <string.h>
//typedef u_int64_ uint64_t;
#include "ioctl_test.h"

int main() {
  uint64_t answer;
  char buf[64] = {0};
  // 创建一个CPU集合并将进程绑定到其中一个CPU核心
  cpu_set_t mask;
  struct my_struct ms = {7, "LKM"};
  int dev = -1;
  size_t size = getpagesize();
  pid_t pid = getpid();
  void * addr = NULL;
  CPU_ZERO(&mask);
  CPU_SET(0, &mask); // 将进程绑定到第一个CPU核心
  if (sched_setaffinity(pid, sizeof(mask), &mask) == -1) {
         perror("sched_setaffinity");
         return 1;
  }

  if(posix_memalign(&addr, size, size)){
    printf("failed to posix_memalign\n");
    return -1;
  }
  dev = open("/dev/my_device", O_RDONLY);
  if (dev == -1) {
    printf("failed to open\n");
    return -1;
  }

  ioctl(dev, RD_VALUE, &answer);
  printf("RD_VALUE - answer: %lu\n", answer);

  answer = (uint64_t)addr;
  ioctl(dev, WR_VALUE, &answer);
  memcpy(buf,addr, 32);
  printf("buf is %s \n", buf);

  ioctl(dev, RD_VALUE, &answer);
  printf("WR_VALUE and RD_VALUE - answer: %lu\n", answer);

  ioctl(dev, GREETER, &ms);
  printf("GREETER\n");

  printf("succeed to open\n");
  close(dev);

  return 0;
}
