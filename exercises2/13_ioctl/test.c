#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "ioctl_test.h"

int main() {
  int answer;
  struct my_struct ms = {7, "LKM"};
  int dev = open("/dev/my_device", O_RDONLY);

  if (dev == -1) {
    printf("failed to open\n");
    return -1;
  }

  ioctl(dev, RD_VALUE, &answer);
  printf("RD_VALUE - answer: %d\n", answer);

  answer = 456;
  ioctl(dev, WR_VALUE, &answer);
  ioctl(dev, RD_VALUE, &answer);
  printf("WR_VALUE and RD_VALUE - answer: %d\n", answer);

  ioctl(dev, GREETER, &ms);
  printf("GREETER\n");

  printf("succeed to open\n");
  close(dev);

  return 0;
}
