#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
 
#define CLOCKFD 3
#define FD_TO_CLOCKID(fd)   ((~(clockid_t) (fd) << 3) | CLOCKFD)
#define CLOCKID_TO_FD(clk)  ((unsigned int) ~((clk) >> 3))
 
int
main(int argc, char *argv[])
{
    struct timespec ts;
    clockid_t clkid;
#ifndef SOFTSTAMP
    int fd;
 
    fd = open("/dev/ptp0", O_RDWR);
    clkid = FD_TO_CLOCKID(fd);
#else
    clkid = CLOCK_REALTIME;
#endif
    if (clock_gettime(clkid, &ts) == -1) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    printf("Date: %s \n", ctime(&ts.tv_sec));
 
    exit(EXIT_SUCCESS);
}
 