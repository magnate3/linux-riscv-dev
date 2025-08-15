#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main()
{
   struct timespec ts_realtime;
   struct timespec ts_tai;

   clock_gettime(CLOCK_REALTIME, &ts_realtime);
   clock_gettime(CLOCK_TAI, &ts_tai);

   printf("Realtime: %lld.%.9ld\n", (long long)ts_realtime.tv_sec, ts_realtime.tv_nsec);
   printf("TAI: %lld.%.9ld\n", (long long)ts_tai.tv_sec, ts_tai.tv_nsec);
}
