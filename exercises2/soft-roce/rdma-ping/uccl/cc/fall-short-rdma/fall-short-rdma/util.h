#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <inttypes.h>
#include <stdlib.h>

#ifndef TRUE
#define TRUE (1)
#endif

#ifndef FALSE
#define FALSE (0)
#endif

#define BILLION 1000000000L
#define MAX_CPUS 16
void bindingCPU(int num);
void die(const char *reason);
double get_nsecs(struct timespec time);

