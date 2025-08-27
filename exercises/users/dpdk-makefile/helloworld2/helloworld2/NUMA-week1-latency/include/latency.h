#ifndef LATENCY_H
    #define LATENCY_H

    #define _GNU_SOURCE
    //#define SIZE 16384
    #define SIZE 2097152 
    #define REPEATS 100000
    #define STEP SIZE
    enum { NS_PER_SECOND = 1000000000 };

#include <stdio.h>
#include <numaif.h>
#include <sched.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

void latency(int memory_node, int cpu_id);

#endif
