#include "latency.h"

volatile int sink = 0;

void sub_timespec(struct timespec t1, struct timespec t2, struct timespec *td)
{
    td->tv_nsec = t2.tv_nsec - t1.tv_nsec;
    td->tv_sec  = t2.tv_sec - t1.tv_sec;
    if (td->tv_sec > 0 && td->tv_nsec < 0)
    {
        td->tv_nsec += NS_PER_SECOND;
        td->tv_sec--;
    }
    else if (td->tv_sec < 0 && td->tv_nsec > 0)
    {
        td->tv_nsec -= NS_PER_SECOND;
        td->tv_sec++;
    }
}


void latency(int memory_node, int cpu_id)
{
    char *buffer = malloc(sizeof(char) * SIZE);
    unsigned long nodemask[1] = {1UL << memory_node};

    mbind(buffer, SIZE, MPOL_BIND, nodemask, 2, 0);
    int i;

    for (i = 0; i < SIZE; i++)
        buffer[i] = 0;
    
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);

    struct timespec start, finish, delta;
    double moy = 0;
    int l;
    for (l = 0; l < REPEATS; l++) {
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        for (i = 0; i < SIZE; i += STEP)
            sink += buffer[i];
        clock_gettime(CLOCK_MONOTONIC, &finish);
        sub_timespec(start, finish, &delta);
        moy += delta.tv_sec * 1e9 + delta.tv_nsec;
    }

    moy /= REPEATS;
    moy /= (SIZE / STEP);

    printf("Thread CPU %d, mémoire node %d → latence moyenne : %.2f ns\n",
        cpu_id, memory_node, moy);
}
