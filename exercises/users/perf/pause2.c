#include <stdio.h>
#define TIMES 5

static inline unsigned long long rdtsc(void)
{
    unsigned long low, high;
    asm volatile("rdtsc" : "=a" (low), "=d" (high) );
    return ((low) | (high) << 32);
}

void pause_test()
{
    int i = 0;
    for (i = 0; i < TIMES; i++) {
        asm(
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"\
                "pause\n"
                ::
                :);
    }
}

unsigned long pause_cycle()
{
    unsigned long start, finish, elapsed;
    start = rdtsc();
    pause_test();
    finish = rdtsc();
    elapsed = finish - start;
    printf("Pause的cycles约为:%ld\n", elapsed / 100);
    return 0;
}

int main()
{
    pause_cycle();
    return 0;
}