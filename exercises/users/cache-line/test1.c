#include<stdio.h>
#include <stdint.h>
#define RTE_CACHE_LINE_SIZE         64
#define __rte_cache_aligned         __attribute__((__aligned__(RTE_CACHE_LINE_SIZE)))
struct A{ 
    uint32_t a;
    uint64_t b;
    uint64_t c;
}__rte_cache_aligned;
struct B{ 
    uint32_t a;
    uint64_t b;
    uint64_t c;
};
int main()
{
    printf("sizeof(A): %d, sizeof(B): %d \n", sizeof(struct A),  sizeof(struct B));
    return 0;
}
