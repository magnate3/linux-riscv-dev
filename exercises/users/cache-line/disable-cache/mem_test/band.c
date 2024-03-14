#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// 测试读取BIG_ARR一共需要多长时间，然后计算出bandwidth
#define BIG_ARR 999999

int main()
{
    int *block = (int *)calloc(BIG_ARR, sizeof(int));

    int i, temp;
    clock_t start = clock(), total_time;
    for(i = 0; i < BIG_ARR; ++i)
    temp += block[i];

    
    total_time = clock() - start;
    double sec = (double)total_time / (double) CLOCKS_PER_SEC;
    printf("read %d times from main memory need %lf sec\n\
    The bandwidth is %lfbyte/sec\n",\
    BIG_ARR, sec, BIG_ARR / sec);
    return 0;
}
 