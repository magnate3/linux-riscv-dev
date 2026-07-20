#include <stdio.h>
#include <emmintrin.h>

int sharedData = 0;

void thread1()
{
    // 线程1写入数据之前执行内存屏障
    _mm_sfence();
    sharedData = 1;
}

void thread2()
{
    // 线程2读取数据之前执行内存屏障
    _mm_lfence();
    int data = sharedData;
    printf("Thread 2: sharedData = %d\n", data);
}

int main()
{
    thread1();
    thread2();
    return 0;
}
 