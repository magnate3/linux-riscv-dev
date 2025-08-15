#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <malloc.h>
#include <thread>
#include <list>
#include <atomic>
// #include "./jemalloc-3.6.0/include/jemalloc/jemalloc.h"
using namespace std;

/*
  连续分配释放10000次内存(1024)性能测试, g++ 未优化编译
  单线程耗时: 
  标准库malloc耗时: 4230us free耗时: 667us
  使用dlmalloc耗时: 4102us, dlfree耗时: 853us
  使用jemalloc耗时: 1119us, jefree耗时: 525us
  4个线程平均耗时:
  标准库malloc, free耗时: 36641-72191us
  使用dlmalloc, dlfree耗时: 2232-6131us
  使用jemalloc耗时: 3071-69143us
*/
#define BLOCK_SIZE (1024)
#define MALLOC_TIMES (10000)
std::atomic<int64_t> alloc_time_;
std::atomic<int64_t> free_time_;

class TimeKeeper
{
public :
    TimeKeeper(bool isalloc):alloc_(isalloc)
    {
        clock_gettime(CLOCK_MONOTONIC, &tp_start_);
    }
    ~TimeKeeper()
    {
        struct timespec tp_end;
        clock_gettime(CLOCK_MONOTONIC, &tp_end);
        //
        int64_t cost = tp_end.tv_sec - tp_start_.tv_sec;
        cost = (tp_end.tv_nsec-tp_start_.tv_nsec) + (cost*1000*1000*1000);
        if (alloc_)
        {
            alloc_time_.fetch_add(cost);
        }
        else
        {
            free_time_.fetch_add(cost);
        }
    }
private :
    struct timespec tp_start_;
    bool alloc_;
};

void* mymalloc(size_t size)
{
    TimeKeeper keeper(true);
    return malloc(size);
}

void myfree(void* ptr)
{
    TimeKeeper keeper(false);
    free(ptr);
}

void test_func()
{
    list<char*> li;
    for (int i = 0; i < MALLOC_TIMES; ++i)
    {
        char* p = (char*)mymalloc(BLOCK_SIZE);
        li.push_back(p);
    }
    for (auto p : li)
    {
        myfree(p);
    }
}

int main(int argc, char** argv)
{
    thread t1(test_func);
    thread t2(test_func);
    thread t3(test_func);
    thread t4(test_func);
    t1.join();
    t2.join();
    t3.join();
    t4.join();
    
    int times = 4 * MALLOC_TIMES;
    int64_t malloc_cost = alloc_time_.load();
    int64_t free_cost = free_time_.load();
    printf("4 thread malloc %d times, malloc avg cost: %lld ns.\n", MALLOC_TIMES, malloc_cost / times);
    printf("4 thread free %d times, free avg cost: %lld ns.\n", MALLOC_TIMES, free_cost / times);
    return 0;
}