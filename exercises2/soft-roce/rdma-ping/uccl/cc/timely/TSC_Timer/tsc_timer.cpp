//
//  tsc_time.cpp
//
//  Created by ZhongUncle on 2024/4/15.
//

#include <iostream>

#define BIT(nr) (1UL << (nr)) //set nr-th bit as 1

// Get TSC clock count
static inline unsigned long long rdtsc(void)
{
    unsigned long long low, high;
    asm volatile ("rdtsc" : "=a" (low), "=d" (high));
    // TSC is a MSR, so it saves data to EDX:EAX
    return low | (high << 32);
}

// Determine whether TSC can be use
static inline bool isTSC(void)
{
    uint32_t a=0x1, b, c, d;
    asm volatile ("cpuid"
         : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
         : "a" (a), "b" (b), "c" (c), "d" (d)
         );
    if ((d & BIT(4))) {
        // TSC exist!
        a=0x80000007;
        asm volatile ("cpuid"
             : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
             : "a" (a), "b" (b), "c" (c), "d" (d)
             );
        if ((d & BIT(8))) {
            // Invariant TSC available!
            return true;
        }
    } else {
        // TSC not exist
        return false;
    }
    return false;
}

// Get CPU model
static inline unsigned int cpu_model(void)
{
    uint32_t a=0x1, b, c, d;
    asm volatile ("cpuid\n\t"
         : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
         : "a" (a), "b" (b), "c" (c), "d" (d)
         );
    uint32_t model = (a>>4)&0b1111;
    uint32_t extend_model = (a>>16)&0b1111;
    
    return (extend_model<<4)|model;
}

// Get TSC Frequency
// tsc_freq = ebx / eax * (crystal_clock or ecx)
static inline unsigned long tsc_freq(void)
{
    uint32_t model = cpu_model();
    
    uint32_t a=0x15, b, c, d;
    asm volatile ("cpuid\n\t"
         : "=a" (a), "=b" (b), "=c" (c), "=d" (d)
         : "0" (a), "1" (b), "2" (c), "3" (d)
         );
    
    if (c != 0)
        return b / a * c;
    
    // Intel® Xeon® Processor Scalable Family with CPUID signature 06_55H.
    if (model == 0x55)
        return b / a * 25000000;
    // Next Generation Intel® Atom'™ processors based on Goldmont Microarchitecture with CPUID signature 06_5CH (does not include Intel Xeon processors).
    if (model == 0x5c)
        return b / a * 19200000;
    
    //Intel® Core'' processors and Intel® Xeon® W Processor Family.
    return b/a*24000000;
}

int main(int argc, const char * argv[]) {
    // Determine whether there is a reliable TSC
    if (!isTSC()) {
        printf("TSC is not exist or variant!");
    }
    
    // Get TSC Frequency
    unsigned long freq = tsc_freq();
    
    // Use clock_gettime() as a comparison
    struct timespec start;
        clock_gettime(CLOCK_MONOTONIC, &start);
    // Get TSC clock count as start
    uint64_t rdtsc1 = rdtsc();
    
    // Testing Code
    for (int i=0; i<100; i++) {
        std::cout << "Hello, World!";
    }
    std::cout << std::endl;
    
    // Get TSC clock count as end
    uint64_t rdtsc2 = rdtsc();
    struct timespec end;
        clock_gettime(CLOCK_MONOTONIC, &end);
    
    // Calculate time from TSC clock count and frequency
    double time = (double)(rdtsc2-rdtsc1)/(double)freq*1e9;
    // Print clock count, frequency and calculated time
    printf("clock\t = %llu cycles\n", rdtsc2-rdtsc1);
    printf("freq\t = %lu Hz\n", freq);
    printf("TSC time = %.0f ns\n", time);
    
    // Calculate duration time from clock_gettime() as a comparison
    double duration = (double)(end.tv_nsec-start.tv_nsec) + (double)(end.tv_sec-start.tv_sec)*((double) 1e9);
    // Print duration time from clock_gettime()
    printf("duration = %.0f ns\n", duration);
    
    return 0;
}
