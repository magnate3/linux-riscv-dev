# TSC_Timer
This repository implements TSC timer using rdtsc and cpuid inline assembly of C/C++, no any system kernel or function. So you can use it in x86 macOS and Linux.

There are 2 files for C and C++ respectively. You can use Clang or GCC to complie them:

```
//C
cc tsc_timer.c -o tsc_timer
//C++
c++ tsc_timer.cpp -o tsc_timer
```

And print below:

```
% ./tsc_timer
Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!Hello, World!
clock    = 222823 cycles
freq     = 3000000000 Hz
TSC time = 74274 ns
duration = 75000 ns
```

`duration` is from `clock_gettime()`, it is a comparison.

You can see the result `TSC time` from `tsc_timer` is closely to `duration`.


## Bugs
If you find a bug, please report it!
## references
[如何使用rdtsc和C/C++来测量运行时间（如何使用内联汇编和获取CPU的TSC时钟频率)](https://blog.csdn.net/qq_33919450/article/details/137979409)
