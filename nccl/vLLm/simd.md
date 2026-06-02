

针对FNN中的激活函数（如SwiGLU, GELU, SiLU）进行SIMD优化、查表（LUT）或分段逼近（Piecewise Approximation），是提升推理性能的关键。

+ SIMD 查表法 (Table Lookup / LUT)     
+ SIMD 分段逼近法 (Piecewise Approximation)分段逼近  

##  SwiGLU SIMD 优化技术
 ggml_swiglu  
 
 
```
[root@centos7 simd]# g++ bench_simd.c  -D__ARM_NEON -o bench_simd 
[root@centos7 simd]# ./bench_simd 
=== SIMD vs SCALAR COMPARISON (0.6B model, 60 frames) ===
Talker: hidden=1024, inter=3072 | CP: hidden=1024, inter=1024

1a. Talker SwiGLU SCALAR:         291.78 ms  (5.2M expf)
1b. Talker SwiGLU NEON fast_sig:  501.75 ms  (0.6x speedup)
   Accuracy: max relative error = 0.005469 (first 4 elements)

2a. CP SwiGLU SCALAR:             257.84 ms  (4.6M expf)
2b. CP SwiGLU NEON fast_sig:      448.18 ms  (0.6x speedup)

3a. GELU erff SCALAR:              23.38 ms  (983k erff)
3b. GELU NEON tanh-approx:         96.33 ms  (0.2x speedup)

4. Depthwise conv NEON:            25.66 ms  (scalar ~0.88ms)
5. LayerNorm NEON accum:            6.47 ms  (scalar ~0.29ms)

=== POTENTIAL SAVINGS ===
MAIN THREAD:    -400.3 ms saved (SwiGLU T+CP: 549.6 → 949.9 ms)
DECODER THREAD: -73.0 ms saved (GELU: 23.4 → 96.3 ms)

volatile sink = -0.463110
[root@centos7 simd]# 
```

##  march=armv8.2-a+sve
[从标量到向量，释放ARM服务器全栈算力](https://developer.huawei.com/home/forum/kunpeng/thread-0225203930782208226-1-1.html)   

```
root@centos7:/workspace/simd# g++ --version
g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

root@centos7:/workspace/simd#
```

```
root@centos7:/workspace/simd# g++ -O3 -march=armv8.2-a+sve -o main  main.cpp
root@centos7:/workspace/simd# ./main 
2 4 6 8 2.5 3 3.5 4 
root@centos7:/workspace/simd# 
```