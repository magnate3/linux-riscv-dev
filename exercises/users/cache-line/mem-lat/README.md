
# x86

```
root@ubuntux86:/work/test# lscpu
Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Byte Order:                      Little Endian
Address sizes:                   39 bits physical, 48 bits virtual
CPU(s):                          20
On-line CPU(s) list:             0-19
Thread(s) per core:              2
Core(s) per socket:              10
Socket(s):                       1
NUMA node(s):                    1
Vendor ID:                       GenuineIntel
CPU family:                      6
Model:                           165
Model name:                      Intel(R) Core(TM) i9-10900K CPU @ 3.70GHz
Stepping:                        5
CPU MHz:                         3700.000
CPU max MHz:                     5300.0000
CPU min MHz:                     800.0000
BogoMIPS:                        7399.70
Virtualization:                  VT-x
L1d cache:                       320 KiB
L1i cache:                       320 KiB
L2 cache:                        2.5 MiB
L3 cache:                        20 MiB
NUMA node0 CPU(s):               0-19
Vulnerability Itlb multihit:     KVM: Mitigation: VMX disabled
Vulnerability L1tf:              Not affected
Vulnerability Mds:               Not affected
Vulnerability Meltdown:          Not affected
Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled via prctl and seccomp
Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, RSB filling
Vulnerability Srbds:             Not affected
Vulnerability Tsx async abort:   Not affected
Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pd
                                 pe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 moni
                                 tor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c
                                  rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow vnmi flexpriority ept 
                                 vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsav
                                 es dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp pku ospke md_clear flush_l1d arch_capabilities
root@ubuntux86:/work/test# 
```

```
root@ubuntux86:/work/test# bash mem-lat.sh 
Buffer size: 1 KB, stride 64, time 0.003821 s, latency 3.64 ns
Buffer size: 2 KB, stride 64, time 0.003837 s, latency 3.66 ns
Buffer size: 4 KB, stride 64, time 0.003834 s, latency 3.66 ns
Buffer size: 8 KB, stride 64, time 0.003839 s, latency 3.66 ns
Buffer size: 16 KB, stride 64, time 0.003839 s, latency 3.66 ns
Buffer size: 32 KB, stride 64, time 0.004050 s, latency 3.86 ns
Buffer size: 64 KB, stride 64, time 0.010903 s, latency 10.40 ns
Buffer size: 128 KB, stride 64, time 0.007224 s, latency 6.89 ns
Buffer size: 256 KB, stride 64, time 0.005715 s, latency 5.45 ns
Buffer size: 512 KB, stride 64, time 0.005076 s, latency 4.84 ns
Buffer size: 1024 KB, stride 64, time 0.004404 s, latency 4.20 ns
Buffer size: 2048 KB, stride 64, time 0.004051 s, latency 3.86 ns
Buffer size: 4096 KB, stride 64, time 0.004064 s, latency 3.88 ns
Buffer size: 8192 KB, stride 64, time 0.003677 s, latency 3.51 ns
Buffer size: 16384 KB, stride 64, time 0.004645 s, latency 4.43 ns
```
# arm64

```
[root@centos7 test]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 test]# lscpu
Architecture:          aarch64
Byte Order:            Little Endian
CPU(s):                128
On-line CPU(s) list:   0-127
Thread(s) per core:    1
Core(s) per socket:    64
Socket(s):             2
NUMA node(s):          4
Model:                 0
CPU max MHz:           2600.0000
CPU min MHz:           200.0000
BogoMIPS:              200.00
L1d cache:             64K
L1i cache:             64K
L2 cache:              512K
L3 cache:              65536K
NUMA node0 CPU(s):     0-31
NUMA node1 CPU(s):     32-63
NUMA node2 CPU(s):     64-95
NUMA node3 CPU(s):     96-127
Flags:                 fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma dcpop
[root@centos7 test]# 

```

```
[root@centos7 test]# bash mem-lat.sh 
Buffer size: 1 KB, stride 64, time 0.001613 s, latency 1.54 ns
Buffer size: 2 KB, stride 64, time 0.001613 s, latency 1.54 ns
Buffer size: 4 KB, stride 64, time 0.001613 s, latency 1.54 ns
Buffer size: 8 KB, stride 64, time 0.001613 s, latency 1.54 ns
Buffer size: 16 KB, stride 64, time 0.001613 s, latency 1.54 ns
Buffer size: 32 KB, stride 64, time 0.001613 s, latency 1.54 ns
Buffer size: 64 KB, stride 64, time 0.001636 s, latency 1.56 ns
Buffer size: 128 KB, stride 64, time 0.003272 s, latency 3.12 ns
Buffer size: 256 KB, stride 64, time 0.003565 s, latency 3.40 ns
Buffer size: 512 KB, stride 64, time 0.003555 s, latency 3.39 ns
Buffer size: 1024 KB, stride 64, time 0.004557 s, latency 4.35 ns
Buffer size: 2048 KB, stride 64, time 0.004574 s, latency 4.36 ns
Buffer size: 4096 KB, stride 64, time 0.005453 s, latency 5.20 ns
Buffer size: 8192 KB, stride 64, time 0.005743 s, latency 5.48 ns
Buffer size: 16384 KB, stride 64, time 0.007008 s, latency 6.68 ns
[root@centos7 test]# 
```

# riscv

```
root@Ubuntu-riscv64:~/test# bash mem-lat.sh 
Buffer size: 1 KB, stride 64, time 0.003255 s, latency 3.10 ns
Buffer size: 2 KB, stride 64, time 0.003279 s, latency 3.13 ns
Buffer size: 4 KB, stride 64, time 0.003199 s, latency 3.05 ns
Buffer size: 8 KB, stride 64, time 0.003341 s, latency 3.19 ns
Buffer size: 16 KB, stride 64, time 0.003317 s, latency 3.16 ns
Buffer size: 32 KB, stride 64, time 0.003604 s, latency 3.44 ns
Buffer size: 64 KB, stride 64, time 0.023478 s, latency 22.39 ns
Buffer size: 128 KB, stride 64, time 0.027512 s, latency 26.24 ns
Buffer size: 256 KB, stride 64, time 0.027894 s, latency 26.60 ns
Buffer size: 512 KB, stride 64, time 0.028074 s, latency 26.78 ns
Buffer size: 1024 KB, stride 64, time 0.029299 s, latency 27.94 ns
Buffer size: 2048 KB, stride 64, time 0.051619 s, latency 49.23 ns
Buffer size: 4096 KB, stride 64, time 0.137568 s, latency 131.20 ns
Buffer size: 8192 KB, stride 64, time 0.159940 s, latency 152.54 ns
Buffer size: 16384 KB, stride 64, time 0.162440 s, latency 154.93 ns
```

```
root@Ubuntu-riscv64:~/test# lscpu
Architecture:        riscv64
Byte Order:          Little Endian
CPU(s):              4
On-line CPU(s) list: 0-3
Thread(s) per core:  4
Core(s) per socket:  1
Socket(s):           1
L1d cache:           32 KiB
L1i cache:           32 KiB
L2 cache:            2 MiB
root@Ubuntu-riscv64:~/test# 
```

# u54
```
root@Ubuntu-riscv64:~# bash mem-lat.sh 
Buffer size: 1 KB, stride 64, time 0.002294 s, latency 2.19 ns
Buffer size: 2 KB, stride 64, time 0.002336 s, latency 2.23 ns
Buffer size: 4 KB, stride 64, time 0.002308 s, latency 2.20 ns
Buffer size: 8 KB, stride 64, time 0.002321 s, latency 2.21 ns
Buffer size: 16 KB, stride 64, time 0.002307 s, latency 2.20 ns
Buffer size: 32 KB, stride 64, time 0.002567 s, latency 2.45 ns
Buffer size: 64 KB, stride 64, time 0.020318 s, latency 19.38 ns
Buffer size: 128 KB, stride 64, time 0.023875 s, latency 22.77 ns
Buffer size: 256 KB, stride 64, time 0.024225 s, latency 23.10 ns
Buffer size: 512 KB, stride 64, time 0.024468 s, latency 23.34 ns
Buffer size: 1024 KB, stride 64, time 0.026238 s, latency 25.02 ns
Buffer size: 2048 KB, stride 64, time 0.069823 s, latency 66.59 ns
Buffer size: 4096 KB, stride 64, time 0.166807 s, latency 159.09 ns
Buffer size: 8192 KB, stride 64, time 0.190931 s, latency 182.10 ns
Buffer size: 16384 KB, stride 64, time 0.192438 s, latency 183.54 ns
```

```
root@Ubuntu-riscv64:~# lscpu
Architecture:        riscv64
Byte Order:          Little Endian
CPU(s):              4
On-line CPU(s) list: 0-3
Thread(s) per core:  4
Core(s) per socket:  1
Socket(s):           1
L1d cache:           32 KiB
L1i cache:           32 KiB
L2 cache:            2 MiB
root@Ubuntu-riscv64:~# 

```

