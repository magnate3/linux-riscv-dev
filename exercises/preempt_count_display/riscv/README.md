
# os

```
root@Ubuntu-riscv64:~# uname -a
Linux Ubuntu-riscv64 5.15.24-rt31 #3 SMP PREEMPT_RT Mon Jun 6 15:25:43 HKT 2022 riscv64 riscv64 riscv64 GNU/Linux
root@Ubuntu-riscv64:~# 
```

# make

```
make  -C ~/linux-5.15.24-rt/linux-5.15.24  M=$(pwd) ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j64 modules
```


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/preempt_count_display/riscv/rt0.png)
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/preempt_count_display/riscv/rt.png)