# module-params

[LINUX KERNEL DEVELOPMENT – KERNEL MODULE PARAMETERS](https://devarea.com/linux-kernel-development-kernel-module-parameters/#.Yka1SmhBx3g)


##  insmod ./simp.ko irq=44 devname=simpdev debug=0

```
root@x86:/home/ubuntu/module-params# insmod ./simp.ko irq=44 devname=simpdev debug=0
insmod: ERROR: could not insert module ./simp.ko: Operation not permitted
```
** 查看是否有啟用 secureboot**
```
[ 9875.663285] Lockdown: insmod: unsigned module loading is restricted; see man kernel_lockdown.7
[ 9893.430797] Lockdown: insmod: unsigned module loading is restricted; see man kernel_lockdown.7
[ 9961.416789] Lockdown: insmod: unsigned module loading is restricted; see man kernel_lockdown.7
```

```
root@x86:/home/ubuntu/module-params# dmesg|grep -i secure
[    0.000000] secureboot: Secure boot enabled
[    0.000000] Kernel is locked down from EFI Secure Boot mode; see man kernel_lockdown.7
[    0.005121] secureboot: Secure boot enabled
[    1.227286] Loaded X.509 cert 'Canonical Ltd. Secure Boot Signing: 61482aa2830d0ab2ad5af10b7250da9033ddcef0'
```

```
root@ubuntu:~/module-params# insmod ./simp.ko irq=44 devname=simpdev debug=0
root@ubuntu:~/module-params# dmesg | tail -n 10
[7361080.593383] simp: loading out-of-tree module taints kernel.
[7361080.593555] simp: module verification failed: signature and/or required key missing - tainting kernel
[7361080.594550] hello... irq=44 name=simpdev debug=0
```

```
root@ubuntu:~/module-params# cat /sys/module/simp/parameters/irqtype 
level
```

## rmmod simp.ko
```
root@ubuntu:~/module-params# dmesg | tail -n 10
[7361080.593383] simp: loading out-of-tree module taints kernel.
[7361080.593555] simp: module verification failed: signature and/or required key missing - tainting kernel
[7361080.594550] hello... irq=44 name=simpdev debug=0
[7361251.561776] bye ...irq=44 name=simpdev debug=0
```
