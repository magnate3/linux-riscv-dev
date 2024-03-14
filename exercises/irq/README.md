 
 # cat  /proc/interrupts |awk -F ":" '{print $1}'
 
 ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/irq/irq1.png)
 # insmod irq_test.ko 
 
 ```
 [root@centos7 irq]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 irq]# 
 ```
 
 ```
 [root@centos7 irq]# insmod irq_test.ko 
[root@centos7 irq]# dmesg | tail -n 15
[896908.085930] Exception stack(0xffff00002eaefec0 to 0xffff00002eaf0000)
[896908.092426] fec0: 0000000035b10248 0000000000000800 14206cc847f53800 0000000000002002
[896908.100305] fee0: 0000ffff9eff0e38 0000ffffd1b3f229 0000ffff9eebaf24 0000000000000000
[896908.108185] ff00: 000000000000006a 1999999999999999 00000000ffffffff 0000000000000000
[896908.116064] ff20: 0000000000000005 ffffffffffffffff 0000ffff9ee71a94 00000000000059d0
[896908.123944] ff40: 0000ffff9ef4fa40 0000000000440320 0000ffffd1b40050 0000000035b101e0
[896908.131824] ff60: 0000ffffd1b4f7cb 0000ffffd1b40570 0000000000000000 0000000035b101e0
[896908.139703] ff80: 00000000004178b0 0000000000000000 0000000000440338 0000000035b10010
[896908.147583] ffa0: 0000ffffd1b40578 0000ffffd1b40280 0000000000411898 0000ffffd1b40280
[896908.155462] ffc0: 0000ffff9ef4fa48 0000000080000000 0000000035b10248 000000000000006a
[896908.163342] ffe0: 0000000000000000 0000000000000000 0000000000000000 0000000000000000
[896908.171222] [<ffff00000808392c>] __sys_trace_return+0x0/0x4
[896908.176854] ---[ end trace e16a2ce2a31c2549 ]---
[896983.094736] [RToax]request irq [RToax]!
[896983.098643] [RToax]: request_irq() failed
[root@centos7 irq]# insmod irq_test.ko 
 ```
 
 
 
 # x86
 
 ```
 root@ubuntux86:/work/uio# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:/work/uio# 
 ```
 
 ```
 root@ubuntux86:/work/uio# insmod irq_test.ko 
root@ubuntux86:/work/uio# dmesg | tail -n 10
[31537.708100] nvme 0000:02:00.0:    [ 0] RxErr                 
[31543.596973] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[31543.596980] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[31543.596982] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[31543.596983] nvme 0000:02:00.0:    [ 0] RxErr                 
[31544.769558] [RToax]request irq [RToax]!
[31549.740487] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[31549.740507] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[31549.740513] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[31549.740520] nvme 0000:02:00.0:    [ 0] RxErr                 
root@ubuntux86:/work/uio# dmesg | grep 'RToax'
[31544.769558] [RToax]request irq [RToax]!
 ```
 
 ```
 root@ubuntux86:/work/uio# cat /proc/interrupts | grep 'RToax'
   2:          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0          0    XT-PIC       [RToax]
root@ubuntux86:/work/uio# 
 ```