#compile dpdk-19

## step 1
```
[root@centos7 dpdk-19.11]# pwd
/root/dpdk-19.11
[root@centos7 dpdk-19.11]# 
```
```
export RTE_TARGET=arm64-armv8a-linuxapp-gcc
export RTE_SDK=`pwd`
```
## step 2

```
[root@centos7 helloworld]# pwd
/root/dpdk-19.11/examples/helloworld
[root@centos7 helloworld]# 
```

```
[root@centos7 helloworld]# make
  CC main.o
/root/dpdk-19.11/examples/helloworld/main.c: In function ‘main’:
/root/dpdk-19.11/examples/helloworld/main.c:44:19: error: expected expression before ‘void’
         rte_pause(void);
                   ^
/root/dpdk-19.11/examples/helloworld/main.c:44:19: error: too many arguments to function ‘rte_pause’
In file included from /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_pause.h:13:0,
                 from /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/generic/rte_rwlock.h:26,
                 from /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_rwlock.h:12,
                 from /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_fbarray.h:40,
                 from /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_memory.h:25,
                 from /root/dpdk-19.11/examples/helloworld/main.c:11:
/root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_pause_64.h:15:20: note: declared here
 static inline void rte_pause(void)
                    ^
/root/dpdk-19.11/examples/helloworld/main.c: At top level:
cc1: warning: unrecognized command line option "-Wno-address-of-packed-member" [enabled by default]
make[1]: *** [/root/dpdk-19.11/mk/internal/rte.compile-pre.mk:116: main.o] Error 1
make: *** [/root/dpdk-19.11/mk/rte.extapp.mk:15: all] Error 2
[root@centos7 helloworld]# cat  /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_pause.h
/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2017 Cavium, Inc
 */

#ifndef _RTE_PAUSE_ARM_H_
#define _RTE_PAUSE_ARM_H_

#ifdef __cplusplus
extern "C" {
#endif

#ifdef RTE_ARCH_64
#include <rte_pause_64.h>
#else
#include <rte_pause_32.h>
#endif

#ifdef __cplusplus
}
#endif

#endif /* _RTE_PAUSE_ARM_H_ */
[root@centos7 helloworld]# 
```


# librte_eal/common/include/arch/arm/rte_pause_64.h:15:static inline void rte_pause(void)
```
/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2017 Cavium, Inc
 */

#ifndef _RTE_PAUSE_ARM64_H_
#define _RTE_PAUSE_ARM64_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <rte_common.h>
#include "generic/rte_pause.h"

static inline void rte_pause(void)
{
        asm volatile("yield" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif /* _RTE_PAUSE_ARM64_H_ */
```
# rte_pause_64.h

```
[root@centos7 helloworld]# gdb  build/app/helloworld
GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-120.el7
Copyright (C) 2013 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.  Type "show copying"
and "show warranty" for details.
This GDB was configured as "aarch64-redhat-linux-gnu".
For bug reporting instructions, please see:
<http://www.gnu.org/software/gdb/bugs/>...
Reading symbols from /root/dpdk-19.11/examples/helloworld/build/app/helloworld...done.
(gdb) set args --no-huge
(gdb) b rte_pause_64.h:15
Breakpoint 1 at 0x4bf368: rte_pause_64.h:15. (118 locations)
(gdb) 
```

```


Breakpoint 1, rte_pause () at /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_pause_64.h:17
17              asm volatile("yield" ::: "memory");
Missing separate debuginfos, use: debuginfo-install glibc-2.17-325.el7_9.aarch64 libgcc-4.8.5-44.el7.aarch64 numactl-libs-2.0.12-5.el7.aarch64
(gdb) bt
#0  rte_pause () at /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_pause_64.h:17
#1  0x00000000005a4610 in rte_delay_us_block (us=10) at /root/dpdk-19.11/lib/librte_eal/common/eal_common_timer.c:35
#2  0x000000000076fa5c in wait_for_status_poll (chain=0xffffbd342200) at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_api_cmd.c:333
#3  0x000000000076fab8 in wait_for_api_cmd_completion (chain=0xffffbd342200, ctxt=0xffffbd327800, ack=0x0, ack_size=0) at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_api_cmd.c:353
#4  0x000000000076fc74 in api_cmd (chain=0xffffbd342200, dest=HINIC_NODE_ID_MGMT_HOST, cmd=0xffffbd33ff00, cmd_size=116, ack=0x0, ack_size=0) at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_api_cmd.c:416
#5  0x000000000076fcb0 in hinic_api_cmd_write (chain=0xffffbd342200, dest=HINIC_NODE_ID_MGMT_HOST, cmd=0xffffbd33ff00, size=116) at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_api_cmd.c:431
#6  0x0000000000779dc8 in send_msg_to_mgmt_sync (pf_to_mgmt=0xffffbd342e80, mod=HINIC_MOD_COMM, cmd=82 'R', msg=0xffffffffef20, msg_len=88, ack_type=HINIC_MSG_ACK, direction=HINIC_MSG_DIRECT_SEND, resp_msg_id=65535)
    at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_mgmt.c:323
#7  0x000000000077a008 in hinic_pf_to_mgmt_sync (hwdev=0xffffbd3aa180, mod=HINIC_MOD_COMM, cmd=82 'R', buf_in=0xffffffffef20, in_size=88, buf_out=0xffffffffef20, out_size=0xffffffffef1e, timeout=0)
    at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_mgmt.c:407
#8  0x000000000077a400 in hinic_msg_to_mgmt_sync (hwdev=0xffffbd3aa180, mod=HINIC_MOD_COMM, cmd=82 'R', buf_in=0xffffffffef20, in_size=88, buf_out=0xffffffffef20, out_size=0xffffffffef1e, timeout=0)
    at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_mgmt.c:495
#9  0x0000000000776ce8 in hinic_get_board_info (hwdev=0xffffbd3aa180, info=0xffffffffefa0) at /root/dpdk-19.11/drivers/net/hinic/base/hinic_pmd_hwdev.c:997
#10 0x000000000078ccc8 in hinic_card_workmode_check (nic_dev=0xffffbd3ad680) at /root/dpdk-19.11/drivers/net/hinic/hinic_pmd_ethdev.c:2559
#11 0x000000000078d1b0 in hinic_nic_dev_create (eth_dev=0x1021380 <rte_eth_devices>) at /root/dpdk-19.11/drivers/net/hinic/hinic_pmd_ethdev.c:2693
#12 0x000000000078d86c in hinic_func_init (eth_dev=0x1021380 <rte_eth_devices>) at /root/dpdk-19.11/drivers/net/hinic/hinic_pmd_ethdev.c:3009
#13 0x000000000078dc34 in hinic_dev_init (eth_dev=0x1021380 <rte_eth_devices>) at /root/dpdk-19.11/drivers/net/hinic/hinic_pmd_ethdev.c:3105
#14 0x0000000000787ea0 in rte_eth_dev_pci_generic_probe (pci_dev=0x1181960, private_data_size=1320, dev_init=0x78db50 <hinic_dev_init>) at /root/dpdk-19.11/arm64-armv8a-linuxapp-gcc/include/rte_ethdev_pci.h:164
#15 0x000000000078dd00 in hinic_pci_probe (pci_drv=0xe431e0 <rte_hinic_pmd>, pci_dev=0x1181960) at /root/dpdk-19.11/drivers/net/hinic/hinic_pmd_ethdev.c:3147
#16 0x00000000005de55c in rte_pci_probe_one_driver (dr=0xe431e0 <rte_hinic_pmd>, dev=0x1181960) at /root/dpdk-19.11/drivers/bus/pci/pci_common.c:199
#17 0x00000000005de754 in pci_probe_all_drivers (dev=0x1181960) at /root/dpdk-19.11/drivers/bus/pci/pci_common.c:274
#18 0x00000000005de820 in rte_pci_probe () at /root/dpdk-19.11/drivers/bus/pci/pci_common.c:309
#19 0x00000000005ab828 in rte_bus_probe () at /root/dpdk-19.11/lib/librte_eal/common/eal_common_bus.c:72
#20 0x0000000000590be8 in rte_eal_init (argc=2, argv=0xfffffffff4f8) at /root/dpdk-19.11/lib/librte_eal/linux/eal/eal.c:1258
#21 0x0000000000464b50 in main ()
(gdb) 

```
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/users/dpdk-arm64/rte_pause.png)