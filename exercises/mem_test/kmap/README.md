#  CONFIG_HIGHMEM
```
ubuntu@ubuntux86:/boot$ uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
ubuntu@ubuntux86:/boot$ grep HIGHMEM  config-5.13.0-30-generic 
ubuntu@ubuntux86:/boo
```
# #define PKMAP_BASE

```
ubuntu@ubuntux86:/work/linux-xlnx/arch/x86$ grep PKMAP_BASE   -rn * | grep define
include/asm/pgtable_32_areas.h:38:#define PKMAP_BASE            \
include/asm/pgtable_32_areas.h:42:# define VMALLOC_END  (PKMAP_BASE - 2 * PAGE_SIZE)
include/asm/highmem.h:57:#define PKMAP_NR(virt)  ((virt-PKMAP_BASE) >> PAGE_SHIFT)
include/asm/highmem.h:58:#define PKMAP_ADDR(nr)  (PKMAP_BASE + ((nr) << PAGE_SHIFT))
```
# insmod  kmap_test.ko
```
root@ubuntux86:/work/kernel_learn# insmod  kmap_test.ko
root@ubuntux86:/work/kernel_learn# dmesg | tail -n 10
[26711.174551] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[26711.174557] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[26711.174563] nvme 0000:02:00.0:    [ 0] RxErr                 
[26816.921081] kmap example init
[26816.921086] ***************************************************
[26816.921088] FIXMAP:      ffffffffff579000 -- ffffffffff7ff000
[26816.921091] VMALLOC:     ffffa9e540000000 -- ffffc9e53fffffff
[26816.921094] LOWMEM:      ffff985980000000 -- ffff98620d800000
[26816.921096] ***************************************************
[26816.921098] [*]unsigned int *KMAP_int:       Address: 0x208af000
```
***Address: 0x208af000 比 FIXMAP 小***


# high api_atomic map

```
insmod kmap_test2.ko 
[27499.622297] high api_atomic map example init
[27499.622304] ***************************************************
[27499.622306] FIXMAP:      ffffffffff579000 -- ffffffffff7ff000
[27499.622312] VMALLOC:     ffffa9e540000000 -- ffffc9e53fffffff
[27499.622315] LOWMEM:      ffff985980000000 -- ffff98620d800000
[27499.622318] ***************************************************
[27499.622320] [*]unsigned int *KMAP_atomic:       Address: 0x24b16000
```

***Address: 0x24b16000 比 FIXMAP 小***