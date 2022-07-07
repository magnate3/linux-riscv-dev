# hugepage_manage
hugepage manage learned from dpdk


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
#  mkdir objs

```
[root@centos7 dpdk-hugepage_manage]# ls objs/
hugepage_malloc.o  hugepage_memory.o  main.o  runtime_info.o  sys_fs.o
[root@centos7 dpdk-hugepage_manage]# 
```

# run  ./test.app

```
[root@centos7 dpdk-hugepage_manage]# umount /mnt/huge
[root@centos7 dpdk-hugepage_manage]# mount -t hugetlbfs nodev /mnt/hugepages
mount: mount point /mnt/hugepages does not exist
[root@centos7 dpdk-hugepage_manage]# mkdir  /mnt/hugepages
[root@centos7 dpdk-hugepage_manage]# mount -t hugetlbfs nodev /mnt/hugepages
[root@centos7 dpdk-hugepage_manage]# ./test.app 
hugepage files init done...
64 hugepage memsegs init done...
Malloc heap init done...
heap_0 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_1 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_2 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_3 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
Malloc done...
test_data:abcdefghij
heap_0 total_size:8589933568 alloc_counter:1
Free_list_11:[elem0-size:536870720]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_1 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_2 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_3 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
Malloc and free test pass...
heap_0 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_1 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_2 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
heap_3 total_size:8589933568 alloc_counter:0
Free_list_11:[elem0-size:536870848]->[elem1-size:536870848]->[elem2-size:536870848]->[elem3-size:536870848]->[elem4-size:536870848]->[elem5-size:536870848]->[elem6-size:536870848]->[elem7-size:536870848]->[elem8-size:536870848]->[elem9-size:536870848]->[elem10-size:536870848]->[elem11-size:536870848]->[elem12-size:536870848]->[elem13-size:536870848]->[elem14-size:536870848]->[elem15-size:536870848]
[root@centos7 dpdk-hugepage_manage]# 
```

[dpdk大页内存实现](https://blog.csdn.net/wangquan1992/article/details/103988667)