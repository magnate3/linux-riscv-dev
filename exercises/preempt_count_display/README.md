# os

```
[root@centos7 kernel]# uname -a
Linux centos7 4.14.0-115.el7a.0.1.aarch64 #1 SMP Sun Nov 25 20:54:21 UTC 2018 aarch64 aarch64 aarch64 GNU/Linux
[root@centos7 kernel]# 
```

# insmod

```
[1186483.426717] preempt_count_test: preempt_count: 0x00000000
[1186483.432280] preempt_count_test: test_preempt_need_resched: 0
[1186483.438086] preempt_count_test: module_init
[1186483.442431] preempt_count_test: preempt_count: 0x00000000
[1186483.447979] preempt_count_test: test_preempt_need_resched: 0
[1186483.453793] preempt_count_test: spin_lock
[1186483.457958] preempt_count_test: preempt_count: 0x00000000
[1186483.463511] preempt_count_test: test_preempt_need_resched: 0
[1186483.469324] preempt_count_test: local_bh_disable
[1186483.474093] preempt_count_test: preempt_count: 0x00000200
[1186483.479647] preempt_count_test: test_preempt_need_resched: 0
[1186483.485454] preempt_count_test: local_bh_disable * 2
[1186483.490574] preempt_count_test: preempt_count: 0x00000400
[1186483.496122] preempt_count_test: test_preempt_need_resched: 0
[1186483.501935] preempt_count_test: local_bh_disable * 3
[1186483.507050] preempt_count_test: preempt_count: 0x00000600
[1186483.512603] preempt_count_test: test_preempt_need_resched: 0
[1186483.518410] preempt_count_test: spin_lock_bh
[1186483.522838] preempt_count_test: preempt_count: 0x00000200
[1186483.528389] preempt_count_test: test_preempt_need_resched: 0
[1186483.534201] preempt_count_test: preempt_disable
[1186483.538884] preempt_count_test: preempt_count: 0x00000000
[1186483.544437] preempt_count_test: test_preempt_need_resched: 0
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/preempt_count_display/nort.png)