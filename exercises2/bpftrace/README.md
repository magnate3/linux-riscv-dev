
# tracepoint:io_uring:io_uring_submit_sqe
```

root@ubuntux86:# bpftrace -e 'tracepoint:io_uring:io_uring_submit_sqe {printf("%s(%d)\n", comm, pid);}'
Attaching 1 probe...
iou-sqp-6236(6236)
iou-sqp-6236(6236)
iou-sqp-6236(6236)
iou-sqp-6236(6236)
iou-sqp-6253(6253)
iou-sqp-6253(6253)
iou-sqp-6253(6253)
iou-sqp-6253(6253)
iou-sqp-6481(6481)
iou-sqp-6481(6481)
iou-sqp-6481(6481)
iou-sqp-6481(6481)
root@ubuntux86:# ps -L --pid  6481
    PID     LWP TTY          TIME CMD
   6481    6481 pts/0    00:00:00 sq_poll
```

# tracepoint:syscalls:sys_enter_io_uring_enter
```
root@ubuntux86:# bpftrace -e 'tracepoint:io_uring:io_uring_enter {printf("%s(%d)\n", comm, pid);}'
Attaching 1 probe...
open(/sys/kernel/debug/tracing/events/io_uring/io_uring_enter/id): No such file or directory
Error attaching probe: tracepoint:io_uring:io_uring_enter
root@ubuntux86:# 
```


```

root@ubuntux86:# bpftrace -e 'tracepoint:syscalls:sys_enter_io_uring_enter { printf ("%u , %u, %s(%d)\n",args->fd,args->to_submit,comm, pid); }'
Attaching 1 probe...
3 , 2, sq_poll(6889)
3 , 0, sq_poll(6889)
3 , 0, sq_poll(6889)
root@ubuntux86:#  ps -L --pid  6889
    PID     LWP TTY          TIME CMD
   6889    6889 pts/0    00:00:00 sq_poll
root@ubuntux86:# 
```


```
root@ubuntux86:# bpftrace -e 'tracepoint:syscalls:sys_enter_io_uring_enter { printf ("%u , %u, %s(%d)\n",args->fd,args->to_submit,comm, pid); }'
Attaching 1 probe...
3 , 2, sq_poll(6909)
3 , 0, sq_poll(6909)
3 , 0, sq_poll(6909)
root@ubuntux86:#  bpftrace -e 'tracepoint:io_uring:io_uring_submit_sqe {printf("%s(%d)\n", comm, pid);}'
Attaching 1 probe...
iou-sqp-6909(6909)
iou-sqp-6909(6909)
iou-sqp-6909(6909)
iou-sqp-6909(6909)
root@ubuntux86:# ps -L --pid  6909
    PID     LWP TTY          TIME CMD
   6909    6909 pts/0    00:00:00 sq_poll
root@ubuntux86:# 
```

# madvise

```sh
bpftrance -l

# 探针总数
bpftrace -l  | wc -l

# 各类探针数量
bpftrace -l  | awk -F ":" '{print $1}' | sort | uniq -c

# 使用通配符查询所有的系统调用跟踪点
bpftrace -l 'tracepoint:syscalls:*'
# 使用通配符查询所有名字包含"open"的跟踪点
bpftrace -l '*open*'

# 查询uprobe
bpftrace -l 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:*'

# 查询USDT
bpftrace -l 'usdt:/usr/lib/x86_64-linux-gnu/libc.so.6:*'

# 参数 -v 将会展示探针使用哪些参数，以供内置 args 变量使用
bpftrace -lv 'tracepoint:syscalls:sys_enter_open'
```

```
root@ubuntux86:# bpftrace -l 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:*' | grep mad
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:__GI_madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:__GI___madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:posix_madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:__madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:posix_madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:__madvise
root@ubuntux86:# bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise'
```

```
root@ubuntux86:# bpftrace -lv 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise'
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise
uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise
root@ubuntux86:# bpftrace -lv 'tracepoint:syscalls:sys_enter_open'
tracepoint:syscalls:sys_enter_open
    int __syscall_nr;
    const char * filename;
    int flags;
    umode_t mode;
root@ubuntux86:# 
```

```
root@ubuntux86:# bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise/pid == 1481/{printf("%s",ustack)}'
Attaching 1 probe...
^C

root@ubuntux86:# bpftrace -e 'uprobe:/usr/lib/x86_64-linux-gnu/libc.so.6:madvise/pid==1481/{printf("%s",ustack)}'
Attaching 1 probe...
^C

root@ubuntux86:# 
```

#  bpftrace  profile


监控进程的 CPU 使用率：
```
$ sudo bpftrace -e 'profile:hz:99 { @[pid,execname] = count(); }'
```
这个命令将会每秒钟捕获 99 次时钟中断事件，并记录每个进程的执行名称和 PID。
 