
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