# vapp

## virtio & vhost数据流动
以VHOST为例，来解释一下数据是如何流动的：

**(1)** client（qemu）创建共享内存，然后通过ioctl与内核通信，告知内核共享内存的信息，这种就是kernel作为server的vhost；或者通过Unix domain来跟其他的进程通信，这叫做vhost-user。下面以Unix domain为例。
Unix domain可以使用sendmsg/recvmsg来传递文件描述符，这样效率更高一些；client创建好共享内存，发送描述符到server，server直接mmap这个描述符就可以了，少了一个open的操作。

**(2)** Client和Server可以有多段共享内存，每段之间不连续。每一段都是一个vring。
**(3)** Client初始化好每一段共享内存vring，Server不用初始化。
**(4)** Client发送vring的desc，avail，used这些地址给server，然后server在重新mmap之后，可以根据这个地址算出desc，avail，used这些在server用户进程的地址，然后就可以直接读写了。注意，由于数据指针是client地址，在Server处理读的时候需要转换。
读写：以net为例，两个vring，一个tx，一个rx
共享内存存放desc，avail，used这些信息，以及avail->idx, used->idx这些信息。
当client写的时候，数据放在vring->last_avail_idx这个描述符里，注意last_avail_idx、last_used_idx这两个值，在client和server看到的是不一样的，各自维护自己的信息，作为链表头尾。添加id到avail里面，shared.avail.idx++。注意，client此处的last_avail_idx指向的是描述符的id，而不是avail数组的id，这个跟Server上的last_avail_idx的意思不一样。为什么这样做呢？
last_avail_idx 表示可用的数据项，可以直接拿出来用，用完之后，取当前的next;
当回收的时候，也就是client在处理used的时候，可以直接插入到last_avail_idx的前面，类似链表插入到表头；
**(5)** Server收到信号后，从自己记录的last_avail_idx开始读数据，读到avail->idx这里，区间就是(server.last_avail_idx, shared.avail.idx)。
**(6)** Server处理每处理完一条请求，自己的 last_avail_idx ++; 同时插入 id 到 used 里面，插入的位置是 shared.used.idx，然后 shared.used.ix+ +。used.idx此时就像avail->idx一样，表示最新的used的位置。
**(7)** Server通知client处理完数据后，client就可以回收used的描述符了，可回收的区间是(client.last_used_idx, shared.used.idx)。
**(8)** Kickfd，callfd都是在client创建，然后通过unix domain发送给server。***client通知server叫kick***。
 
## Compilation is straightforward:
```

$ cd vapp
$ make
```

## To run the vhost-user reference backend:
```

$ ./vhost -s ./vhost.sock
```

```
Breakpoint 1, _set_vring_kick (vhost_server=0x2a200010, msg=0xffffdd402f90) at vhost_server.c:318
318         fprintf(stdout, "%s\n", __FUNCTION__);
(gdb) bt
#0  _set_vring_kick (vhost_server=0x2a200010, msg=0xffffdd402f90) at vhost_server.c:318
#1  0x00000000004052cc in in_msg_server (context=0x2a200010, msg=0xffffdd402f90) at vhost_server.c:399
#2  0x0000000000404020 in receive_sock_server (node=0x2a2017f8) at server.c:139
#3  0x000000000040296c in process_fd_set (fd_list=0x2a2017d0, type=FD_READ, fdset=0xffffdd4031f0) at fd_list.c:116
#4  0x0000000000402aa8 in traverse_fd_list (fd_list=0x2a2017d0) at fd_list.c:142
#5  0x00000000004041b0 in loop_server (server=0x2a2007c0) at server.c:191
#6  0x00000000004053fc in run_vhost_server (vhost_server=0x2a200010) at vhost_server.c:450
#7  0x00000000004015f0 in main (argc=3, argv=0xffffdd403448) at main.c:57
(gdb) 
(gdb) bt
#0  kick (vring_table=0x5880120, v_idx=0) at vring.c:322
#1  0x000000000040538c in poll_server (context=0x5880010) at vhost_server.c:424
#2  0x00000000004041d4 in loop_server (server=0x58807c0) at server.c:193
#3  0x00000000004053fc in run_vhost_server (vhost_server=0x5880010) at vhost_server.c:450
#4  0x00000000004015f0 in main (argc=3, argv=0xffffd3bc6918) at main.c:57
(gdb) q
```

## The reference client can be run like this:


```

$ ./vhost -q ./vhost.sock
```

```
(gdb) bt
#0  poll_client (context=0x430010) at vhost_client.c:184
#1  0x0000000000405b18 in loop_client (client=0x4301e0) at client.c:209
#2  0x00000000004062d0 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:220
#3  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) 

(gdb) bt
#0  _kick_client (node=0x4311f8) at vhost_client.c:158
#1  0x000000000040296c in process_fd_set (fd_list=0x4311e8, type=FD_READ, fdset=0xfffffffff300) at fd_list.c:116
#2  0x0000000000402aa8 in traverse_fd_list (fd_list=0x4311e8) at fd_list.c:142
#3  0x0000000000405af4 in loop_client (client=0x4301e0) at client.c:207
#4  0x00000000004062d0 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:220
#5  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) 

(gdb) bt
#0  set_host_vring (client=0x4301e0, vring=0xffffbb4a0000, index=0) at vring.c:85
#1  0x00000000004030cc in set_host_vring_table (vring_table=0x430128, vring_table_num=2, client=0x4301e0) at vring.c:121
#2  0x0000000000405d78 in init_vhost_client (vhost_client=0x430010) at vhost_client.c:81
#3  0x0000000000406284 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:213
#4  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) 
```

```
(gdb) bt
#0  0x0000ffffbe6b0d24 in read () from /lib64/libc.so.6
#1  0x00000000004021e8 in vhost_user_recv_fds (fd=9, msg=0xfffffffff180, fds=0xfffffffff160, fd_num=0xfffffffff158) at common.c:259
#2  0x00000000004059c8 in vhost_ioctl (client=0x4301e0, request=VHOST_USER_GET_FEATURES) at client.c:175
#3  0x0000000000405d48 in init_vhost_client (vhost_client=0x430010) at vhost_client.c:77
#4  0x0000000000406284 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:213
#5  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) 
```

```
(gdb) bt
#0  0x0000ffffbe6b0d84 in write () from /lib64/libc.so.6
#1  0x0000000000403cb8 in kick (vring_table=0x430138, v_idx=1) at vring.c:325
#2  0x00000000004060c0 in send_packet (vhost_client=0x430010, p=0x406e80 <arp_request>, size=60) at vhost_client.c:143
#3  0x0000000000406218 in poll_client (context=0x430010) at vhost_client.c:192
#4  0x0000000000405b18 in loop_client (client=0x4301e0) at client.c:209
#5  0x00000000004062d0 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:220
#6  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) 
(gdb) bt
#0  0x0000ffffbe6b0d24 in read () from /lib64/libc.so.6
#1  0x000000000040611c in _kick_client (node=0x4311f8) at vhost_client.c:163
#2  0x000000000040296c in process_fd_set (fd_list=0x4311e8, type=FD_READ, fdset=0xfffffffff300) at fd_list.c:116
#3  0x0000000000402aa8 in traverse_fd_list (fd_list=0x4311e8) at fd_list.c:142
#4  0x0000000000405af4 in loop_client (client=0x4301e0) at client.c:207
#5  0x00000000004062d0 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:220
#6  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) c
(gdb) bt
#0  0x0000ffffbe6c0344 in sendmsg () from /lib64/libc.so.6
#1  0x0000000000401f2c in vhost_user_send_fds (fd=9, msg=0xfffffffff180, fds=0xfffffffff160, fd_num=0) at common.c:203
#2  0x0000000000405960 in vhost_ioctl (client=0x4301e0, request=VHOST_USER_SET_OWNER) at client.c:165
#3  0x0000000000405d30 in init_vhost_client (vhost_client=0x430010) at vhost_client.c:76
#4  0x0000000000406284 in run_vhost_client (vhost_client=0x430010) at vhost_client.c:213
#5  0x0000000000401618 in main (argc=3, argv=0xfffffffff558) at main.c:61
(gdb) 
```


# kick and call

```
client.c
Client* new_client(const char* path)
{
    Client* client = (Client*) calloc(1, sizeof(Client));

    //TODO: handle errors here

    strncpy(client->sock_path, path ? path : VHOST_SOCK_NAME, PATH_MAX);
    client->status = INSTANCE_CREATED;

    return client;
}

int vhost_ioctl(Client* client, VhostUserRequest request, ...)

```

```
int set_host_vring(Client* client, struct vhost_vring *vring, int index)
{
    vring->kickfd = eventfd(0, EFD_NONBLOCK);
    vring->callfd = eventfd(0, EFD_NONBLOCK);
    assert(vring->kickfd >= 0);
    assert(vring->callfd >= 0);

    struct vhost_vring_state num = { .index = index, .num = VHOST_VRING_SIZE };
    struct vhost_vring_state base = { .index = index, .num = 0 };
    struct vhost_vring_file kick = { .index = index, .fd = vring->kickfd };
    struct vhost_vring_file call = { .index = index, .fd = vring->callfd };
    struct vhost_vring_addr addr = { .index = index,
            .desc_user_addr = (uintptr_t) &vring->desc,
            .avail_user_addr = (uintptr_t) &vring->avail,
            .used_user_addr = (uintptr_t) &vring->used,
            .log_guest_addr = (uintptr_t) NULL,
            .flags = 0 };

    if (vhost_ioctl(client, VHOST_USER_SET_VRING_NUM, &num) != 0)
        return -1;
    if (vhost_ioctl(client, VHOST_USER_SET_VRING_BASE, &base) != 0)
        return -1;
    if (vhost_ioctl(client, VHOST_USER_SET_VRING_KICK, &kick) != 0)
        return -1;
    if (vhost_ioctl(client, VHOST_USER_SET_VRING_CALL, &call) != 0)
        return -1;
    if (vhost_ioctl(client, VHOST_USER_SET_VRING_ADDR, &addr) != 0)
        return -1;

    return 0;
}
```


#  参考

[UNDERSTANDING VHOST KERNEL IMPLEMENTATION](https://emb-team.com/understanding-how-vhost-kernel-works/)