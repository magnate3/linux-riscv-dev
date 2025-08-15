

```
git clone https://github.com/axboe/liburing.git
make -C ./liburing
gcc program.c -o ./program -I./liburing/src/include/ -L./liburing/src/ -Wall -O2 -D_GNU_SOURCE -luring 
gcc io_uring-test.c -o io_uring-test -I./liburing/src/include/ -L./liburing/src/ -Wall -O2 -D_GNU_SOURCE -luring 
gcc -Wall -O2 -D_GNU_SOURCE -o io_uring-test io_uring-test.c -luring -I./liburing/src/include/ -L./liburing/src/ 
```

#  Sample application io_uring-test
io_uring-test reads a maximum of 16KB from a user specified file using 4 SQEs. Each SQE is a request to read 4KB of data from a fixed file offset. io-uring then reaps each CQE and checks whether the full 4KB was read from the file as requested.   

If the file is smaller than 16KB, all 4 SQEs are still submitted but some CQE results will indicate either a partial read, or zero bytes read, depending on the actual size of the file.   

io-uring finally reports the number of SQEs and CQEs it has processed.   

## Description
An io_uring instance is created in the default interrupt driven mode, specifying only the size of the ring.    
```
ret = io_uring_queue_init(QD, &ring, 0);
```
All of the ring SQEs are next fetched and prepared for the IORING_OP_READV operation which provides an asynchronous interface to readv(2) system call. Liburing provides numerous helper functions to prepare io_uring operations.   

Each SQE will point to an allocated buffer described by an iovec structure. The buffer will contain the result of the corresponding readv operation upon completion.   
```
sqe = io_uring_get_sqe(&ring); io_uring_prep_readv(sqe, fd, &iovecs[i], 1, offset);
```
The SQEs are submitted with a single call to io_uring_submit() which returns the number of submitted SQEs.   
```
ret = io_uring_submit(&ring);
```
The CQEs are reaped with repeated calls to io_uring_wait_cqe(), and the success of a given submission is verified with the cqe->res field; each matching call to io_uring_cqe_seen() informs the kernel that the given CQE has been consumed.      
```
ret = io_uring_wait_cqe(&ring, &cqe); io_uring_cqe_seen(&ring, cqe);
```   
The io_uring instance is finally dismantled.
```
void io_uring_queue_exit(struct io_uring *ring)
```
## run  io_uring-test
```
root@ubuntux86:# dd if=/dev/urandom of=test.txt bs=1024 count=1024
1024+0 records in
1024+0 records out
1048576 bytes (1.0 MB, 1.0 MiB) copied, 0.00975142 s, 108 MB/s
root@ubuntux86:# ./io_uring-test   test.txt 
Submitted=4, completed=4
root@ubuntux86:# 
```





# references
[An Introduction to the io_uring Asynchronous I/O Framework](https://medium.com/oracledevs/an-introduction-to-the-io-uring-asynchronous-i-o-framework-fad002d7dfc1)   
[[译] Linux 异步 I/O 框架 io_uring：基本原理、程序示例与性能压测（2020）](http://arthurchiao.art/blog/intro-to-io-uring-zh/)   
[关于 linux io_uring 性能测试 及其 实现原理的一些探索](https://blog.csdn.net/Z_Stand/article/details/120235413)   