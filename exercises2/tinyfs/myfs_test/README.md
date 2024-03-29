
# run

```
[root@centos7 myfs2]# rmmod  myfs.ko 
[root@centos7 myfs2]# insmod  myfs.ko 
[root@centos7 myfs2]# mount -t myfs none /mnt/myfs
[root@centos7 myfs2]# touch /mnt/myfs/myfile
[root@centos7 myfs2]# echo 'hellowrld' >  /mnt/myfs/myfile
[root@centos7 myfs2]# cat /mnt/myfs/myfile
hellowrld
[root@centos7 myfs2]#
[root@centos7 myfs2]# umount /mnt/myfs
```

```
[root@centos7 myfs2]# ./mmap_test 
/mnt/myfs/myfile size 196608 
Zero page frame number
virt addr: 0xffff8d470000, phyaddr: 0x0 

Write/Read test ...
0x66616365
0x66616365
0x66616365
[root@centos7 myfs2]# dmesg | tail -n 100
```

> ## simple_readpage
+ 1 generic_file_buffered_read ->  test_simple_readpage   
执行cat  /mnt/myfs/myfile触发    
```
[root@centos7 myfs2]# cat  /mnt/myfs/myfile
facebookfacebookfacebook[root@centos7 myfs2]# 
```

```
[  640.907585] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[  640.912959] [<ffff00000808a074>] show_stack+0x24/0x2c
[  640.917990] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[  640.923021] [<ffff0000008f00a0>] test_simple_readpage+0x20/0x58 [myfs]
[  640.929523] [<ffff00000820cf04>] generic_file_buffered_read+0x1f8/0x768
[  640.936106] [<ffff00000820e180>] generic_file_read_iter+0x11c/0x16c
[  640.942345] [<ffff0000082b235c>] __vfs_read+0x110/0x178
[  640.947547] [<ffff0000082b2454>] vfs_read+0x90/0x14c
[  640.952487] [<ffff0000082b2b70>] SyS_read+0x60/0xc0
```
generic_file_buffered_read调用copy_page_to_iter把page拷贝到用户层的buffer中



+ filemap_fault ->  test_simple_readpage    
```
[  644.839308] [<ffff000008089e14>] dump_backtrace+0x0/0x23c
[  644.844681] [<ffff00000808a074>] show_stack+0x24/0x2c
[  644.849709] [<ffff0000088568a8>] dump_stack+0x84/0xa8
[  644.854739] [<ffff0000008f00a0>] test_simple_readpage+0x20/0x58 [myfs]
[  644.861236] [<ffff00000820d8fc>] filemap_fault+0x37c/0x550
[  644.866700] [<ffff00000824b07c>] __do_fault+0x30/0xf4
[  644.871729] [<ffff00000824faa8>] do_fault+0x4c/0x4b8
[  644.876670] [<ffff0000082514a4>] __handle_mm_fault+0x3f4/0x560
[  644.882477] [<ffff0000082516f0>] handle_mm_fault+0xe0/0x178
[  644.888023] [<ffff000008873a44>] do_page_fault+0x1c4/0x3cc
[  644.893484] [<ffff000008873c9c>] do_translation_fault+0x50/0x5c
[  644.899375] [<ffff0000080813e8>] do_mem_abort+0x64/0xe4
```
> ##  generic_file_mmap
```
[  644.691445] [<ffff0000008f0040>] test_generic_file_mmap+0x20/0x60 [myfs]
[  644.698118] [<ffff000008257c74>] mmap_region+0x348/0x51c
[  644.703405] [<ffff000008258138>] do_mmap+0x2f0/0x34c
[  644.708347] [<ffff00000823467c>] vm_mmap_pgoff+0xf0/0x124
[  644.713722] [<ffff000008255a7c>] SyS_mmap_pgoff+0xc0/0x23c
[  644.719183] [<ffff000008089548>] sys_mmap+0x54/0x68
```

# read和read_iter

```
if (file->f_op->read)
    return file->f_op->read(file, buf, count, pos);
else if (file->f_op->read_iter)
    return new_sync_read(file, buf, count, pos);
```
``read``    
called by read(2) and related system calls

``read_iter``    
possibly asynchronous read with iov_iter as destination