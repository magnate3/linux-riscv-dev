

# pread/pwrite和read/write有什么区别和联系

[pread/pwrite和read/write有什么区别和联系](https://blog.popkx.com/linux-multithreaded-programming-in-io-read-write-security-functions-pread-pwrite-and-read-write-what-is-the-difference-and-relat/)   


# nvme_queue_rq调试

blkdev_ioctl --> 

```
(gdb) bt
#0  nvme_queue_rq (hctx=0xffff8881038dc600, bd=0xffffc900006db6d0) at drivers/nvme/host/pci.c:878
#1  0xffffffff816cb1a6 in __blk_mq_issue_directly (last=true, rq=0xffff888105198000, hctx=0xffff8881038dc600) at block/blk-mq.c:2598
#2  __blk_mq_try_issue_directly (hctx=0xffff8881038dc600, rq=rq@entry=0xffff888105198000, bypass_insert=bypass_insert@entry=false, last=last@entry=true)
    at block/blk-mq.c:2651
#3  0xffffffff816cbfd9 in blk_mq_try_issue_directly (hctx=<optimized out>, rq=0xffff888105198000) at block/blk-mq.c:2674
#4  0xffffffff816ccecd in blk_mq_submit_bio (bio=bio@entry=0xffff88812311ba00) at block/blk-mq.c:3001
#5  0xffffffff816bb322 in __submit_bio (bio=<optimized out>, bio@entry=0xffff88812311ba00) at block/blk-core.c:602
#6  0xffffffff816bb94d in __submit_bio_noacct_mq (bio=0xffff88812311ba00) at block/blk-core.c:679
#7  submit_bio_noacct_nocheck (bio=<optimized out>) at block/blk-core.c:708
#8  submit_bio_noacct_nocheck (bio=<optimized out>) at block/blk-core.c:685
#9  0xffffffff816bbb5d in submit_bio_noacct (bio=bio@entry=0xffff88812311ba00) at block/blk-core.c:807
#10 0xffffffff816bbf77 in submit_bio (bio=bio@entry=0xffff88812311ba00) at block/blk-core.c:843
#11 0xffffffff8148beef in submit_bh_wbc (opf=<optimized out>, opf@entry=0, bh=0xffff888122f61820, wbc=wbc@entry=0x0 <fixed_percpu_data>)
    at fs/buffer.c:2750
#12 0xffffffff8148ed80 in submit_bh (bh=<optimized out>, opf=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>)
    at fs/buffer.c:2755
#13 block_read_full_folio (folio=<optimized out>, get_block=get_block@entry=0xffffffff816b2830 <blkdev_get_block>) at fs/buffer.c:2373
#14 0xffffffff816b28c8 in blkdev_read_folio (file=<optimized out>, folio=<optimized out>) at block/fops.c:396
#15 0xffffffff8132ad54 in filemap_read_folio (file=0xffff8881038dc600, filler=0xffffffff816b28b0 <blkdev_read_folio>, folio=0xffff8881028a6000)
    at mm/filemap.c:2424
#16 0xffffffff8132c431 in do_read_cache_folio (mapping=0xffff888101e41e10, index=index@entry=0, filler=0xffffffff816b28b0 <blkdev_read_folio>, 
    filler@entry=0x0 <fixed_percpu_data>, file=file@entry=0x0 <fixed_percpu_data>, gfp=1051840) at mm/filemap.c:3683
#17 0xffffffff8132c722 in read_cache_folio (mapping=<optimized out>, index=index@entry=0, filler=filler@entry=0x0 <fixed_percpu_data>, 
    file=file@entry=0x0 <fixed_percpu_data>) at ./include/linux/pagemap.h:274
#18 0xffffffff816d9d8a in read_mapping_folio (file=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>, index=0, 
    mapping=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>) at ./include/linux/pagemap.h:775
#19 read_part_sector (state=state@entry=0xffff888106beec00, n=n@entry=0, p=p@entry=0xffffc900006dbb48) at block/partitions/core.c:717
#20 0xffffffff816dffd0 in read_lba (state=state@entry=0xffff888106beec00, lba=lba@entry=0, buffer=buffer@entry=0xffff888106a21600 "", 
    count=count@entry=512) at block/partitions/efi.c:248
#21 0xffffffff816e05bb in find_valid_gpt (ptes=<synthetic pointer>, gpt=<synthetic pointer>, state=0xffff888106beec00) at block/partitions/efi.c:603
#22 efi_partition (state=0xffff888106beec00) at block/partitions/efi.c:720
#23 0xffffffff816d990e in check_partition (hd=0xffff88810253e400, 
    hd@entry=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>) at block/partitions/core.c:146
#24 blk_add_partitions (disk=0xffff88810253e400, disk@entry=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>)
    at block/partitions/core.c:602
#25 bdev_disk_changed (invalidate=<optimized out>, disk=<optimized out>) at block/partitions/core.c:688
#26 bdev_disk_changed (disk=disk@entry=0xffff88810253e400, invalidate=invalidate@entry=false) at block/partitions/core.c:655
#27 0xffffffff816b12a1 in blkdev_get_whole (bdev=bdev@entry=0xffff888101e41900, mode=mode@entry=1212809309) at block/bdev.c:607
#28 0xffffffff816b21c9 in blkdev_get_by_dev (holder=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>, 
    mode=1212809309, mode@entry=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>, dev=<optimized out>, 
    dev@entry=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>) at block/bdev.c:744
#29 blkdev_get_by_dev (dev=<optimized out>, mode=mode@entry=1212809309, holder=holder@entry=0x0 <fixed_percpu_data>) at block/bdev.c:708
#30 0xffffffff816d4f3a in disk_scan_partitions (disk=0xffff88810253e400, mode=mode@entry=1212809309) at ./include/linux/blkdev.h:244
#31 0xffffffff816d3eb9 in blkdev_common_ioctl (bdev=bdev@entry=0xffff888101e41900, mode=mode@entry=1212809309, cmd=cmd@entry=4703, arg=arg@entry=0, 
    argp=argp@entry=0x0 <fixed_percpu_data>) at block/ioctl.c:531
#32 0xffffffff816d4431 in blkdev_ioctl (file=0xffff88810e93b800, cmd=4703, arg=0) at block/ioctl.c:609
#33 0xffffffff814556f3 in vfs_ioctl (arg=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>, cmd=<optimized out>, 
    filp=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>) at fs/ioctl.c:51
#34 __do_sys_ioctl (arg=0, cmd=4703, fd=14) at fs/ioctl.c:870
#35 __se_sys_ioctl (arg=0, cmd=4703, fd=14) at fs/ioctl.c:856
--Type <RET> for more, q to quit, c to continue without paging--
#36 __x64_sys_ioctl (regs=<optimized out>) at fs/ioctl.c:856
#37 0xffffffff8203b2a8 in do_syscall_x64 (nr=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>, 
    regs=<error reading variable: dwarf2_find_location_expression: Corrupted DWARF expression.>) at arch/x86/entry/common.c:50
#38 do_syscall_64 (regs=0xffffc900006dbf58, nr=<optimized out>) at arch/x86/entry/common.c:80
#39 0xffffffff822000aa in entry_SYSCALL_64 () at arch/x86/entry/entry_64.S:120
#40 0xffffffff00000001 in ?? ()
#41 0x00007ffd15909ae8 in ?? ()
#42 0x0000556f9d2f7a20 in ?? ()
#43 0x0000556f9ce99176 in ?? ()
#44 0x00007ffd15909b30 in ?? ()
#45 0x00007ffd15909b20 in ?? ()
#46 0x0000000000000246 in ?? ()
#47 0x0000000000000000 in ?? ()
```