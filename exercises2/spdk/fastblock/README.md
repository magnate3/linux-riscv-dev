

# 关键技术

+ 1 创建pool和image   

```
fastblock-client -op=createpool -poolname=fb -pgcount=128 -pgsize=3
fastblock-client -op=createimage -imagesize=$((100*1024*1024*1024))  -imagename=fbimage -poolname=fb
```
注意，创建pool之后，需要稍微等待raft选出pg的leader才可进行IO和性能测试。   
fastblock-client -op=createimage类似dd image    

+ 2 创建spdk bdev    

基于fastblock-client -op=createimage生成的fbimage通过bdev_fastblock_create创建spdk bdev    

```
spdk/scripts/rpc.py -s /var/tmp/socket.bdev.sock  bdev_fastblock_create -P 1 -p fb -i fbimage -k 4096 -I 100G -m "127.0.0.1:3333" -b fbdev
```

> ## vhost 支持

```
/root/fastblock/build/src/bdev/fastblock-vhost -m ['8'] -C vhost.json &
/root/spdk/scripts/rpc.py -s /var/tmp/socket.bdev.sock  vhost_create_blk_controller --cpumask 0x8 vhost.1 fbdev
```