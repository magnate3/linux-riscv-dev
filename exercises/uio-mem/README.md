 
 
 # UIO_IRQ_NONE
 
 ```
 struct uio_info kpart_info = {
        .name = "kpart",
        .version = "0.1",
        .irq = UIO_IRQ_NONE,
};
 ```
 
 
 
 
 
 
 # insmod uio_test.ko
 
 ```
 [root@centos7 uio2]# insmod uio_test.ko
[root@centos7 uio2]# 
 ```
 
 ```
 [root@centos7 uio2]# dmesg | tail -n 5
[895048.823956] releasing my uio device
[895048.827524] Un-Registered UIO handler for IRQ=1
[895284.775883] drv_kpart_probe(ffff803fcaba9410)
[895899.225340] drv_kpart_probe(ffff803fcbc1fc10)
[895899.229765] kpart_info.mem[0].addr (cbc19000)
 ```
 
 #  uio_user
 
 ```
 gcc uio_user.c  -o uio_user
 [root@centos7 uio2]# ./uio_user 
The device address 0xffff803fcabae80 (lenth 4)
can be accessed over
logical address 0xffffad960000
You have new mail in /var/spool/mail/root
[root@centos7 uio2]# cat /sys/class/uio/uio0/maps/map0/addr
0xffff803fcabae800
[root@centos7 uio2]# cat /sys/class/uio/uio0/maps/map0/size
0x0000000000000400  //kmalloc(1024,GFP_KERNEL)
[root@centos7 uio2]# 
 ```