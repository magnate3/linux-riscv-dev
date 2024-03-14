
# ls /sys/devices/virtual/

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs/sysfs1.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs/sysfs2.png)

```
 demo_class = class_create(THIS_MODULE, "demo_class");
cur->demo_device = device_create(demo_class, NULL, cur->dev, NULL, "demo_device");
```

```
[root@centos7 sysfs]# ls /sys/devices/
armv8_pmuv3_0     hisi_sccl1_ddrc3  hisi_sccl1_l3c12  hisi_sccl3_ddrc1  hisi_sccl3_l3c0  hisi_sccl3_l3c5   hisi_sccl5_hha6   hisi_sccl5_l3c27  hisi_sccl7_ddrc2  hisi_sccl7_l3c17  LNXSYSTM:00  pci0000:7b  pci0000:ba  software
breakpoint        hisi_sccl1_hha2   hisi_sccl1_l3c13  hisi_sccl3_ddrc2  hisi_sccl3_l3c1  hisi_sccl5_ddrc0  hisi_sccl5_hha7   hisi_sccl5_l3c28  hisi_sccl7_ddrc3  hisi_sccl7_l3c18  pci0000:00   pci0000:7c  pci0000:bb  system
hisi_sccl1_ddrc0  hisi_sccl1_hha3   hisi_sccl1_l3c8   hisi_sccl3_ddrc3  hisi_sccl3_l3c2  hisi_sccl5_ddrc1  hisi_sccl5_l3c24  hisi_sccl5_l3c29  hisi_sccl7_hha4   hisi_sccl7_l3c19  pci0000:74   pci0000:80  pci0000:bc  tracepoint
hisi_sccl1_ddrc1  hisi_sccl1_l3c10  hisi_sccl1_l3c9   hisi_sccl3_hha0   hisi_sccl3_l3c3  hisi_sccl5_ddrc2  hisi_sccl5_l3c25  hisi_sccl7_ddrc0  hisi_sccl7_hha5   hisi_sccl7_l3c20  pci0000:78   pci0000:b4  platform    virtual
hisi_sccl1_ddrc2  hisi_sccl1_l3c11  hisi_sccl3_ddrc0  hisi_sccl3_hha1   hisi_sccl3_l3c4  hisi_sccl5_ddrc3  hisi_sccl5_l3c26  hisi_sccl7_ddrc1  hisi_sccl7_l3c16  hisi_sccl7_l3c21  pci0000:7a   pci0000:b8  pnp0
[root@centos7 sysfs]# ls /sys/devices/virtual/
bdi  demo_class  dmi  drm  graphics  input  iscsi_transport  mem  misc  net  raw  tty  usbmon  vc  vtconsole  workqueue
[root@centos7 sysfs]# ls /sys/devices/virtual/demo_class/
demo_device
[root@centos7 sysfs]# ls /sys/devices/virtual/demo_class/demo_device/
dev  power  subsystem  uevent
[root@centos7 sysfs]# 
```

```
[root@centos7 sysfs]# insmod  sys_test1.ko 
[root@centos7 sysfs]# ls /sys/devices/virtual/demo_class/demo_device/
albert  dev  nes  power  subsystem  test  uevent
[root@centos7 sysfs]# cat /sys/devices/virtual/demo_class/demo_device/albert 
[root@centos7 sysfs]# cat /sys/devices/virtual/demo_class/demo_device/dev 
240:0
[root@centos7 sysfs]# cat /sys/devices/virtual/demo_class/demo_device/new
cat: /sys/devices/virtual/demo_class/demo_device/new: No such file or directory
[root@centos7 sysfs]# cat /sys/devices/virtual/demo_class/demo_device/news
cat: /sys/devices/virtual/demo_class/demo_device/news: No such file or directory
[root@centos7 sysfs]# cat /sys/devices/virtual/demo_class/demo_device/nes
[root@centos7 sysfs]# cat /sys/devices/virtual/demo_class/demo_device/test
[root@centos7 sysfs]# 
```

#  insmod  sys_test2.ko 

```
[ 1708.319585] [<ffffffff8074314c>] dev_attr_show+0x24/0x50
[ 1708.319597] [<ffffffff80242094>] sysfs_kf_seq_show+0x90/0xec
[ 1708.319611] [<ffffffff802404dc>] kernfs_seq_show+0x28/0x30
[ 1708.319619] [<ffffffff801d5254>] seq_read_iter+0x14a/0x2da
[ 1708.319637] [<ffffffff8024119e>] kernfs_fop_read_iter+0x120/0x160
[ 1708.319646] [<ffffffff801ad5be>] new_sync_read+0xd4/0x138
[ 1708.319662] [<ffffffff801af062>] vfs_read+0xf2/0x122
[ 1708.319673] [<ffffffff801af36a>] ksys_read+0x5e/0xba
[ 1708.319682] [<ffffffff801af3e8>] sys_read+0x22/0x2a
[ 1708.319692] [<ffffffff8000385e>] ret_from_syscall+0x0/0x2
```

```
[root@centos7 sysfs]# ls /sys/devices/virtual/gko_buffer/
cdev03
[root@centos7 sysfs]# ls /sys/devices/virtual/gko_buffer/cdev03/
buffer_end  buffer_start  dev  power  subsystem  uevent
[root@centos7 sysfs]# cat  /sys/devices/virtual/gko_buffer/cdev03/buffer_end 
65536
[root@centos7 sysfs]# cat  /sys/devices/virtual/gko_buffer/cdev03/buffer_start 
0
[root@centos7 sysfs]# 
```


# insmod  sys_test3.ko 

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs/sysfs3.png)


```

struct hwmon_attr {
        struct device_attribute dev_attr;
        char name[12];
        };

struct demo_dev {
        struct cdev chr_dev;
        struct device *demo_device;
        dev_t dev;
        struct hwmon_attr igb_attr;
};
```

```
        cur->igb_attr.dev_attr.store = albert_store;
        cur->igb_attr.dev_attr.show = albert_show;
        cur->igb_attr.dev_attr.attr.mode = 0444;
        cur->igb_attr.dev_attr.attr.name = "igb_test";
        strncpy(cur->igb_attr.name,"igb_demo", sizeof("igb_demo"));
        ret = device_create_file(cur->demo_device, & cur->igb_attr.dev_attr);
```

```
[ 3998.942100] albert:major = 240,minor = 0
[ 4014.398124] albert:albert_show, name igb_demo
```