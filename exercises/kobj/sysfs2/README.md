
#   platform_device_register_simple(PLAT_NAME, -1, NULL, 0);

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs2/sys1.png)

# sysfs

```
#define OURMODNAME		"sysfs_simple_intf"
#define SYSFS_FILE1		llkdsysfs_debug_level
#define SYSFS_FILE2		llkdsysfs_pgoff
#define SYSFS_FILE3		llkdsysfs_pressure
static DEVICE_ATTR_RO(llkdsysfs_pressure);
static DEVICE_ATTR_RO(llkdsysfs_pgoff);
static DEVICE_ATTR_RW(SYSFS_FILE1);
```

```
root@ubuntux86:/work/kernel_learn/sysfs# insmod  sysfs_test2.ko 
root@ubuntux86:/work/kernel_learn/sysfs# cat  /sys/devices/platform/llkd_sysfs_simple_intf_device/llkdsysfs_debug_level 
0
root@ubuntux86:/work/kernel_learn/sysfs# cat  /sys/devices/platform/llkd_sysfs_simple_intf_device/llkdsysfs_pressure 
25
root@ubuntux86:/work/kernel_learn/sysfs cat  /sys/devices/platform/llkd_sysfs_simple_intf_device/llkdsysfs_pgoff 
0xffff88b480000000
root@ubuntux86:/work/kernel_learn/sysfs# 
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs2/sys2.png)

# 返回值

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs2/ret1.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/kobj/sysfs2/ret2.png)