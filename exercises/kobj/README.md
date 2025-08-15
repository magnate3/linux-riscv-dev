

# VERIFY_OCTAL_PERMISSIONS(_mode) }
 

修改内核创建/sys文件系统下的文件的权限时

将代码从 __ATTR(type, 0666, xx_show, NULL);
改为 __ATTR(type, 0644, xx_show, NULL);

```
 make[1]: Entering directory '/usr/src/kernels/4.14.0-115.el7a.0.1.aarch64'
  CC [M]  /root/programming/kernel/kobj/kobj_test.o
In file included from ./arch/arm64/include/asm/sysreg.h:538:0,
                 from ./arch/arm64/include/asm/cputype.h:114,
                 from ./arch/arm64/include/asm/cache.h:19,
                 from ./include/linux/cache.h:6,
                 from ./include/linux/printk.h:9,
                 from ./include/linux/kernel.h:14,
                 from ./include/linux/list.h:9,
                 from ./include/linux/kobject.h:20,
                 from /root/programming/kernel/kobj/kobj_test.c:10:
./include/linux/build_bug.h:30:45: error: negative width in bit-field ‘<anonymous>’
 #define BUILD_BUG_ON_ZERO(e) (sizeof(struct { int:(-!!(e)); }))
                                             ^
./include/linux/kernel.h:967:3: note: in expansion of macro ‘BUILD_BUG_ON_ZERO’
   BUILD_BUG_ON_ZERO((perms) & 2) +     \
   ^
./include/linux/sysfs.h:103:12: note: in expansion of macro ‘VERIFY_OCTAL_PERMISSIONS’
    .mode = VERIFY_OCTAL_PERMISSIONS(_mode) },  \
            ^
/root/programming/kernel/kobj/kobj_test.c:42:46: note: in expansion of macro ‘__ATTR’
 static struct kobj_attribute foo_attribute = __ATTR(foo, 0666, foo_show, foo_store);
                                              ^
In file included from ./include/linux/kobject.h:21:0,
                 from /root/programming/kernel/kobj/kobj_test.c:10:
./include/linux/sysfs.h:101:45: error: expected identifier or ‘(’ before ‘{’ token
 #define __ATTR(_name, _mode, _show, _store) {    \
                                             ^
/root/programming/kernel/kobj/kobj_test.c:77:2: note: in expansion of macro ‘__ATTR’
  __ATTR(baz, 0666, b_show, b_store);
  ^
/root/programming/kernel/kobj/kobj_test.c:51:16: warning: ‘b_show’ defined but not used [-Wunused-function]
 static ssize_t b_show(struct kobject *kobj, struct kobj_attribute *attr,
                ^
/root/programming/kernel/kobj/kobj_test.c:63:16: warning: ‘b_store’ defined but not used [-Wunused-function]
 static ssize_t b_store(struct kobject *kobj, struct kobj_attribute *attr,
                ^
```



# /sys/kernel/kobject_example/

```
[root@centos7 kobj]# ls /sys/kernel/kobject_example/
bar  baz  foo
[root@centos7 kobj]# ls /sys/kernel/kobject_example/bar 
/sys/kernel/kobject_example/bar
[root@centos7 kobj]# cat /sys/kernel/kobject_example/bar 
0
[root@centos7 kobj]# cat /sys/kernel/kobject_example/baz
0
[root@centos7 kobj]# echo '1' >  /sys/kernel/kobject_example/baz
[root@centos7 kobj]# echo '2' >  /sys/kernel/kobject_example/baz
[root@centos7 kobj]# cat /sys/kernel/kobject_example/baz
2
[root@centos7 kobj]# cat /sys/kernel/kobject_example/bar 
0
[root@centos7 kobj]# 
```

#  insmod  dev_test.ko

```
[root@centos7 kobj]# ls /sys/devices/virtual/
bdi  dmi  drm  graphics  input  iscsi_transport  mem  misc  mytest_class  net  raw  tty  usbmon  vc  vtconsole  workqueue
[root@centos7 kobj]# ls /sys/devices/virtual/mytest_class/
mytest_device
[root@centos7 kobj]# ls /sys/devices/virtual/mytest_class/mytest_device/
dev  my_device_test  power  subsystem  uevent
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/my_device_test 
123
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/dev 
241:0
[root@centos7 kobj]# ls /dev/* -al  | grep 241
crw------- 1 root root    241,   0 Sep 10 05:37 /dev/mytest_device
crw------- 1 root root     10, 241 Aug 29 05:41 /dev/vhost-vsock
lrwxrwxrwx  1 root root   16 Sep 10 05:37 241:0 -> ../mytest_device
[root@centos7 kobj]# ls /dev/mytest_device 
/dev/mytest_device
[root@centos7 kobj]# ls /dev/mytest_device  -al
crw------- 1 root root 241, 0 Sep 10 05:37 /dev/mytest_device
[root@centos7 kobj]# 
```

```
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/dev 
241:0
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/my_device_test 
123
[root@centos7 kobj]# echo '88888' >   /sys/devices/virtual/mytest_class/mytest_device/my_device_test 
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/my_device_test 
88888

[root@centos7 kobj]# 
```

## rmmod  dev_test.ko 
```
[root@centos7 kobj]# rmmod  dev_test.ko 
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/my_device_test 
cat: /sys/devices/virtual/mytest_class/mytest_device/my_device_test: No such file or directory
[root@centos7 kobj]# 
```

# insmod  dev_test3.ko 

```
[root@centos7 kobj]# insmod  dev_test3.ko 
[root@centos7 kobj]# ls /sys/devices/virtual/m
mem/          misc/         mytest_class/ 
[root@centos7 kobj]# ls /sys/devices/virtual/mytest_class/mytest_device/
dev        my_kobj    power/     subsystem/ uevent     
[root@centos7 kobj]# ls /sys/devices/virtual/mytest_class/mytest_device/
dev  my_kobj  power  subsystem  uevent
[root@centos7 kobj]# ls /sys/devices/virtual/mytest_class/mytest_device/my_kobj 
/sys/devices/virtual/mytest_class/mytest_device/my_kobj
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/my_kobj 
123
[root@centos7 kobj]# echo '8888'  >  /sys/devices/virtual/mytest_class/mytest_device/my_kobj 
-bash: /sys/devices/virtual/mytest_class/mytest_device/my_kobj: Permission denied
[root@centos7 kobj]# cat  /sys/devices/virtual/mytest_class/mytest_device/my_kobj 
```
## container_of(attr, struct hwmon_attr,dev_attr);

```
struct hwmon_attr {
        struct device_attribute dev_attr;
        struct e1000_hw *hw;
        struct e1000_thermal_diode_data *sensor;
        char name[12];
        };
		
/* hwmon callback functions */
static ssize_t ixgbe_hwmon_show_location(struct device *dev,
                                         struct device_attribute *attr,
                                         char *buf)
{
        struct hwmon_attr *ixgbe_attr = container_of(attr, struct hwmon_attr,
                                                     dev_attr);
        return sprintf(buf, "loc%u\n",
                       ixgbe_attr->sensor->location);
}
```

# kobject_put

```
void kobject_put(struct kobject *kobj)
{
	if (kobj) {
		if (!kobj->state_initialized)
			WARN(1, KERN_WARNING "kobject: '%s' (%p): is not "
			       "initialized, yet kobject_put() is being "
			       "called.\n", kobject_name(kobj), kobj);
		kref_put(&kobj->kref, kobject_release);
	}
}

```

# netdev_register_sysfs

```
/* Create sysfs entries for network device. */
int netdev_register_sysfs(struct net_device *net)
{
	struct class_device *class_dev = &(net->class_dev);
	int i;
	struct class_device_attribute *attr;
	int ret;

	class_dev->class = &net_class;
	class_dev->class_data = net;

	strlcpy(class_dev->class_id, net->name, BUS_ID_SIZE);
	if ((ret = class_device_register(class_dev)))
		goto out;

	for (i = 0; (attr = net_class_attributes[i]) != NULL; i++) {
		if ((ret = class_device_create_file(class_dev, attr)))
		    goto out_unreg;
	}


	if (net->get_stats &&
	    (ret = sysfs_create_group(&class_dev->kobj, &netstat_group)))
		goto out_unreg; 

#ifdef WIRELESS_EXT
	if (net->get_wireless_stats &&
	    (ret = sysfs_create_group(&class_dev->kobj, &wireless_group)))
		goto out_cleanup; 

	return 0;
out_cleanup:
	if (net->get_stats)
		sysfs_remove_group(&class_dev->kobj, &netstat_group);
#else
	return 0;
#endif

out_unreg:
	printk(KERN_WARNING "%s: sysfs attribute registration failed %d\n",
	       net->name, ret);
	class_device_unregister(class_dev);
out:
	return ret;
}
```

##  netdev_register_kobject
```
/* Create sysfs entries for network device. */
int netdev_register_kobject(struct net_device *ndev)
{
        struct device *dev = &ndev->dev;
        const struct attribute_group **groups = ndev->sysfs_groups;
        int error = 0;

        device_initialize(dev);
        dev->class = &net_class;
        dev->platform_data = ndev;
        dev->groups = groups;

        dev_set_name(dev, "%s", ndev->name);
```


## to_net_dev

```
static int netdev_hotplug(struct class_device *cd, char **envp,
			  int num_envp, char *buf, int size)
{
	struct net_device *dev = to_net_dev(cd);
	int i = 0;
	int n;

	/* pass interface in env to hotplug. */
	envp[i++] = buf;
	n = snprintf(buf, size, "INTERFACE=%s", dev->name) + 1;
	buf += n;
	size -= n;

	if ((size <= 0) || (i >= num_envp))
		return -ENOMEM;

	envp[i] = NULL;
	return 0;
}
```


#  to_dev kobject

## chardev_sysfs_init

chardev_sysfs_init(device)

```
struct chardev {
   struct cdev cdev;
   struct class *class;
   struct device *dev;
   struct mutex buffer_lock;
   struct mutex num_proc_lock;
   int minor;

   struct {
      u8 max_num_proc:4;
      u8 curr_num_proc:4;
   };

   char buffer[MAX_BUFF_SIZE];
};
```

```
int chardev_sysfs_init(struct chardev *device) {
   int result = 0;

   num_proc_kobj = kobject_create_and_add("exercise_sysfs", &device->dev->kobj);
   if (!num_proc_kobj) {
      printk(KERN_WARNING "Sysfs kobj create failed\n");
      return -EFAULT;
   }

   result = sysfs_create_file(num_proc_kobj, &max_num_proc_attr.attr);
   if (result) {
      printk(KERN_WARNING "Sysfs attribute create failed\n");
      kobject_put(num_proc_kobj);
   }

   return result;
}
```

# device_create_file

```
int device_create_file(struct device *dev,
                       const struct device_attribute *attr)
{
        int error = 0;

        if (dev) {
                WARN(((attr->attr.mode & S_IWUGO) && !attr->store),
int device_create_file(struct device *dev,
                       const struct device_attribute *attr)
{
        int error = 0;

        if (dev) {
                WARN(((attr->attr.mode & S_IWUGO) && !attr->store),
                        "Attribute %s: write permission without 'store'\n",
                        attr->attr.name);
                WARN(((attr->attr.mode & S_IRUGO) && !attr->show),
                        "Attribute %s: read permission without 'show'\n",
                        attr->attr.name);
                error = sysfs_create_file(&dev->kobj, &attr->attr);
        }

        return error;
}
EXPORT_SYMBOL_GPL(device_create_file);

/**
 * device_remove_file - remove sysfs attribute file.
 * @dev: device.
 * @attr: device attribute descriptor.
 */
void device_remove_file(struct device *dev,
                        const struct device_attribute *attr)
{
        if (dev)
                sysfs_remove_file(&dev->kobj, &attr->attr);
}
  
```

