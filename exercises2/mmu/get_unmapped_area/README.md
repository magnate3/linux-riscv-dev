


```
[root@centos7 simple]# cat /proc/devices  | grep simple
240 simple
```


```
mknod /dev/simpler c 240 0
```


或者加上

```
      my_class= class_create(THIS_MODULE, "simple");//建立一个叫simple的内核class，目的是下一步创建设备节点文件

      device_create(my_class, NULL, MKDEV(simple_major, 0),

                     NULL, "led");//创建设备节点文件

```

```
 demo_class = class_create(THIS_MODULE, "simple");
        err = PTR_ERR(demo_class);
        if (IS_ERR(demo_class))
                goto error_class;

        //demo_class->devnode = demo_devnode;

        demo_device = device_create(demo_class, NULL,
                                        MKDEV(simple_major, 0),
                                        NULL, "simpler");
        err = PTR_ERR(demo_device);
        if (IS_ERR(demo_device))
                goto error_device;

```