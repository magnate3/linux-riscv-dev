

#  vm_munmap2

```
[root@centos7 vm_munmap2]# insmod  mmap-example.ko 
[root@centos7 vm_munmap2]# ./test 
Changed message: Hello from *user* this is file: mmap-test

Write/Read test ...
[root@centos7 vm_munmap2]# dmesg | tail -n 10
[ 3987.113456] page index offset = 1 
[ 3987.116865] page index offset = 2 
[ 3987.120281] page index offset = 3 
[ 4032.662778] mmap-example: Module exit correctly
[12735.646466] sample char device init
[12735.646651] mmap-example: mmap-test registered with major 241
[12743.319416] page index offset = 0 
[12743.322856] page index offset = 1 
[12743.326264] page index offset = 2 
[12743.329668] page index offset = 3 
[root@centos7 vm_munmap2]# 
```


> ## 虚拟地址分配

```
case IOCTL_ALLOC_VMA:
        if(copy_from_user(&addr,(char*)ioctl_param,sizeof(struct address))){
                pr_err("alloc vma address error\n");
                return ret;
        }
        user_addr = vm_mmap(file, 0, total,
                            PROT_READ | PROT_WRITE | PROT_EXEC,
                            MAP_ANONYMOUS | MAP_PRIVATE, 0);
        if (user_addr >= (unsigned long)(TASK_SIZE)) {
                pr_warn("Failed to allocate user memory\n");
                goto err1;
        }

        vm_area = find_vma(current->mm, user_addr);
        if(NULL == vm_area)
        {
            goto err2;
        }
```

> ## 虚拟地址分配物理内存


```
static int mmap_fault(struct vm_fault *vmf)
{
        struct page *page;
        struct mmap_info *info;
        unsigned long offset;
        struct vm_area_struct *vma = vmf->vma;
        info = (struct mmap_info *)vma->vm_private_data;
        if (!info->data) {
                printk("No data\n");
                return 0;
        }
        offset = ((unsigned long)vmf->address) - ((unsigned long)vmf->vma->vm_start);
        offset = offset >>  PAGE_SHIFT;
        if (offset > (1 << NR_PAGES_ORDER)) {
            printk(KERN_ERR "Invalid address deference, offset = %lu \n",
           offset);
          return 0;
        }
        printk(KERN_ERR "page index offset = %lu \n",offset);
        page = virt_to_page(info->data) + offset;

        get_page(page);
        vmf->page = page;

        return 0;
}
```




# simple

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