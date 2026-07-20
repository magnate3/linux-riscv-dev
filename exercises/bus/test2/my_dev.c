#include <linux/device.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/string.h>

//包含总线
extern struct device my_bus;
extern struct bus_type my_bus_type;

static void my_dev_release(struct device *dev){
    printk("<0>my_dev release !\n");
}

//设置设备的名字  dev_set_name(&dev,"name");
struct device my_dev = {
    .bus = &my_bus_type,
    .parent = &my_bus,        //父目录为my_bus
    .release = my_dev_release,
};

ssize_t mydev_show(struct device *dev,struct device_attribute *attr,char *buf){
    return sprintf(buf, "%s\n", "This is my device");
}

//产生后面的 dev_attr_dev 结构体
static DEVICE_ATTR(dev,S_IRUGO,mydev_show,NULL);

static int __init my_dev_init(void){
    int ret = 0;

    /* 初始化设备 以后看驱动与设备是否匹配就看这个名字 */
      dev_set_name(&my_dev,"my_dev");

    /* 注册设备 */
    ret = device_register(&my_dev);
    if(ret)
        printk("<0>Fail to register device: my_dev");
    /* 创建属性文件 */
    if(device_create_file(&my_dev, &dev_attr_dev))
        printk("<0>Fail to create device file: my_dev");

    return ret;
}

static void my_dev_exit(void){
    device_remove_file(&my_dev, &dev_attr_dev);
    device_unregister(&my_dev);
}

module_init(my_dev_init);
module_exit(my_dev_exit);


MODULE_AUTHOR("Lover雪儿");
MODULE_LICENSE("GPL");