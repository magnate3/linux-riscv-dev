#include <linux/device.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/string.h>


static char *Version = "$LoverXueEr : 1.0 $";

//检测驱动是否匹配设备，dev->bus_id 和 driver->name相等的
static int my_match(struct device *dev ,struct device_driver *driver){
    return !strncmp(dev_name(dev),driver->name,strlen(driver->name));
}

static void my_bus_release(struct device *dev){
    printk("<0>my bus release\n");
}

//设置设备的名字  dev_set_name(&dev,"name");
struct device my_bus = {
    .init_name = "my_bus0",
    .release = my_bus_release,
};

struct bus_type my_bus_type = {
    .name = "my_bus",
    .match = my_match,
};
EXPORT_SYMBOL(my_bus);  //导出符号
EXPORT_SYMBOL(my_bus_type);

//显示总线版本号
static ssize_t show_bus_version(struct bus_type *bus,char *buf){
    return snprintf(buf,PAGE_SIZE,"%s\n",Version);
}

//产生后面的 bus_attr_version 结构体
static BUS_ATTR(version,S_IRUGO, show_bus_version, NULL);

static int __init my_bus_init(void){
    int ret;
    /* 注册总线 */
    ret = bus_register(&my_bus_type);
    if(ret)
        return ret;
    /*  创建属性文件 */
    if(bus_create_file(&my_bus_type, &bus_attr_version))
        printk("<0>Fail to create version attribute! \n");

    /* 注册总线设备 */
    ret = device_register(&my_bus);
    if(ret)
        printk("<0>Fail to register device: my_bus");
    return ret;
}

static void my_bus_exit(void){
    bus_unregister(&my_bus_type);
    device_unregister(&my_bus);
}

module_init(my_bus_init);
module_exit(my_bus_exit);


MODULE_AUTHOR("kk");
MODULE_LICENSE("GPL");