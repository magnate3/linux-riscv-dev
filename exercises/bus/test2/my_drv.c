#include <linux/device.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/string.h>

//包含总线
extern struct device my_bus;
extern struct bus_type my_bus_type;

static int my_probe(struct device *dev){
    printk("<0>Driver found device which my driver can handle !\n");
    return 0;
}

static int my_remove(struct device *dev){
    printk("<0>Driver found device unpluged !\n");
    return 0;
}
// 驱动结构体
struct device_driver my_driver = {
    .name = "my_dev",        //此处声明了 本驱动程序可以处理的设备 名字
    .bus = &my_bus_type,
    .probe = my_probe,
    .remove = my_remove,
};

ssize_t mydriver_show(struct device_driver *driver,char *buf){
    return sprintf(buf, "%s\n", "This is my driver");
}

#define DRIVER_ATTR(_name, _mode, _show, _store) \
 struct driver_attribute driver_attr_##_name = __ATTR(_name, _mode, _show, _store)
//产生后面的 driver_attr_drv 结构体
static DRIVER_ATTR(drv,S_IRUGO,mydriver_show,NULL);

static int __init my_driver_init(void){
    int ret = 0;

    /* 注册驱动 */
    ret = driver_register(&my_driver);
    if(ret)
        printk("<0>Fail to register driver: my_driver");
    /* 创建属性文件 */
    if(driver_create_file(&my_driver, &driver_attr_drv))
        printk("<0>Fail to create driver file: my_drv");

    return ret;
}

static void my_driver_exit(void){
    driver_remove_file(&my_driver, &driver_attr_drv);
    driver_unregister(&my_driver);
}

module_init(my_driver_init);
module_exit(my_driver_exit);


MODULE_AUTHOR("");
MODULE_LICENSE("GPL");
