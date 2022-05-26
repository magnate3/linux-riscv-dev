/*
* This is simple demon of uio driver.
* Version 1
*Compile:
*    Save this file name it simple.c
*    #echo "obj -m := simple.o" > Makefile
*    #make -Wall -C /lib/modules/'uname -r'/build M='pwd' modules
*Load the module:
*    #modprobe uio
*    #insmod simple.ko
*/



#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/uio_driver.h>
#include <linux/slab.h>


/*struct uio_info { 
    struct uio_device   *uio_dev; // 在__uio_register_device中初始化
    const char      *name; // 调用__uio_register_device之前必须初始化
    const char      *version; //调用__uio_register_device之前必须初始化
    struct uio_mem      mem[MAX_UIO_MAPS];
    struct uio_port     port[MAX_UIO_PORT_REGIONS];
    long            irq; //分配给uio设备的中断号，调用__uio_register_device之前必须初始化
    unsigned long       irq_flags;// 调用__uio_register_device之前必须初始化
    void            *priv; //
    irqreturn_t (*handler)(int irq, struct uio_info *dev_info); //uio_interrupt中调用，用于中断处理
                                                                // 调用__uio_register_device之前必须初始化
    int (*mmap)(struct uio_info *info, struct vm_area_struct *vma); //在uio_mmap中被调用，
                                                                // 执行设备打开特定操作
    int (*open)(struct uio_info *info, struct inode *inode);//在uio_open中被调用，执行设备打开特定操作
    int (*release)(struct uio_info *info, struct inode *inode);//在uio_device中被调用，执行设备打开特定操作
    int (*irqcontrol)(struct uio_info *info, s32 irq_on);//在uio_write方法中被调用，执行用户驱动的
                                                        //特定操作。
};*/

struct uio_info kpart_info = {  
        .name = "kpart",  
        .version = "0.1",  
        .irq = UIO_IRQ_NONE,  
}; 
static int drv_kpart_probe(struct device *dev);
static int drv_kpart_remove(struct device *dev);
static struct device_driver uio_dummy_driver = {
    .name = "kpart",
    .bus = &platform_bus_type,
    .probe = drv_kpart_probe,
    .remove = drv_kpart_remove,
};

static int drv_kpart_probe(struct device *dev)
{
    printk("drv_kpart_probe(%p)\n",dev);
    kpart_info.mem[0].addr = (unsigned long)kmalloc(1024,GFP_KERNEL);
    
    if(kpart_info.mem[0].addr == 0)
        return -ENOMEM;
    printk("kpart_info.mem[0].addr (%x)\n",kpart_info.mem[0].addr);
    kpart_info.mem[0].memtype = UIO_MEM_LOGICAL;
    kpart_info.mem[0].size = 1024;

    if(uio_register_device(dev,&kpart_info))
        return -ENODEV;
    return 0;
}

static int drv_kpart_remove(struct device *dev)
{
    uio_unregister_device(&kpart_info);
    return 0;
}

static struct platform_device * uio_dummy_device;

static int __init uio_kpart_init(void)
{
    uio_dummy_device = platform_device_register_simple("kpart",-1,NULL,0);
    return driver_register(&uio_dummy_driver);
}

static void __exit uio_kpart_exit(void)
{
    platform_device_unregister(uio_dummy_device);
    driver_unregister(&uio_dummy_driver);
}

module_init(uio_kpart_init);
module_exit(uio_kpart_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("IGB_UIO_TEST");
MODULE_DESCRIPTION("UIO dummy driver");
