#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/io.h>
#include <linux/uio_driver.h>
#include <linux/of_device.h>

static struct uio_info the_uio_info;

static int my_prove(struct platform_device *pdev){
    int ret;
    struct resource *res;
    struct device *dev = &pdev->dev;
    void __iomem *g_ioremap_addr;

    dev_info(dev, "platform_probe enter\n");

    res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
    if(!res){
        dev_err(dev, "IORESOURCE_MEM, 0 does not exist.\n");
        return -EINVAL;
    }
    dev_info(dev, "res->start = 0x%08lx\n", (long unsigned int)res->start);
    dev_info(dev, "res->end = 0x%08lx\n", (long unsigned int)res->end);

    g_ioremap_addr = devm_ioremap(dev, res->start, resource_size(res));
    if(!g_ioremap_addr){
        dev_err(dev, "ioremap failed\n");
        return -ENOMEM;
    }

    the_uio_info.name = "led_uio";
    the_uio_info.version = "1.0";
    the_uio_info.mem[0].memtype = UIO_MEM_PHYS;
    the_uio_info.mem[0].addr = res->start;
    the_uio_info.mem[0].size = resource_size(res);
    the_uio_info.mem[0].name = "uio_driver_hw_resion";
    the_uio_info.mem[0].internal_addr = g_ioremap_addr;

    ret = uio_register_device(&pdev->dev, &the_uio_info);
    if(ret != 0){
        dev_info(dev, "Could not register device \"led_uio\"...");
    }

    return 0;
}

static int my_remove(struct platform_device *pdev){
    uio_unregister_device(&the_uio_info);
    dev_info(&pdev->dev, "platform_remove exit\n");

    return 0;
}

static const struct of_device_id my_of_ids[] = {
    { .compatible = "arrow,UIO" },
    {},
};
MODULE_DEVICE_TABLE(of, my_of_ids);

static struct platform_driver my_platform_driver = {
    .probe = my_prove,
    .remove = my_remove,
    .driver = {
        .name = "UIO",
        .of_match_table = my_of_ids,
        .owner = THIS_MODULE,
    }
};
module_platform_driver(my_platform_driver);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Masaki TSUKADA");
MODULE_DESCRIPTION("This is a UIO platform driver");