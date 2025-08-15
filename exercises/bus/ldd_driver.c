#include <linux/module.h>
#include <linux/device.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/string.h>
#include "lddbus.h"

static char *Version = "$Revision: 1.9 $";

// Respond to hotplug events.
static int ldd_hotplug(struct device *dev, struct kobj_uevent_env *env)
{
        if (add_uevent_var(env, "LDDBUS_VERSION=%s", Version)) {
                return -ENOMEM;
        }
	return 0;
}

// Match LDD devices to drivers.
static int ldd_match(struct device *dev, struct device_driver *driver)
{
	return !strncmp(dev->init_name, driver->name, strlen(driver->name));
}


// The LDD bus device.
static void ldd_bus_release(struct device *dev)
{
	printk(KERN_DEBUG "lddbus release\n");
}
	
struct device ldd_bus = {
	.init_name   = "ldd0",
	.release  = ldd_bus_release
};


// And the bus type.
struct bus_type ldd_bus_type = {
	.name = "ldd0",
	.match = ldd_match,
	.uevent = ldd_hotplug,
};

// Export a simple attribute.
static ssize_t show_bus_version(struct bus_type *bus, char *buf)
{
	return snprintf(buf, PAGE_SIZE, "%s\n", Version);
}

static BUS_ATTR(version, S_IRUGO, show_bus_version, NULL);


static void ldd_dev_release(struct device *dev)
{ }

int register_ldd_device(struct ldd_device *ldddev)
{
	ldddev->dev.bus = &ldd_bus_type;
	ldddev->dev.parent = &ldd_bus;
	ldddev->dev.release = ldd_dev_release;
	ldddev->dev.init_name = ldddev->name;
	return device_register(&ldddev->dev);
}
EXPORT_SYMBOL(register_ldd_device);

void unregister_ldd_device(struct ldd_device *ldddev)
{
	device_unregister(&ldddev->dev);
}
EXPORT_SYMBOL(unregister_ldd_device);


static ssize_t show_version(struct device_driver *driver, char *buf)
{
	struct ldd_driver *ldriver = to_ldd_driver(driver);

	sprintf(buf, "%s\n", ldriver->version);
	return strlen(buf);
}
		

int register_ldd_driver(struct ldd_driver *driver)
{
	int ret;
	
	driver->driver.bus = &ldd_bus_type;
	ret = driver_register(&driver->driver);
	if (ret)
		return ret;
	driver->version_attr.attr.name = "version";
	driver->version_attr.attr.mode = S_IRUGO;
	driver->version_attr.show = show_version;
	driver->version_attr.store = NULL;
	return driver_create_file(&driver->driver, &driver->version_attr);
}

void unregister_ldd_driver(struct ldd_driver *driver)
{
	driver_unregister(&driver->driver);
}
EXPORT_SYMBOL(register_ldd_driver);
EXPORT_SYMBOL(unregister_ldd_driver);



static int __init ldd_driver_init(void)
{
	int ret;

	ret = bus_register(&ldd_bus_type);
	if (ret)
		return ret;
	if (bus_create_file(&ldd_bus_type, &bus_attr_version))
		printk(KERN_NOTICE "Unable to create version attribute\n");
	ret = device_register(&ldd_bus);
	if (ret)
		printk(KERN_NOTICE "Unable to register ldd0\n");
	return ret;
}

static void ldd_driver_exit(void)
{
	device_unregister(&ldd_bus);
	bus_unregister(&ldd_bus_type);
}

module_init(ldd_driver_init);
module_exit(ldd_driver_exit);

MODULE_AUTHOR("Vegi Mohnish");
MODULE_LICENSE("Dual BSD/GPL");
MODULE_DESCRIPTION("LDD Driver - Bus/Device/Driver interface for scull");
