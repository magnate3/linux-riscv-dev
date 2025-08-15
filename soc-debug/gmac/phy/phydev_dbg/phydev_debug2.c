#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>

#include <linux/phy.h>
#include <linux/netdevice.h>
#include "macb.h"
#define ETH_NAME "eth1"
char * g_gmac_name = ETH_NAME;

struct phy_device *g_ti_phydev = NULL;
static struct device *dev;
static struct kobject *root, *phy;

static ssize_t dev1_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "gmac name : %s\n", g_gmac_name);
}

static ssize_t dev1_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count) {
    u32 read, reg_addr, page, reg_value;
    if (NULL == g_ti_phydev)
    {
	 pr_err("phydev not set and return \n ");
	 return count;
    }
    sscanf(buf, "%x,%x,%x,%x",&read, &reg_addr,&page, &reg_value);
    pr_err("read %d, reg add %x ,page %x, reg value %x\n",read, reg_addr,page, reg_value);
    if (0== read)
    {
    pr_err("before write %x \n", phy_read_paged(g_ti_phydev, page, reg_addr));
    phy_write_paged(g_ti_phydev,page,reg_addr, reg_value);
    }
    else if(1 == read)
    {
       pr_err("reg add %x ,page %x \n",reg_addr,page);
       pr_err("status %x \n", phy_read_paged(g_ti_phydev, page, reg_addr));
    }
    else if (3 == read)
    {
       pr_err("tracing genphy_soft_reset \n");
       genphy_soft_reset(g_ti_phydev);
    }
    //phy_write(g_ti_phydev,reg_addr, reg_value);
    //phy_write_paged(g_ti_phydev,
 #if 0
	return phy_modify_paged(phydev, MII_phyL_MISC_TEST_PAGE,
				MII_88E6390_TEMP_SENSOR,
				MII_88E6393_TEMP_SENSOR_THRESHOLD_MASK,
				temp << MII_88E6393_TEMP_SENSOR_THRESHOLD_SHIFT);
#endif
    return count;

}

static struct device_attribute dev_attr = __ATTR(dev1, 0660, dev1_show, dev1_store);

static int sysfs_demo_init(void) {
    struct net_device * dst_dev;
    struct macb *bp ;
    dst_dev = dev_get_by_name(&init_net,ETH_NAME);
    pr_err("******** nic %s phy init  ***", ETH_NAME);
    if(!dst_dev)
    {
	   	 pr_err("******** nic %s not exist ***", ETH_NAME);
    }
    else
    {
        bp = netdev_priv(dst_dev);
	g_ti_phydev = phy_find_first(bp->mii_bus);
	if (g_ti_phydev)
	{
	   	 pr_err("******** nic %s phy find  ***", ETH_NAME);
	}
	dev_put(dst_dev);
    }
    dev = root_device_register("sysfs_phy");
    root = &dev->kobj;
    phy = kobject_create_and_add("phyDbg", root);
    sysfs_create_file(phy, &dev_attr.attr);
    return 0;
}

static void sysfs_demo_exit(void) {
    printk(KERN_INFO "sysfs demo exit\n");
    sysfs_remove_file(root, &dev_attr.attr);
    kobject_put(phy);
    root_device_unregister(dev);
    g_ti_phydev = NULL;
}
module_init(sysfs_demo_init);
module_exit(sysfs_demo_exit);
MODULE_LICENSE("GPL");
