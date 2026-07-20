#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>

#include <linux/phy.h>
#include <linux/netdevice.h>
#include "macb.h"
#include "common.h"
#define ETH_NAME "eth1"
char  g_marvel_gmac_name[32] ;
//char * g_marvel_gmac_name[32] = ETH_NAME;

struct phy_device *g_mar_phydev = NULL;
static struct device *dev;
static struct kobject *root, *phy;

static ssize_t dev1_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "gmac name : %s\n", g_marvel_gmac_name);
}

static ssize_t dev1_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count) {
    u32 read, reg_addr, page, reg_value;
    if (NULL == g_mar_phydev)
    {
	 pr_err("phydev not set and return \n ");
	 return count;
    }
    sscanf(buf, "%x,%x,%x,%x",&read, &reg_addr,&page, &reg_value);
    pr_err("read %d, reg add %x ,page %x, reg value %x\n",read, reg_addr,page, reg_value);
    if (0== read)
    {
    pr_err("before write %x \n", phy_read_paged(g_mar_phydev, page, reg_addr));
    phy_write_paged(g_mar_phydev,page,reg_addr, reg_value);
    }
    else if(1 == read)
    {
       pr_err("reg add %x ,page %x \n",reg_addr,page);
       pr_err("status %x \n", phy_read_paged(g_mar_phydev, page, reg_addr));
    }
    else if (3 == read)
    {
       pr_err("tracing genphy_soft_reset \n");
       genphy_soft_reset(g_mar_phydev);
    }
    else if (4 == read)
    {
       snprintf(g_marvel_gmac_name,sizeof(g_marvel_gmac_name), "eth%d", reg_value);
       pr_err("set gmac name %s\n", g_marvel_gmac_name);
       set_gmac_name(g_marvel_gmac_name, &g_mar_phydev);
    }
    //phy_write(g_mar_phydev,reg_addr, reg_value);
    //phy_write_paged(g_mar_phydev,
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
    snprintf(g_marvel_gmac_name,sizeof(g_marvel_gmac_name), "%s", ETH_NAME);
    set_gmac_name(ETH_NAME, &g_mar_phydev);
    dev = root_device_register("sysfs_mar_phy");
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
    g_mar_phydev = NULL;
}
module_init(sysfs_demo_init);
module_exit(sysfs_demo_exit);
MODULE_LICENSE("GPL");
