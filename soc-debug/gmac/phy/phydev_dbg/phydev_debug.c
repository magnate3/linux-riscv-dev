#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/device.h>
#include <linux/types.h>
#include <linux/cdev.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/phy.h>
#include <linux/netdevice.h>
#include "macb.h"
#define ETH_NAME "eth2"
char * g_gmac_name = "eth2";

struct phy_device *g_ti_phydev = NULL;
static int phy_value;
static ssize_t phy_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
#if 0
   if (!g_ti_phydev)
	return 0;
   u32 reg_addr, page;
   sscanf(buf, "%x,%x", &reg_addr,&page);
   pr_err("reg add %x ,page %x, reg value %x\n",reg_addr,page);
   pr_err("before write %x \n", phy_read_paged(g_ti_phydev, page, reg_addr));
 #endif
   return 0;
}
static ssize_t phy_store(struct kobject *kobj, struct kobj_attribute *attr, const char *buf, size_t count)
{
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
static int  set_g_phydev(char * name)
{
    struct net_device * dst_dev;
    dst_dev = dev_get_by_name(&init_net,name);
    if(!dst_dev)
    {
	     	 pr_err("******** nic %s not exist ***", name);
    }
    return 0;
}
static struct kobj_attribute phy_attribute = __ATTR(phy_value, 0664, phy_show, phy_store);
static struct kobject *phy_kobj = NULL;
static int phy_sys_init(void)
{
    //struct net_device *ndev = phydev->attached_dev
    int retval;
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
    phy_kobj = kobject_create_and_add("phyDbg", kernel_kobj);
    if (!phy_kobj)
        return -ENOMEM;
    retval = sysfs_create_file(phy_kobj, &phy_attribute.attr);
    if (retval)
        kobject_put(phy_kobj);
   return retval;
}
//static void phy_remove(struct phy_device *phydev)
static void phy_remove(void)
{
	//if (0 != strncmp(ETH_NAME, phydev->mdio.bus->id,strlen(ETH_NAME)))
	//{
        //     return;
	//}
	g_ti_phydev = NULL;
	if (!phy_kobj)
            kobject_put(phy_kobj);
}
static int demo_init(void)
{
        int ret = 0;
	ret = phy_sys_init();
	return ret;
}

static void demo_exit(void)
{
    phy_remove();
}

module_init(demo_init);
module_exit(demo_exit);

MODULE_LICENSE("GPL");
 
