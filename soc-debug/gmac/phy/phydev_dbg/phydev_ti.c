#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>

#include <linux/phy.h>
#include <linux/netdevice.h>
#include "macb.h"
#include "common.h"
#include "dp83867.h"
#define ETH_NAME "eth2"
char  g_ti_gmac_name[32];

struct phy_device *g_ti_phydev = NULL;
static struct device *dev;
static struct kobject *root, *phy;

static ssize_t dev1_show(struct device *dev, struct device_attribute *attr, char *buf) {
   return sprintf(buf, "gmac name : %s\n", g_ti_gmac_name);
}

static int dp83867_show_reg(void)
{
   u32 reg ;
    if (NULL == g_ti_phydev)
    {
	 pr_err("phydev not set and return \n ");
	 return 0;
    }
   reg  = phy_read(g_ti_phydev,0x00);
   pr_err("BMCR %x \n",reg);
   reg  = phy_read(g_ti_phydev,0x01);
   pr_err("BMSR %x \n",reg);
   reg = phy_read_mmd(g_ti_phydev, DP83867_DEVADDR, DP83867_RGMIICTL);
   pr_err("DP83867_RGMIICTL %x \n",reg);
   reg = phy_read_mmd(g_ti_phydev, DP83867_DEVADDR, DP83867_RGMIIDCTL);
   pr_err("DP83867_RGMIIDCTL %x \n",reg);
   reg = phy_read(g_ti_phydev, MII_DP83867_PHYSTS);
   pr_err("MII_DP83867_PHYSTS %x \n",reg);
   reg = phy_read(g_ti_phydev, DP83867_BISCR);
   pr_err("DP83867_BISCR %x \n",reg);
   pr_err("phy state %d \n",g_ti_phydev->state);
   return 0;
}
static int dp83867_phy_reset(struct phy_device *phydev)
{
	int err;

	err = phy_write(phydev, DP83867_CTRL, DP83867_SW_RESTART);
	if (err < 0)
		return err;

	usleep_range(10, 20);

	return phy_modify(phydev, MII_DP83867_PHYCTRL,
			 DP83867_PHYCR_FORCE_LINK_GOOD, 0);
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
        if(DP83867_RGMIIDCTL == reg_addr || DP83867_RGMIICTL == reg_addr)
        {
            phy_write_mmd(g_ti_phydev, DP83867_DEVADDR, reg_addr, reg_value);
        }
        else if (0x00 ==  reg_addr || DP83867_BISCR == reg_addr)
        {
            phy_write(g_ti_phydev,reg_addr, reg_value);
        }
    }
    else if(1 == read)
    {
	 dp83867_show_reg();
    }
    else if (3 == read)
    {
       pr_err("tracing dp83867_phy_reset \n");
       //genphy_soft_reset(g_ti_phydev);
       dp83867_phy_reset(g_ti_phydev);
    }
    else if (4 == read)
    {
       snprintf(g_ti_gmac_name,sizeof(g_ti_gmac_name), "eth%d", reg_value);
       pr_err("set gmac name %s\n", g_ti_gmac_name);
       set_gmac_name(g_ti_gmac_name, &g_ti_phydev);
    }
    return count;

}

static struct device_attribute dev_attr = __ATTR(dev1, 0660, dev1_show, dev1_store);

static int sysfs_ti_init(void) {
    snprintf(g_ti_gmac_name,sizeof(g_ti_gmac_name), "%s", ETH_NAME);
    set_gmac_name(ETH_NAME, &g_ti_phydev);
    dev = root_device_register("sysfs_ti_phy");
    root = &dev->kobj;
    phy = kobject_create_and_add("phyDbg", root);
    sysfs_create_file(phy, &dev_attr.attr);
    return 0;
}

static void sysfs_ti_exit(void) {
    printk(KERN_INFO "sysfs demo exit\n");
    sysfs_remove_file(root, &dev_attr.attr);
    kobject_put(phy);
    root_device_unregister(dev);
    g_ti_phydev = NULL;
}
module_init(sysfs_ti_init);
module_exit(sysfs_ti_exit);
MODULE_LICENSE("GPL");
