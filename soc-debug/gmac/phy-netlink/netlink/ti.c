#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>

#include <linux/phy.h>
#include <linux/netdevice.h>
#include "ti.h"
#include "dp83867.h"
#include "common.h"
int ti_drv_match(const char * drv_name,struct phy_device *phydev)
{
    return com_drv_match(TI_PHY_DRIVER_NAME,phydev );
}


static int dp83867_read_status(struct phy_device *phydev)
{
	int status = phy_read(phydev, MII_DP83867_PHYSTS);
	int ret;

	ret = genphy_read_status(phydev);
	if (ret)
		return ret;

	if (status < 0)
		return status;

	if (status & DP83867_PHYSTS_DUPLEX)
	{
		//phydev->duplex = DUPLEX_FULL;
	}
	else
	{
		//phydev->duplex = DUPLEX_HALF;

	}
	if (status & DP83867_PHYSTS_1000)
	{
		
                pr_err("1000M speed \n");
		//phydev->speed = SPEED_1000;
	}
	else if (status & DP83867_PHYSTS_100)
	{
                pr_err("100M speed \n");
		//phydev->speed = SPEED_100;
	}
	else
	{
		//phydev->speed = SPEED_10;

	}
	return 0;
}
static int dp83867_show_reg(struct phy_device *phydev)
{
   u32 reg ;
    if (NULL == phydev)
    {
	 pr_err("phydev not set and return \n ");
	 return 0;
    }
   reg  = phy_read(phydev,0x00);
   pr_err("BMCR %x \n",reg);
   reg  = phy_read(phydev,0x01);
   pr_err("BMSR %x \n",reg);
   reg  = phy_read(phydev,0x0A);
   pr_err("1000M remote status reg %x \n",reg);
   reg  = phy_read(phydev,0x0f);
   pr_err("1000M local status reg %x \n",reg);
   reg = phy_read_mmd(phydev, DP83867_DEVADDR, DP83867_RGMIICTL);
   pr_err("DP83867_RGMIICTL %x and enable rgmii %d \n",reg, reg & 0x80);
   reg = phy_read_mmd(phydev, DP83867_DEVADDR, DP83867_RGMIIDCTL);
   pr_err("DP83867_RGMIIDCTL %x \n",reg);
   reg = phy_read(phydev, MII_DP83867_PHYSTS);
   pr_err("MII_DP83867_PHYSTS %x \n",reg);
   reg = phy_read(phydev, DP83867_BISCR);
   pr_err("DP83867_BISCR %x \n",reg);
   reg = phy_read_mmd(phydev,  DP83867_DEVADDR, DP83867_STRAP_STS1);
   pr_err("DP83867_STRAP_STS1 %x \n",reg);
   pr_err("phy state %d \n",phydev->state);
   dp83867_read_status(phydev);
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
int ti_phy_do(struct phy_op_t *phy)
{
#if 1
    int page, reg, val, op;
    struct phy_device *phydev;
    page = phy->page;
    op = phy->op;
    reg = phy->reg;
    val = phy->val;
    phydev = get_phy(phy->name);
    if (SET_GMAC_OP !=op && (NULL == phydev || !ti_drv_match(NULL, phydev)))
    {
         pr_err("phydev not set  or not match driver and return \n ");
         return -1;
    }
    if (WRITE_OP == op)
    {
        if(DP83867_RGMIIDCTL == reg|| DP83867_RGMIICTL == reg)
        {
            phy_write_mmd(phydev, DP83867_DEVADDR, reg, val);
        }
        else if (0x00 ==  reg|| DP83867_BISCR == reg)
        {
            phy_write(phydev,reg, val);
        }
    }
    else if(READ_OP == op)
    {
       dp83867_show_reg(phydev);
    }
    else if (RESET_OP == op)
    {
       pr_err("dp83867 phy reset \n");
       dp83867_phy_reset(phydev);
    }
    else if (SET_GMAC_OP == op)
    {
       //snprintf(g_marvel_gmac_name,sizeof(g_marvel_gmac_name), "eth%d", reg_value);
       //pr_err("set gmac name %s\n", g_marvel_gmac_name);
       //set_gmac_name(g_marvel_gmac_name, &phydev);
    }
#else
    pr_err("%s return  \n",__func__);
    struct net_device * dst_dev;
    dst_dev = dev_get_by_name(&init_net,phy->name);
    if(!dst_dev)
    {
                 pr_err("******** nic %s not exist *** \n", phy->name);
    }
    else
    {
                 pr_err("******** nic %s  exist *** \n", phy->name);
    }
#endif
    return 0;
}
