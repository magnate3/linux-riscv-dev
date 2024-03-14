#include <linux/device.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sysfs.h>

#include <linux/phy.h>
#include <linux/netdevice.h>
#include "marvel.h"
#include "common.h"
int marvel_drv_match(const char * drv_name,struct phy_device *phydev)
{
    return com_drv_match(MARVEL_PHY_DRIVER_NAME,phydev );
}

int marvel_phy_do(struct phy_op_t *phy)
{
#if 1
    int page, reg, val, op;
    int ret ;
    struct phy_device *phydev;
    page = phy->page;
    op = phy->op;
    reg = phy->reg;
    val = phy->val;
    phydev = get_phy(phy->name);
    if (SET_GMAC_OP !=op && (NULL == phydev || !marvel_drv_match(NULL, phydev)))
    {
         pr_err("phydev not set  or not match driver and return \n ");
         return -1;
    }
    if (WRITE_OP == op)
    {
        pr_err("before write %x \n", phy_read_paged(phydev, page, reg));
        phy_write_paged(phydev,page,reg, val);
    }
    else if(READ_OP == op)
    {
       pr_err("reg add %x ,page %x \n",reg,page);
       ret = phy_read_paged(phydev, page, reg);
       pr_err("status %x \n", ret);
       phy->val = ret;
    }
    else if (RESET_OP == op)
    {
       pr_err("tracing genphy_soft_reset \n");
       genphy_soft_reset(phydev);
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
