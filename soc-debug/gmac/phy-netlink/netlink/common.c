#include <linux/netdevice.h>
#include "macb.h"
#include "common.h"
struct phy_device * get_phy(const char * eth_name)
{
    struct net_device * dst_dev;
    struct macb *bp ;
    struct phy_device * phydev = NULL;
    //int addr = 0;
    //u32 phy_id;
    dst_dev = dev_get_by_name(&init_net,eth_name);
    if(!dst_dev)
    {
                 pr_err("******** nic %s not exist *** \n", eth_name);
    }
    else
    {
        bp = netdev_priv(dst_dev);
        phydev = phy_find_first(bp->mii_bus);
        if (phydev)
        {
                 pr_err("******** nic %s phy find  *** \n", eth_name);
        }
        else
        {
                 pr_err("******** nic %s phy not  find  *** \n", eth_name);
        }
        dev_put(dst_dev);
    }
    return phydev;
 }

int com_drv_match(const char * drv_name,struct phy_device *phydev )
{
        if (0 == strncmp(drv_name, phydev->drv->name, strlen(drv_name)))
        {
             pr_err("******** debug drv name  %s  match phydev drv name  *** \n", drv_name);
             return 1;
        }
        else
        {
             pr_err("******** debug drv name  %s  not  match phydev drv name %s  *** \n", drv_name, phydev->drv->name);
             return 0;
        }
        return 0;
}
