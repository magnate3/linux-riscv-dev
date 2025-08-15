#include <linux/netdevice.h>
#include "macb.h"
#include "common.h"
int set_gmac_name(const char * eth_name,struct phy_device **phydev)
{
    struct net_device * dst_dev;
    struct macb *bp ;
    dst_dev = dev_get_by_name(&init_net,eth_name);
    pr_err("******** nic %s phy init  ***", eth_name);
    if(!dst_dev)
    {
	   	 pr_err("******** nic %s not exist *** \n", eth_name);
    }
    else
    {
        bp = netdev_priv(dst_dev);
	*phydev = phy_find_first(bp->mii_bus);
	if (*phydev)
	{
	   	 pr_err("******** nic %s phy find  *** \n", eth_name);
	}
	dev_put(dst_dev);
    }
    return 0;
 }
