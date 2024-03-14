#ifndef COMMON_H
#define COMMON_H
#include "netlink_test.h"
#include "macb.h"
#include <linux/list.h>
#define MARVEL_PHY_DRIVER_NAME "Marvell 88E1510"
#define TI_PHY_DRIVER_NAME "TI DP83867"
enum
{
PHY_NOT_EXIST = -9999,
PHY_NOT_MATCH,
PHY_PROBE_ERR
};
struct phy_driver_op_t {
int (*drv_match)(const char * drv_name,struct phy_device *phydev );
int (*phy_do)(struct phy_op_t *phy);
struct list_head list;
};
int com_drv_match(const char * drv_name,struct phy_device *phydev );
struct phy_device * get_phy(const char * eth_name);
//struct phy_driver_t{
//     struct phy_driver_op_t *opt;
//     
//}
#endif
