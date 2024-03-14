#ifndef DBG_TI_H
#define DBG_TI_H
#include "netlink_test.h"
#define TI_PHY_DRIVER_NAME "TI DP83867"
int ti_drv_match(const char * drv_name,struct phy_device *phydev );
int ti_phy_do(struct phy_op_t *phy);
#endif
