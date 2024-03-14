#ifndef DBG_MARVEL_H
#define DBG_MARVEL_H
#include "netlink_test.h"
int marvel_drv_match(const char * drv_name,struct phy_device *phydev );
int marvel_phy_do(struct phy_op_t *phy);
#endif
