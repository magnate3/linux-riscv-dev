#ifndef _NETLINK_TEST_H
#define _NETLINK_TEST_H
 
/* This header is shared with the user-mode application. We must have the same communication interface. */
 
#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
 
typedef unsigned int uint;
#endif
 
/* To get this value, just increase the last value found in the
 *  * defines: https://elixir.bootlin.com/linux/v5.8.16/source/include/uapi/linux/netlink.h#L9
 *   * Just make sure it's not over MAX_LINKS. */
#define NETLINK_COOL 23
#define NETLINK_TEST	30
#define USER_PORT	100
enum {
        MSG_TYPE_STRING = NLMSG_MIN_TYPE, /* Anything below this value it's a control message, and won't be passed to us in kernel. */
        MSG_TYPE_TEST,
        MSG_TYPE_PHY,
        MSG_TYPE_ERROR,
        /* ... */
 
        __MSG_TYPE_MAX
};
enum PHYDEV_OP
{
	     WRITE_OP=0,READ_OP=1,RESET_OP=3,SET_GMAC_OP=4
};
struct phy_op_t{
	char name[64];
        int page;
        int reg;
        int val;
        int op;
};
 
#endif 
