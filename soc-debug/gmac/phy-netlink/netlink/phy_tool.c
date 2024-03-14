#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <linux/sockios.h>

#ifndef __GLIBC__
#include <linux/if_arp.h>
#include <linux/if_ether.h>
#endif
#define GOOD_PHY 0
#include "nlcore.h"
#include "netlink_test.h"

int phy_op(struct phy_op_t * phy);
int mdio_read(const char * name, int page, int addr)
{
	return 0;
}
int mdio_write(const char * name, int page, int addr, int value)
{
	return 0;
}
int nlr_init(struct nl_sock * nlsock)
{
        /* Only the first call actually inits. */
        if (nl_open(nlsock, NETLINK_TEST))
        {
                        return -1;
        }
        return 0;
}

void nlr_fin(struct nl_sock * nlsock)
{
     nl_close(nlsock);
}

int phy_op(struct phy_op_t * phy) {
     struct nl_sock nlsock;
     char buf[128],  *p;;
     int total;
     memset(buf, 0, sizeof(buf));
     nlr_init(&nlsock);
     p=nl_pad_msg(&nlsock, phy, sizeof(struct phy_op_t), buf,  MSG_TYPE_PHY, 0);
     total = p - buf;
     nl_send_msg_simple(&nlsock, buf, total);
     nl_recv_msg_simple(&nlsock, 1, NULL, NULL);
     nlr_fin(&nlsock);
     return 0;
}
// ./phy_tool  w enp125s0f1  0x9 0x18 0x18
int main(int argc, char **argv)
{
	int page, addr, dev, val;
	char name[64];
	int skfd = -1;
	struct ifreq ifr;
	struct phy_op_t phy;
	int op = 0;
        memset(name, 0, sizeof(name));
	if(argc < 6) {
		printf("Usage phytool [r/w] [dev] [page] [reg] [val]\n");
		return 0;
	}
#if GOOD_PHY
	/* Open a basic socket. */
	if ((skfd = socket(AF_INET, SOCK_DGRAM,0)) < 0) {
		perror("socket");
		return -1;
	}

	/* Get the vitals from the interface. */
	strncpy(ifr.ifr_name, argv[2], IFNAMSIZ);
	strncpy(name, argv[2], strlen(argv[2]));
	if (ioctl(skfd, SIOCGMIIPHY, &ifr) < 0) {
		printf("SIOCGMIIPHY on '%s' failed: %s\n",
			argv[2], strerror(errno));
		return -1;
	}
#endif
	if(argv[1][0] == 'r') {
		page = strtol(argv[3], NULL, 16);
		addr = strtol(argv[4], NULL, 16);
		//printf("0x%.4x\n", mdio_read(page, addr));
		op = READ_OP;
	}
	else if(argv[1][0] == 'w') {
		page = strtol(argv[3], NULL, 16);
		addr = strtol(argv[4], NULL, 16);
		val = strtol(argv[5], NULL, 16);
	        //printf("dev name %s , page 0x%x, reg: 0x%x, val: 0x%x \n", name, page, addr, val);
		//mdio_write(page, addr, val);
		op = WRITE_OP;

	}
	else if(argv[1][0] == '3') {
		page = strtol(argv[3], NULL, 16);
		addr = strtol(argv[4], NULL, 16);
		val = strtol(argv[5], NULL, 16);
		op = RESET_OP;

	}
	else {
		printf("Fout!\n");
	}
	//printf("dev name %s , page 0x%x, reg: 0x%x, val: 0x%x \n", name, page, addr, val);
	memset(&phy, 0, sizeof(struct phy_op_t));
	strncpy(phy.name, argv[2], strlen(argv[2]));
	phy.reg = addr;
	phy.val = val;
	phy.page = page;
	phy.op= op;
	printf("dev name %s ,op %d,  page 0x%x, reg: 0x%x, val: 0x%x \n", phy.name,phy.op,phy.page, phy.reg, phy.val);
        phy_op(&phy);
#if GOOD_PHY
	close(skfd);
#endif
}
