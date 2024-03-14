#ifndef _IFACE_H
#define _IFACE_H

#include <netinet/in.h>
#include <arpa/inet.h>

/*
 * /sys/class/net/wlp3s0 files:
 *  mtu: 1500
 *  address: b8:88:e3:3d:10:02
 *  ifindex: 2
 *  type: 1 - Ethernet, 772 - loopback
 *  operstate: up/down/unknown
 *  carrier: 0/1
 *  speed: n Mbits/sec
 *  duplex: full/half
 */

enum iface_state {
	IFACE_STATE_UNKNOWN = -1,
	IFACE_STATE_DOWN = 0,
	IFACE_STATE_UP = 1,
};

enum iface_type {
	IFACE_TYPE_UNKNOWN = -1,
	IFACE_TYPE_LOOPBACK = 0,
	IFACE_TYPE_ETHERNET = 1,
	IFACE_TYPE_WIFI = 2,
};

enum iface_duplex {
	IFACE_DUPLEX_UNKNOWN = -1,
	IFACE_DUPLEX_HALF = 0,
	IFACE_DUPLEX_FULL = 1,
};

struct iface {
	char *name; /* enp5s0, wlp3s0 */
	int idx;
	enum iface_type type;
	char *phy; /* For Wi-Fi only */
	unsigned char mac[6];
	int mtu; /* 1-1500 */
	enum iface_state state;
	enum iface_duplex duplex;
	int speed; /* Mbits/sec */
	struct sockaddr_in sa;
	struct iface *pnext;
};

struct iface_stat {
	long rx_bytes, rx_pkts;
	long tx_bytes, tx_pkts;
};

int iface_idx(const char *name);

struct iface *iface_enum(void);

void iface_print(struct iface *iface);

const char *iface_state2str(enum iface_state s);
const char *iface_type2str(enum iface_type t);

void iface_free(struct iface *iface);

int iface_info(const char *name, struct iface *iface);

int iface_state(const char *name);

int iface_stat(const char *name, struct iface_stat *stat);

int iface_up(const char *name);
int iface_down(const char *name);

int iface_add_addr(const char *name, const char *addr);

#endif
