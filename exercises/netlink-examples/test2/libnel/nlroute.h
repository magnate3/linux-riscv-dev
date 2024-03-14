#ifndef _NLROUTE_H
#define _NLROUTE_H

#include <net/if.h>
#include <arpa/inet.h>
#include <linux/rtnetlink.h>

int nlr_init(void);
void nlr_fin(void);

int nlr_iface_idx(const char *name);
char *nlr_iface_name(int idx);

enum nlr_iface_type {
	NLR_IFACE_TYPE_ETHERNET = 0,
	NLR_IFACE_TYPE_LOOPBACK,
	NLR_IFACE_TYPE_WIRELESS,
	NLR_IFACE_TYPE_BRIDGE,
	NLR_IFACE_TYPE_VLAN,
	NLR_IFACE_TYPE_TUNNEL,
	NLR_IFACE_TYPE_BONDING,
};

struct nlr_iface {
	int idx;
	char *name;
	unsigned char addr[6];
	int mtu;
	enum nlr_iface_type type;
	int is_up;
	int carrier_on;
	int master_idx; /* Master iface index -- iface that gets all
	packets from this iface. E.g. if iface is included in bridge,
	it has an master iface -- bridge iface, the same with bonding.
	A kernel says you: "all packets from this iface I will send not
	to TCP/IP stack, but to this iface/device". */
	int link_idx; /* VLAN ifaces has no master iface (IFLA_MASTER attr),
	but has IFLA_LINK attr -- index of the iface from that it gets
	packets. ip util shows this in this way: "iface@link", e.g. "eth0@100".
	IMPORTANT: in Cisco VLAN iface is a subinterface, not an independent
	interface.
	*/
	union {
		int vlan_id;
	} options;
	struct {
		long tx_bytes, tx_packets;
		long rx_bytes, rx_packets;
	} stats;
	struct nlr_iface *pnext;
};

/*
 * @iface_idx can be <0, in this case return all ifaces.
 * If returns NULL, you can distinguish 'no-ifaces' case and
 * 'error-occured' case by @err value.
 */
struct nlr_iface *nlr_iface(int iface_idx, int *err);

void nlr_iface_free(struct nlr_iface *iface);

int nlr_add_addr(int iface_idx, in_addr_t addr, int prefix_len);
int nlr_del_addr(int iface_idx, in_addr_t addr, int prefix_len);

struct nlr_addr {
	in_addr_t addr;
	int prefix_len;
	struct nlr_addr *pnext;
	int iface_idx;
};

struct nlr_addr *nlr_get_addr(int iface_idx, int *err);

void nlr_addr_free(struct nlr_addr *addr);

int nlr_set_iface(int iface_idx, int up);

#define NLR_IFACE_UP(idx) nlr_set_iface(idx, 1)
#define NLR_IFACE_DOWN(idx) nlr_set_iface(idx, 0)

int nlr_set_mac_addr(int iface_idx, char addr[6]);

struct nlr_route {
	int table;
	int type;
	int scope;
	int proto;
	int metrics;
	unsigned flags;

	in_addr_t dest;
	int dest_plen;
	in_addr_t gw;
	int oif; /* Output interface index */
	in_addr_t prefsrc; /* Preffered source address */

	struct nlr_route *pnext;
};

int nlr_add_route(in_addr_t dest, int dest_plen, in_addr_t gw);
int nlr_del_route(in_addr_t dest, int dest_plen, in_addr_t gw);
/*
 * You can filter what routes you want to get by setting this
 * fields of @filter: @table, @type, @scope, @proto, @dest,
 * @dest_plen, @gw, @oif, @prefsrc. If int field has negative value,
 * it will be treated as unset ("any"). If addr field has INADDR_NONE(-1)
 * value, it will be treated as unset ("any"). @filter can be NULL.
 */
struct nlr_route *nlr_get_routes(struct nlr_route *filter, int *err);
void nlr_free_routes(struct nlr_route *r);

int nlr_add_bridge(const char *name);
int nlr_add_vlan(const char *name, int master_idx, int vlan_id);
int nlr_del_iface(int iface_idx);
/* To unset master, set @master_idx<0. */
int nlr_set_master(int iface_idx, int master_idx);

#endif
