/*
 * Due to the bug in the kernel netlink code you can not get addresses
 * of only one interface. Kernel always returns addresses of all interfaces.
 * The same is true when you try to get info about the specified interface.
 * Programs depend on this bug, so it will never be fixed.
 *
 * The same thing with get routes: kernel doesn't do filtering at all.
 * See https://patchwork.ozlabs.org/project/netdev/patch/1325080915.26559.43.camel@hakki
 *
 * So, you should do filtering in the userland.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/socket.h>
#include <net/if.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <linux/if_arp.h>

#include "nlcore.h"
#include "nlroute.h"

static struct nl_sock nlsock;
static int nlr_initialized;

int nlr_init(void)
{
	if (!nlr_initialized) {
		/* Only the first call actually inits. */
		if (nl_open(&nlsock, NETLINK_ROUTE))
			return -1;
	}
	nlr_initialized++;
	return 0;
}

void nlr_fin(void)
{
	if (nlr_initialized == 1) {
		nl_close(&nlsock);
	}
	if (nlr_initialized)
		--nlr_initialized;
}

static char *add_hdr(char *p, void *hdr, int len)
{
	memcpy(p, hdr, len);

	return p + NLMSG_ALIGN(len);
}

static char *add_rta(char *p, int type, int len, void *data)
{
	struct rtattr *rta = (struct rtattr *)p;

	rta->rta_type = type;
	rta->rta_len = RTA_LENGTH(len);
	memcpy(RTA_DATA(p), data, len);

	return p + RTA_SPACE(len);
}

struct iface_name_cb_priv {
	int idx;
	char name[32];
};

static int iface_name_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct ifinfomsg *ifi;
	struct rtattr *rta;
	struct iface_name_cb_priv *priv = (struct iface_name_cb_priv *)_priv;
	int n;

	if (!nlhdr)
		return 0;

	ifi = NLMSG_DATA(nlhdr);

	if (ifi->ifi_index != priv->idx)
		return 0;

	for (rta = IFLA_RTA(ifi), n = RTM_PAYLOAD(nlhdr);
	     RTA_OK(rta, n); rta = RTA_NEXT(rta, n)) {
		if (rta->rta_type == IFLA_IFNAME) {
			strncpy(priv->name, RTA_DATA(rta), sizeof(priv->name));
			break;
		}
	}

	return 0;
}

char *nlr_iface_name(int idx)
{
	char buf[128], *p;
	struct ifinfomsg ifi;
	struct iface_name_cb_priv priv;

	memset(buf, 0, sizeof(buf));
	p = nlmsg_put_hdr(buf, RTM_GETLINK, NLM_F_DUMP);

	memset(&ifi, 0, sizeof(ifi));
	ifi.ifi_index = idx;
	//ifi.ifi_change = 0xffffffff;
	p = add_hdr(p, &ifi, sizeof(ifi));

	if (nl_send_msg(&nlsock, buf, p - buf))
		return NULL;

	priv.idx = idx;
	priv.name[0] = '\0';

	if (nl_recv_msg(&nlsock, RTM_NEWLINK, iface_name_cb, &priv))
		return NULL;

	return strdup(priv.name);
}

struct iface_idx_cb_priv {
	int idx;
	const char *name;
};

static int iface_idx_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct ifinfomsg *ifi;
	struct rtattr *rta;
	struct iface_idx_cb_priv *priv = (struct iface_idx_cb_priv *)_priv;
	int n;

	if (!nlhdr)
		return 0;

	ifi = NLMSG_DATA(nlhdr);

	for (rta = IFLA_RTA(ifi), n = RTM_PAYLOAD(nlhdr);
	     RTA_OK(rta, n); rta = RTA_NEXT(rta, n)) {
		if (rta->rta_type == IFLA_IFNAME) {
			DEBUG("IFLA_NAME: %s", (char *)RTA_DATA(rta));
			if (!strcmp(priv->name, RTA_DATA(rta))) {
				priv->idx = ifi->ifi_index;
				DEBUG("iface %s has idx %d", priv->name,
				      priv->idx);
			}
			break;
		}
	}

	return 0;
}

int nlr_iface_idx(const char *name)
{
	char buf[128], *p;
	struct ifinfomsg ifi;
	struct iface_idx_cb_priv priv;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_GETLINK, NLM_F_DUMP);

	memset(&ifi, 0, sizeof(ifi));
	//ifi.ifi_change = 0xffffffff;
	ifi.ifi_flags = 0xffffffff;
	p = add_hdr(p, &ifi, sizeof(ifi));

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	priv.idx = -1;
	priv.name = name;

	if (nl_recv_msg(&nlsock, RTM_NEWLINK, iface_idx_cb, &priv))
		return -1;

	return priv.idx;
}

void nlr_iface_free(struct nlr_iface *iface)
{
	struct nlr_iface *p;

	while (iface) {
		free(iface->name);
		p = iface->pnext;
		free(iface);
		iface = p;
	}
}

//#define IFF_LOWER_UP (1<<16)

struct iface_cb_priv {
	struct nlr_iface *iface;
	int iface_idx;
	int err;
};

static int ifi_type2nlr_iface_type(int ifi_type)
{
	switch(ifi_type) {
	case ARPHRD_ETHER:
		return NLR_IFACE_TYPE_ETHERNET;
	case ARPHRD_LOOPBACK:
		return NLR_IFACE_TYPE_LOOPBACK;
	case ARPHRD_IEEE80211:
		return NLR_IFACE_TYPE_WIRELESS;
	default:
		DEBUG("unknown iface type: %d.", ifi_type);
		return -1; /* Unknown */
	}
}

static int iface_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct ifinfomsg *ifi;
	struct rtattr *rta;
	struct iface_cb_priv *priv = (struct iface_cb_priv *)_priv;
	int n;
	struct nlr_iface *iface;
	struct rtnl_link_stats *stats;

	if (!nlhdr || priv->err)
		return 0;

	ifi = NLMSG_DATA(nlhdr);

	if (priv->iface_idx >= 0 && priv->iface_idx != ifi->ifi_index)
		return 0;

	iface = calloc(1, sizeof(struct nlr_iface));
	if (!iface) {
		ERRNO("failed to alloc nlr_iface");
		priv->err = 1;
		return 0;
	}

	iface->idx = ifi->ifi_index;
	iface->type = ifi_type2nlr_iface_type(ifi->ifi_type);
	iface->is_up = ifi->ifi_flags & IFF_UP;
	iface->carrier_on = ifi->ifi_flags & IFF_LOWER_UP;
	iface->mtu = -1;
	iface->master_idx = -1;
	iface->link_idx = -1;

	for (rta = IFLA_RTA(ifi), n = RTM_PAYLOAD(nlhdr);
	     RTA_OK(rta, n); rta = RTA_NEXT(rta, n)) {
		if (rta->rta_type == IFLA_IFNAME) {
			iface->name = strdup(RTA_DATA(rta));
		} else if (rta->rta_type == IFLA_MTU) {
			iface->mtu = *(int *)RTA_DATA(rta);
		} else if (rta->rta_type == IFLA_ADDRESS) {
			memcpy(iface->addr, RTA_DATA(rta), 6);
		} else if (rta->rta_type == IFLA_STATS) {
			stats = (struct rtnl_link_stats *)RTA_DATA(rta);
			iface->stats.tx_bytes = stats->tx_bytes;
			iface->stats.tx_packets = stats->tx_packets;
			iface->stats.rx_bytes = stats->rx_bytes;
			iface->stats.rx_packets = stats->rx_packets;
		} else if (rta->rta_type == IFLA_MASTER) {
			iface->master_idx = *(int *)RTA_DATA(rta);
		} else if (rta->rta_type == IFLA_LINK) {
			iface->link_idx = *(int *)RTA_DATA(rta);
		} else if (rta->rta_type == IFLA_LINKINFO) {
			int m;
			struct rtattr *rta2;
			for (rta2 = RTA_DATA(rta), m = RTA_PAYLOAD(rta);
				RTA_OK(rta2, m); rta2 = RTA_NEXT(rta2, m)) {
				if (rta2->rta_type == IFLA_INFO_KIND) {
					char kind[32];
					int l = RTA_PAYLOAD(rta2);
					if (l > sizeof(kind) + 1) {
						DEBUG("IFLA_INFO_KIND payload is greater then our static buffer");
						continue;
					}
					memcpy(kind, RTA_DATA(rta2), l);
					kind[l] = '\0';
					if (!strcmp(kind, "bridge"))
						iface->type =
							NLR_IFACE_TYPE_BRIDGE;
					else if (!strcmp(kind, "vlan"))
						iface->type =
							NLR_IFACE_TYPE_VLAN;
					else if (!strcmp(kind, "tun"))
						iface->type =
							NLR_IFACE_TYPE_TUNNEL;
					else if (!strcmp(kind, "bond"))
						iface->type =
							NLR_IFACE_TYPE_BONDING;
				} else if (rta2->rta_type == IFLA_INFO_DATA) {
					/* TODO: get VLAN id */
				}
			}
		}
	}

	iface->pnext = priv->iface;
	priv->iface = iface;

	return 0;
}

struct nlr_iface *nlr_iface(int iface_idx, int *err)
{
	char buf[128], *p;
	struct ifinfomsg ifi;
	struct iface_cb_priv priv;

	if (err)
		*err = -1;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_GETLINK, NLM_F_DUMP);

	memset(&ifi, 0, sizeof(ifi));
	//ifi.ifi_change = 0xffffffff;
	p = add_hdr(p, &ifi, sizeof(ifi));

	if (nl_send_msg(&nlsock, buf, p - buf))
		return NULL;

	priv.iface = NULL;
	priv.err = 0;
	priv.iface_idx = iface_idx;

	if (nl_recv_msg(&nlsock, RTM_NEWLINK, iface_cb, &priv))
		return NULL;

	if (priv.err) {
		nlr_iface_free(priv.iface);
		return NULL;
	}

	if (err)
		*err = 0;

	return priv.iface;
}

static int iface_set_flags(int iface_idx, int flags)
{
	char buf[128], *p;
	struct ifinfomsg ifi;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_NEWLINK, NLM_F_ACK);

	memset(&ifi, 0, sizeof(ifi));

	ifi.ifi_index = iface_idx;
	ifi.ifi_flags = flags;
	ifi.ifi_change = 0xffffffff;

	p = add_hdr(p, &ifi, sizeof(ifi));

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}

int nlr_set_iface(int iface_idx, int up)
{
	return iface_set_flags(iface_idx, up ? IFF_UP : 0);
}

static int manage_addr(int iface_idx, in_addr_t addr, int prefix_len,
		       int type, int flags)
{
	char buf[128], *p;
	struct ifaddrmsg ifa;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, type, flags | NLM_F_ACK);

	memset(&ifa, 0, sizeof(ifa));
	ifa.ifa_family = AF_INET;
	ifa.ifa_prefixlen = prefix_len;
	ifa.ifa_scope = RT_SCOPE_UNIVERSE;
	ifa.ifa_index = iface_idx;

	p = add_hdr(p, &ifa, sizeof(ifa));

	p = add_rta(p, IFA_LOCAL, 4, &addr);
	/* p = add_rta(p, IFA_ADDRESS, 4, &addr); */

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}

int nlr_add_addr(int iface_idx, in_addr_t addr, int prefix_len)
{
	return manage_addr(iface_idx, addr, prefix_len, RTM_NEWADDR,
			   NLM_F_CREATE | NLM_F_EXCL);
}

int nlr_del_addr(int iface_idx, in_addr_t addr, int prefix_len)
{
	return manage_addr(iface_idx, addr, prefix_len, RTM_DELADDR, 0);
}

struct addr_cb_priv {
	struct nlr_addr *addr;
	int iface_idx;
	int err;
};

static int addr_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct addr_cb_priv *priv = (struct addr_cb_priv *)_priv;
	struct ifaddrmsg *ifa;
	struct rtattr *rta;
	int n;
	struct in_addr in;
	struct nlr_addr *addr;

	if (!nlhdr || priv->err)
		return 0;

	ifa = NLMSG_DATA(nlhdr);

	if (ifa->ifa_family != AF_INET)
		return 0;

	if (priv->iface_idx >= 0 && priv->iface_idx != ifa->ifa_index)
		return 0;

	for (rta = IFA_RTA(ifa), n = RTM_PAYLOAD(nlhdr);
	     RTA_OK(rta, n); rta = RTA_NEXT(rta, n)) {
		if (rta->rta_type == IFA_ADDRESS) {
			in.s_addr = *(in_addr_t *)RTA_DATA(rta);
			DEBUG("%d %s/%d", ifa->ifa_index,
			       inet_ntoa(in), ifa->ifa_prefixlen);

			addr = malloc(sizeof(struct nlr_addr));
			if (!addr) {
				ERRNO("failed to alloc nlr_addr");
				priv->err = 1;
				return 0;
			}

			addr->iface_idx = ifa->ifa_index;
			addr->addr = *(in_addr_t *)RTA_DATA(rta);
			addr->prefix_len = ifa->ifa_prefixlen;

			addr->pnext = priv->addr;
			priv->addr = addr;
		}
	}

	return 0;
}

void nlr_addr_free(struct nlr_addr *addr)
{
	struct nlr_addr *p;

	while (addr) {
		p = addr->pnext;
		free(addr);
		addr = p;
	}
}

struct nlr_addr *nlr_get_addr(int iface_idx, int *err)
{
	char buf[64], *p;
	struct ifaddrmsg ifa;
	struct addr_cb_priv priv;

	if (err)
		*err = -1;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_GETADDR, NLM_F_DUMP);

	memset(&ifa, 0, sizeof(ifa));
	ifa.ifa_family = AF_INET;
	/* ifa.ifa_index = iface_idx; */

	p = add_hdr(p, &ifa, sizeof(ifa));

	if (nl_send_msg(&nlsock, buf, p - buf))
		return NULL;

	priv.addr = NULL;
	priv.err = 0;
	priv.iface_idx = iface_idx;

	if (nl_recv_msg(&nlsock, RTM_NEWADDR, addr_cb, &priv))
		return NULL;

	if (priv.err) {
		nlr_addr_free(priv.addr);
		return NULL;
	}

	if (err)
		*err = 0;

	return priv.addr;
}

/* Iface must be down? */
int nlr_set_mac_addr(int iface_idx, char addr[6])
{
	char buf[128], *p;
	struct ifinfomsg ifi;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_SETLINK, NLM_F_REQUEST|NLM_F_ACK);

	memset(&ifi, 0, sizeof(ifi));

	ifi.ifi_index = iface_idx;

	p = add_hdr(p, &ifi, sizeof(ifi));

	p = add_rta(p, IFLA_ADDRESS, 6, addr);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}

void nlr_free_routes(struct nlr_route *r)
{
	struct nlr_route *q;

	while (r) {
		q = r->pnext;
		free(r);
		r = q;
	}
}

struct route_cb_priv {
	struct nlr_route *route, *end;
	int err;
};

/*
https://man7.org/linux/man-pages/man7/rtnetlink.7.html
*/
static int route_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct route_cb_priv *priv = (struct route_cb_priv *)_priv;
	struct rtmsg *r;
	struct rtattr *rta;
	int n;
	struct nlr_route *p;

	if (!nlhdr || priv->err)
		return 0;

	r = NLMSG_DATA(nlhdr);

	if (r->rtm_family != AF_INET)
		return 0;

	p = calloc(sizeof(*p), 1);
	if (!p) {
		priv->err = 1;
		return 0;
	}
	p->pnext = NULL;
	if (priv->end) {
		priv->end->pnext = p;
	} else {
		priv->route = p;
	}
	priv->end = p;

	p->table = r->rtm_table;
	p->type = r->rtm_type;
	p->scope = r->rtm_scope;
	p->proto = r->rtm_protocol;
	p->flags = r->rtm_flags;

	for (rta = RTM_RTA(r), n = RTM_PAYLOAD(nlhdr);
	     RTA_OK(rta, n); rta = RTA_NEXT(rta, n)) {
		switch(rta->rta_type) {
		case RTA_GATEWAY:
			p->gw = *(in_addr_t *)RTA_DATA(rta);
			break;
		case RTA_PRIORITY: /* metrics */
			p->metrics = *(uint32_t *)RTA_DATA(rta);
			break;
		case RTA_PREFSRC:
			p->prefsrc = *(in_addr_t *)RTA_DATA(rta);
			break;
		case RTA_METRICS: /* not used */
			break;
		case RTA_DST:
			p->dest = *(in_addr_t *)RTA_DATA(rta);
			p->dest_plen = r->rtm_dst_len;
			break;
		case RTA_TABLE: /* Equals to r->rtm_table? */
			break;
		case RTA_OIF: /* Output interface */
			p->oif = *(int *)RTA_DATA(rta);
			break;
		default:
			break;
		}
	}

	return 0;
}

struct nlr_route *nlr_get_routes(struct nlr_route *filter, int *err)
{
	char buf[64], *p;
	struct rtmsg r;
	struct route_cb_priv priv;
	struct nlr_route *prev, *q;

	if (err)
		*err = -1;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_GETROUTE, NLM_F_DUMP);

	memset(&r, 0, sizeof(r));
	r.rtm_family = AF_INET;

	p = add_hdr(p, &r, sizeof(r));

	if (filter) {
		/* Kernel doesn't do filtering at all :( */
		/*
		if (filter->table >= 0) {
			p = add_rta(p, RTA_TABLE, 4, &filter->table);
			r.rtm_table = filter->table;
		}
		if (filter->type >= 0)
			r.rtm_type = filter->type;
		if (filter->scope >= 0)
			r.rtm_scope = filter->scope;
		if (filter->proto >= 0)
			r.rtm_protocol = filter->proto;
		if (filter->oif >= 0)
			p = add_rta(p, RTA_OIF, sizeof(int), &filter->oif);
		if (filter->gw != INADDR_NONE)
			p = add_rta(p, RTA_GATEWAY, 4, &filter->gw);
		if (filter->dest != INADDR_NONE) {
			p = add_rta(p, RTA_DST, 4, &filter->dest);
			r.rtm_dst_len = filter->dest_plen;
		}
		*/
	}

	if (nl_send_msg(&nlsock, buf, p - buf))
		return NULL;

	priv.route = priv.end = NULL;
	priv.err = 0;

	if (nl_recv_msg(&nlsock, RTM_NEWROUTE, route_cb, &priv))
		return NULL;

	if (priv.err) {
		nlr_free_routes(priv.route);
		priv.route = NULL;
		return NULL;
	}

	if (err)
		*err = 0;

	if (filter) {
		for (prev = NULL, q = priv.route; q; ) {
			if (filter->table >= 0 && q->table != filter->table
			  || filter->type >= 0 && q->type != filter->type
			  || filter->scope >= 0 && q->scope != filter->scope
			  || filter->proto >= 0 && q->proto != filter->proto
			  || filter->gw != INADDR_NONE && q->gw != filter->gw
			  || filter->dest != INADDR_NONE && (q->dest !=
			  filter->dest || q->dest_plen != filter->dest_plen)) {
				/* Delete route from the result list */
				if (prev) {
					prev->pnext = q->pnext;
					free(q);
					q = prev->pnext;
				} else {
					priv.route = q->pnext;
					free(q);
					q = priv.route->pnext;
				}
			} else {
				prev = q;
				q = q->pnext;
			}
		}
	}

	return priv.route;
}

int route_do(int msg_type, in_addr_t dest, int dest_plen, in_addr_t gw)
{
	char buf[128], *p;
	struct rtmsg r;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, msg_type,
		msg_type == RTM_NEWROUTE ? NLM_F_CREATE | NLM_F_EXCL | NLM_F_ACK
		: NLM_F_ACK);

	memset(&r, 0, sizeof(r));

	r.rtm_family = AF_INET;
	r.rtm_table = RT_TABLE_MAIN;
	r.rtm_type = RTN_UNICAST;
	r.rtm_scope = RT_SCOPE_UNIVERSE;
	r.rtm_protocol = RTPROT_STATIC;
	r.rtm_dst_len = dest_plen;

	p = add_hdr(p, &r, sizeof(r));

	p = add_rta(p, RTA_DST, 4, &dest);
	p = add_rta(p, RTA_GATEWAY, 4, &gw);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;
	return nl_wait_ack(&nlsock);
}

int nlr_add_route(in_addr_t dest, int dest_plen, in_addr_t gw)
{
	return route_do(RTM_NEWROUTE, dest, dest_plen, gw);
}

int nlr_del_route(in_addr_t dest, int dest_plen, in_addr_t gw)
{
	return route_do(RTM_DELROUTE, dest, dest_plen, gw);
}

/*
 * ip link set dev eth0 master br0
 * ip link set dev eth0 nomaster
 *
 * To delete master, set @master_idx<0.
 *
 * To add iface to bridge set it a bridge as master.
 * But! VLAN iface has no master iface!
 */
int nlr_set_master(int iface_idx, int master_idx)
{
	char buf[128], *p;
	struct ifinfomsg ifi;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_NEWLINK, NLM_F_ACK);

	memset(&ifi, 0, sizeof(ifi));
	ifi.ifi_family = AF_UNSPEC;
	ifi.ifi_index = iface_idx;

	p = add_hdr(p, &ifi, sizeof(ifi));

	if (master_idx < 0)
		master_idx = 0;
	p = add_rta(p, IFLA_MASTER, 4, &master_idx);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}

/*
 * ip link add link eth0 name eth0.100 type vlan id 100
 */
int nlr_add_vlan(const char *name, int master_idx, int vlan_id)
{
	char buf[128], *p;
	struct ifinfomsg ifi;
	static const char *vlan_type = "vlan";
	uint16_t vlan = vlan_id;

	memset(buf, 0, sizeof(buf));

	/* NLM_F_EXCL -- if it exists, do nothing. */
	p = nlmsg_put_hdr(buf, RTM_NEWLINK,
		NLM_F_CREATE | NLM_F_EXCL | NLM_F_ACK);

	memset(&ifi, 0, sizeof(ifi));
	ifi.ifi_family = AF_UNSPEC;

	p = add_hdr(p, &ifi, sizeof(ifi));

	p = add_rta(p, IFLA_LINK, 4, &master_idx);
	p = add_rta(p, IFLA_IFNAME, strlen(name) + 1, (char *)name);
	struct rtattr *linkinfo = (struct rtattr *)p;
	p = add_rta(p, IFLA_LINKINFO, 0, NULL);
	p = add_rta(p, IFLA_INFO_KIND, strlen(vlan_type), (char *)vlan_type);
	struct rtattr *linkinfo_data = (struct rtattr *)p;
	p = add_rta(p, IFLA_INFO_DATA, 0, NULL);
	p = add_rta(p, IFLA_VLAN_ID, 2, &vlan);
	linkinfo_data->rta_len = p - (char *)linkinfo_data;
	linkinfo->rta_len = p - (char *)linkinfo;

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}

/*
 * ip link add name br0 type bridge
 */
int nlr_add_bridge(const char *name)
{
	char buf[128], *p;
	struct ifinfomsg ifi;
	static const char *bridge_type = "bridge";

	memset(buf, 0, sizeof(buf));

	/* NLM_F_EXCL -- if it exists, do nothing. */
	p = nlmsg_put_hdr(buf, RTM_NEWLINK,
		NLM_F_CREATE | NLM_F_EXCL | NLM_F_ACK);

	memset(&ifi, 0, sizeof(ifi));
	ifi.ifi_family = AF_UNSPEC;

	p = add_hdr(p, &ifi, sizeof(ifi));

	p = add_rta(p, IFLA_IFNAME, strlen(name) + 1, (char *)name);
	struct rtattr *linkinfo = (struct rtattr *)p;
	p = add_rta(p, IFLA_LINKINFO, 0, NULL);
	p = add_rta(p, IFLA_INFO_KIND, strlen(bridge_type),
		(char *)bridge_type);
	linkinfo->rta_len = p - (char *)linkinfo;

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);

}

/* ip link del name br0 */
int nlr_del_iface(int iface_idx)
{
	char buf[128], *p;
	struct ifinfomsg ifi;

	memset(buf, 0, sizeof(buf));

	p = nlmsg_put_hdr(buf, RTM_DELLINK, NLM_F_ACK);

	memset(&ifi, 0, sizeof(ifi));
	ifi.ifi_family = AF_UNSPEC;
	ifi.ifi_index = iface_idx;

	p = add_hdr(p, &ifi, sizeof(ifi));

	p = add_rta(p, IFLA_MASTER, 4, &iface_idx);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}

