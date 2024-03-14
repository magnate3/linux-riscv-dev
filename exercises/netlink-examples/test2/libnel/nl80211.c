#include <string.h>
#include <syslog.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <linux/nl80211.h>

#include "genlcore.h"
#include "nl80211.h"

static struct nl_sock nlsock;
static int nl80211_id;
static int nl80211_initialized;

int nl80211_init(void)
{
	if (!nl80211_initialized) {
		if (genl_open(&nlsock))
			return -1;

		nl80211_id = genl_service_id(&nlsock, "nl80211");
		if (nl80211_id < 0) {
			ERROR("Failed to get generic netlink service id of \"nl80211\".");
			nl_close(&nlsock);
			return -1;
		}

		DEBUG("nl80211 has id=%d", nl80211_id);
	}

	nl80211_initialized++;

	return 0;
}

void nl80211_fin(void)
{
	if (nl80211_initialized == 1) {
		nl_close(&nlsock);
	}

	if (nl80211_initialized)
		--nl80211_initialized;
}

void nl80211_iface_free(struct nl80211_iface *iface)
{
	struct nl80211_iface *p;

	while (iface) {
		free(iface->name);
		free(iface->ssid);

		p = iface->pnext;
		free(iface);
		iface = p;
	}
}

static int ieee80211_freq_to_channel(int freq)
{
	if (freq == 2484)
		return 14;
	else if (freq < 2484)
		return (freq - 2407) / 5;
	else if (freq >= 4910 && freq <= 4980)
		return (freq - 4000) / 5;
	else
		return (freq - 5000) / 5;
}

struct iface_cb_priv {
	struct nl80211_iface *iface;
	int idx;
	int err;
};

static int iface_cb(struct nlmsghdr *nlhdr, void *_priv)
{
	struct iface_cb_priv *priv = (struct iface_cb_priv *)_priv;
	struct nlattr *nla;
	int n, type;
	struct nl80211_iface iface, *p;

	if (!nlhdr || priv->err)
		return 0;

	memset(&iface, 0, sizeof(iface));
	iface.idx = -1;
	iface.wiphy = -1;
	iface.managed = -1;
	iface.tx_power = -1;
	iface.channel = -1;
	iface.freq = -1;

	for (nla = (struct nlattr *)GENLMSG_DATA(nlhdr), n = GENLMSG_DATA_LEN(nlhdr);
	     NLA_OK(nla, n); nla = NLA_NEXT(nla, n)) {
		if (nla->nla_type == NL80211_ATTR_IFINDEX) {
			iface.idx = *(uint32_t *)NLA_DATA(nla);
		} else if (nla->nla_type == NL80211_ATTR_IFNAME) {
			iface.name = strdup(NLA_DATA(nla));
		} else if (nla->nla_type == NL80211_ATTR_SSID) {
			iface.ssid = strdup(NLA_DATA(nla));
		} else if (nla->nla_type == NL80211_ATTR_WIPHY) {
			iface.wiphy = *(uint32_t *)NLA_DATA(nla);
		} else if (nla->nla_type == NL80211_ATTR_IFTYPE) {
			type = *(uint32_t *)NLA_DATA(nla);
			if (type == NL80211_IFTYPE_AP) {
				iface.managed = 0;
			} else if (type == NL80211_IFTYPE_STATION) {
				iface.managed = 1;
			}
		} else if (nla->nla_type == NL80211_ATTR_MAC) {
			memcpy(iface.mac, NLA_DATA(nla), 6);
		} else if (nla->nla_type == NL80211_ATTR_WIPHY_TX_POWER_LEVEL) {
			iface.tx_power = (*(uint32_t *)NLA_DATA(nla))/100;
		} else if (nla->nla_type == NL80211_ATTR_WIPHY_FREQ) {
			iface.freq = *(uint32_t *)NLA_DATA(nla);
			iface.channel = ieee80211_freq_to_channel(iface.freq);
		}
	}

	if (priv->idx >= 0 && priv->idx != iface.idx) {
		free(iface.name);
		free(iface.ssid);
	} else {
		p = malloc(sizeof(*p));
		if (!p) {
			priv->err = 1;
			nl80211_iface_free(&iface);
		} else {
			memcpy(p, &iface, sizeof(iface));
			p->pnext = priv->iface;
			priv->iface = p;
		}
	}

	return 0;
}

struct nl80211_iface *nl80211_iface(int iface_idx, int *err)
{
	char buf[128], *p;
	struct iface_cb_priv priv;

	if (err)
		*err = -1;

	memset(buf, 0, sizeof(buf));
	p = nlmsg_put_hdr(buf, nl80211_id, NLM_F_DUMP);
	p = genlmsg_put_hdr(p, NL80211_CMD_GET_INTERFACE);
	if (iface_idx >= 0)
		p = genlmsg_add_nla(p, NL80211_ATTR_IFINDEX, 4, &iface_idx);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return NULL;

	priv.iface = NULL;
	priv.err = 0;
	priv.idx = iface_idx;
	if (nl_recv_msg(&nlsock, nl80211_id, iface_cb, &priv))
		return NULL;

	if (priv.err) {
		nl80211_iface_free(priv.iface);
		return NULL;
	}

	if (err)
		*err = 0;

	return priv.iface;
}

/*
 * Possible type values: NL80211_IFTYPE_AP, NL80211_IFTYPE_STATION, ...
 * The first phy has wiphy=0. Return iface idx.
 */
struct nl80211_iface *nl80211_create_iface(int wiphy,
					   const char *name,
					   int type)
{
	char buf[128], *p;
	struct iface_cb_priv priv;

	memset(buf, 0, sizeof(buf));
	p = nlmsg_put_hdr(buf, nl80211_id, NLM_F_REQUEST | NLM_F_ACK);
	p = genlmsg_put_hdr(p, NL80211_CMD_NEW_INTERFACE);
	p = genlmsg_add_nla(p, NL80211_ATTR_WIPHY, 4, &wiphy);
	p = genlmsg_add_nla(p, NL80211_ATTR_IFTYPE, 4, &type);
	p = genlmsg_add_nla(p, NL80211_ATTR_IFNAME, strlen(name) + 1,
			    (char *)name);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return NULL;

	priv.iface = NULL;
	priv.err = 0;
	priv.idx = -1;
	if (nl_recv_msg(&nlsock, nl80211_id, iface_cb, &priv))
		return NULL;

	if (priv.err) {
		nl80211_iface_free(priv.iface);
		return NULL;
	}

	nl_wait_ack(&nlsock);

	return priv.iface;
}

int nl80211_del_iface(int iface_idx)
{
	char buf[128], *p;

	memset(buf, 0, sizeof(buf));
	p = nlmsg_put_hdr(buf, nl80211_id, NLM_F_REQUEST | NLM_F_ACK);
	p = genlmsg_put_hdr(p, NL80211_CMD_DEL_INTERFACE);
	p = genlmsg_add_nla(p, NL80211_ATTR_IFINDEX, 4, &iface_idx);

	if (nl_send_msg(&nlsock, buf, p - buf))
		return -1;

	return nl_wait_ack(&nlsock);
}
