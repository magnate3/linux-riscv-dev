#ifndef _NL80211_H
#define _NL80211_H

int nl80211_init(void);
void nl80211_fin(void);

struct nl80211_iface_stat {
	long tx_bytes, tx_packets;
	long rx_bytes, rx_packets;
};

struct nl80211_iface {
	int idx;
	int wiphy;
	char *name;
	char *ssid;
	int managed; /* 1 -- managed, 0 -- AP */
	unsigned char mac[6];
	int tx_power; /* dBm */
	int channel; /* 1-14 */
	int freq; /* MHz */
	struct nl80211_iface *pnext;
};

void nl80211_iface_free(struct nl80211_iface *iface);

struct nl80211_iface *nl80211_iface(int iface_idx, int *err);

struct nl80211_iface *nl80211_create_iface(int wiphy, const char *name,
					   int type);

int nl80211_del_iface(int iface_idx);

#define NL80211_IFTYPE_STATION 2
#define NL80211_IFTYPE_AP 3

#define NL80211_CREATE_AP(name) nl80211_create_iface(0, name, NL80211_IFTYPE_AP)

#endif


