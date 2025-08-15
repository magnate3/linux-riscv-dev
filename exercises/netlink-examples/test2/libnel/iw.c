#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <syslog.h>

#include "nl80211.h"
#include "nlroute.h"

static int iface_info(const char *name)
{
	struct nl80211_iface *iface, *p;
	int idx = -1, err;

	if (name) {
		idx = nlr_iface_idx(name);
		if (idx < 0) {
			printf("Failed to get index of %s\n", name);
			return -1;
		}
	}

	iface = nl80211_iface(idx, &err);

	for (p = iface; p; p = p->pnext) {
		printf("\niface %s\n"
		       "      idx: %d\n"
		       "      phy: %d\n"
		       "     ssid: %s\n"
		       "     type: %s\n"
		       "      mac: %02x:%02x:%02x:%02x:%02x:%02x\n"
		       "  txpower: %d dBm\n"
		       "  channel: %d\n"
		       "     freq: %d MHz\n",
		       p->name, p->idx, p->wiphy, p->ssid, p->managed > 0 ? "managed" : !p->managed ? "AP" : "unknown",
		       p->mac[0], p->mac[1], p->mac[2], p->mac[3], p->mac[4], p->mac[5],
		       p->tx_power, p->channel, p->freq
		);
	}

	nl80211_iface_free(iface);

	return err;
}

static int add_ap(const char *ifname)
{
	struct nl80211_iface *iface;

	iface = NL80211_CREATE_AP(ifname);

	if (!iface)
		return -1;

	nl80211_iface_free(iface);

	return 0;
}

static int del_ap(const char *ifname)
{
	int idx;

	idx = nlr_iface_idx(ifname);
	if (idx < 0) {
		printf("Failed to get iface idx\n");
		return -1;
	}

	return nl80211_del_iface(idx);
}

static void help(void)
{
	printf("\nUtil for managing Wi-Fi.\n" \
	       "\nUsage: [options] cmd [cmd-args]" \
	       "\nOptions: -d -- log level info, -d2 --log level debug, -h -- help" \
	       "\n  $ iw [show [IFNAME]]" \
	       "\n  $ iw add ap IFNAME" \
	       "\n  $ iw del iface IFNAME" \
	       "\n" \
	);
}

int main(int argc, char *argv[])
{
	int r = -1, i = 1, logmask = LOG_MASK(LOG_ERR);

	if (argv[i]) {
		if (!strcmp(argv[i], "-d")) {
			logmask |= LOG_MASK(LOG_INFO);
			i++;
		} else if (!strcmp(argv[1], "-d2")) {
			logmask |= LOG_MASK(LOG_INFO) |
				LOG_MASK(LOG_DEBUG);
			/* Enable debugging in the libnel */
			setenv("LIBNEL_DEBUG", "1", 0);
			i++;
		} else if (!strcmp(argv[1], "-h")) {
			help();
			return 0;
		}
	}

	openlog(NULL, LOG_PERROR, LOG_USER);
	setlogmask(logmask);

	if (nl80211_init() || nlr_init())
		goto fin;

	if (!argv[i]) {

		r = iface_info(argv[i]);

	} else if (!strcmp(argv[i], "show")) {

		r = iface_info(argv[i + 1]);

	} else if (!strcmp(argv[i], "add")) {

		if (!argv[i + 1] || strcmp(argv[i + 1], "ap")) {
			printf("Expected 'ap', get '%s'\n", argv[i + 1]);
			goto help;
		}

		if (!argv[i + 2]) {
			printf("Missing iface name\n");
			goto help;
		}

		if (argv[i + 3]) {
			printf("Too many args\n");
			goto help;
		}

		r = add_ap(argv[i + 2]);
	} else if (!strcmp(argv[i], "del")) {

		if (!argv[i + 1] || strcmp(argv[i + 1], "iface")) {
			printf("Expected 'iface', get '%s'\n",
			       argv[i + 1]);
			goto help;
		}

		if (!argv[i + 2]) {
			printf("Missing iface name\n");
			goto help;
		}

		if (argv[i + 3]) {
			printf("Too many args\n");
			goto help;
		}

		r = del_ap(argv[i + 2]);

	} else {
		printf("Unknown command.\n");
		goto help;
	}

fin:
	nl80211_fin();
	nlr_fin();
	closelog();

	printf("%s\n", r ? "Failed" : "OK");

	return r;

help:
	help();
	goto fin;
}
