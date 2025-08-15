/*
 * Legacy: manage network interfaces with ioctls and get network info from sysfs.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <syslog.h>
#include <dirent.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "iface.h"

const char *iface_state2str(enum iface_state s)
{
	static const char *t[] = {
		[IFACE_STATE_UP] = "Up",
		[IFACE_STATE_DOWN] = "Down",
	};

	return s >= 0 && s < sizeof(t)/sizeof(t[0]) && t[s] ? t[s] : "Unknown";
}

const char *iface_type2str(enum iface_type type)
{
	static const char *t[] = {
		[IFACE_TYPE_LOOPBACK] = "Loopback",
		[IFACE_TYPE_ETHERNET] = "Ethernet",
		[IFACE_TYPE_WIFI] = "Wi-Fi",
	};

	return type >= 0 && type < sizeof(t)/sizeof(t[0]) && t[type] ?
		t[type] : "Unknown";
}

void iface_print(struct iface *iface)
{
	printf("\n*** iface %s\n", iface->name);
	printf("  idx: %d\n", iface->idx);
	printf("  state: %s\n", iface_state2str(iface->state));
	printf("  type: %s\n", iface_type2str(iface->type));
	printf("  mtu: %d\n", iface->mtu);
	printf("  addr: %02x:%02x:%02x:%02x:%02x:%02x\n", iface->mac[0],
	       iface->mac[1], iface->mac[2], iface->mac[3], iface->mac[4],
	       iface->mac[5]);
}

void iface_free(struct iface *iface)
{
	struct iface *cur;

	while (iface) {
		cur = iface;
		iface = iface->pnext;
		free(cur->name);
		free(cur->phy);
		free(cur);
	}
}

struct iface *iface_enum(void)
{
	DIR *dir;
	struct dirent *entry;
	struct iface *head = NULL, *iface;

	dir = opendir("/sys/class/net");
	if (!dir)
		return NULL;

	while (entry = readdir(dir)) {
		if (!strcmp(entry->d_name, ".")
		    || !strcmp(entry->d_name, ".."))
			continue;

		iface = calloc(sizeof(*iface), 1);
		if (!iface)
			goto err;

		if (iface_info(entry->d_name, iface)) {
			iface_free(iface);
			goto err;
		}

		iface->pnext = head;
		head = iface;
	}

	closedir(dir);
	return head;

err:
	iface_free(head);
	return NULL;
}

int iface_info(const char *name, struct iface *iface)
{
	FILE *fp;
	char path[64];
	char *p;
	char buf[32];
	int n;
	DIR *dir;
	unsigned mac[6];

	if (strlen(name) > 14)
		return -1;
	p = path + sprintf(path, "/sys/class/net/%s/", name);

	iface->name = strdup(name);
	if (!iface->name)
		return -1;

	iface->mtu = -1;
	strcpy(p, "mtu");
	fp = fopen(path, "r");
	if (fp) {
		fscanf(fp, "%d", &iface->mtu);
		fclose(fp);
	}

	iface->idx = -1;
	strcpy(p, "ifindex");
	fp = fopen(path, "r");
	if (fp) {
		fscanf(fp, "%d", &iface->idx);
		fclose(fp);
	}

	iface->state = -1;
	strcpy(p, "operstate");
	fp = fopen(path, "r");
	if (fp) {
		fscanf(fp, "%32s", buf);
		if (!strcmp(buf, "up"))
			iface->state = IFACE_STATE_UP;
		else if (!strcmp(buf, "down"))
			iface->state = IFACE_STATE_DOWN;
		fclose(fp);
	}

	iface->type = -1;
	strcpy(p, "type");
	fp = fopen(path, "r");
	if (fp) {
		fscanf(fp, "%d", &n);
		switch(n) {
			case 1:
				strcpy(p, "wireless");
				dir = opendir(path);
				if (!dir) {
					iface->type = IFACE_TYPE_ETHERNET;
				} else {
					iface->type = IFACE_TYPE_WIFI;
					closedir(dir);
				}
				break;

			case 772:
				iface->type = IFACE_TYPE_LOOPBACK;
				break;
		}

		fclose(fp);
	}


	strcpy(p, "address");
	fp = fopen(path, "r");
	if (fp) {
		fscanf(fp, "%02x:%02x:%02x:%02x:%02x:%02x", &mac[0],
		       &mac[1], &mac[2], &mac[3], &mac[4], &mac[5]);
		iface->mac[0] = mac[0];
		iface->mac[1] = mac[1];
		iface->mac[2] = mac[2];
		iface->mac[3] = mac[3];
		iface->mac[4] = mac[4];
		iface->mac[5] = mac[5];
		fclose(fp);
	}

	return 0;
}

static int iface_set_flags(const char *name, int flags)
{
	int sock, r;
	struct ifreq ifr;

	memset(&ifr, 0, sizeof(ifr));
	strncpy(ifr.ifr_name, name, IFNAMSIZ);
	ifr.ifr_flags = flags;

	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0)
		return -1;

	r = ioctl(sock, SIOCSIFFLAGS, &ifr);

	close(sock);
	return r;
}

static int flags;

int iface_up(const char *name)
{
	flags |= IFF_UP;
	return iface_set_flags(name, flags);
}

int iface_down(const char *name)
{
	flags &= ~IFF_UP;
	return iface_set_flags(name, flags);
}

int iface_add_addr(const char *name, const char *addr)
{
	int sock, r = -1;
	struct ifreq ifr;
	struct sockaddr_in sa;

	sock = socket(AF_INET, SOCK_DGRAM, 0);
	if (sock < 0)
		return -1;

	memset(&ifr, 0, sizeof(ifr));
	strncpy(ifr.ifr_name, name, IFNAMSIZ);
	memset(&sa, 0, sizeof(sa));
	sa.sin_family = AF_INET;
	sa.sin_port = 0;
	sa.sin_addr.s_addr = inet_addr(addr);
	memcpy(&ifr.ifr_addr, &sa, sizeof(sa));

	r = ioctl(sock, SIOCSIFADDR, &ifr);
	if (r)
		goto fin;

	sa.sin_addr.s_addr = inet_addr("255.255.255.0");
	memcpy(&ifr.ifr_addr, &sa, sizeof(sa));
	r = ioctl(sock, SIOCSIFNETMASK, &ifr);
	if (r)
		goto fin;

	/*
	ifr.ifr_flags |= IFF_UP | IFF_RUNNING;
	if (ioctl(sock, SIOCSIFFLAGS, &ifr))
		goto fin;
	*/

	r = 0;

fin:
	close(sock);
	return r;
}

int iface_idx(const char *name)
{
	FILE *fp;
	char path[256];
	int idx, n;

	sprintf(path, "/sys/class/net/%s/ifindex", name);

	fp = fopen(path, "r");
	if (!fp)
		return -1;

	n = fscanf(fp, "%d", &idx);

	fclose(fp);

	return n == 1 ? idx : -1;
}

