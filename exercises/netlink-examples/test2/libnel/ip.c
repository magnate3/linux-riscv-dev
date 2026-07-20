#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <syslog.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "nlroute.h"

static int stats;

static int set_iface(const char *name, int up)
{
	int idx;

	idx = nlr_iface_idx(name);
	if (idx < 0)
		return -1;

	return nlr_set_iface(idx, up);
}

#define IFACE_IDX_FAILED(ifname) printf("Failed to determine index of \"%s\" iface\n", ifname);

static int set_iface_addr(const char *name, const char * s_addr)
{
	int idx;
	unsigned char addr[6];

	idx = nlr_iface_idx(name);
	if (idx < 0) {
		IFACE_IDX_FAILED(name);
		return -1;
	}

	if (sscanf(s_addr, "%02hhx:%02hhx:%02hhx:%02hhx:%02hhx:%02hhx",
	       &addr[0], &addr[1], &addr[2], &addr[3], &addr[4], &addr[5])
	       != 6) {
	       printf("Invalid format of MAC addr\n");
	       return -1;
	}

	return nlr_set_mac_addr(idx, addr);
}

static int set_iface_master(const char*name, const char *master)
{
	int iface_idx, master_idx;

	iface_idx = nlr_iface_idx(name);
	if (iface_idx < 0) {
		IFACE_IDX_FAILED(name);
		return -1;
	}

	if (master) {
		master_idx = nlr_iface_idx(master);
		if (master_idx < 0) {
			IFACE_IDX_FAILED(master);
			return -1;
		}
	} else {
		master_idx = -1;
	}

	return nlr_set_master(iface_idx, master_idx);
}

static int del_iface(const char *name)
{
	int iface_idx;

	iface_idx = nlr_iface_idx(name);
	if (iface_idx < 0) {
		IFACE_IDX_FAILED(name);
		return -1;
	}

	return nlr_del_iface(iface_idx);
}

static int add_bridge(const char *name)
{
	return nlr_add_bridge(name);
}

static int add_vlan(const char *name, const char *master, const char *s_vlan_id)
{
	int master_idx, vlan_id;
	char c;

	if (sscanf(s_vlan_id, "%d%c", &vlan_id, &c) != 1 || vlan_id < 0) {
		fprintf(stderr, "Invalid vlan id.\n");
		return -1;
	}

	master_idx = nlr_iface_idx(master);
	if (master_idx < 0) {
		IFACE_IDX_FAILED(master);
		return -1;
	}

	return nlr_add_vlan(name, master_idx, vlan_id);
}

/* "x.x.x.x" or "y.y.y.y/n" */
static int parse_addr(const char *s, in_addr_t *addr,
	int *plen)
{
	char buf[32], *p, c;

	strncpy(buf, s, sizeof(buf));
	p = strchr(buf, '/');
	if (!p) {
		*plen = 32;
	} else {
		if (sscanf(p + 1, "%d%c", plen, &c) != 1 ||
		  *plen < 0 || *plen > 32) {
			return -1;
		}
		*p = '\0';
	}
	*addr = inet_addr(buf);
	return (*addr != INADDR_NONE) ? 0 : -1;
}

static int manage_addr(const char *iface, const char *s_addr,
		       int (*f)(int, in_addr_t, int))
{
	in_addr_t addr;
	int iface_idx, plen;

	iface_idx = nlr_iface_idx(iface);
	if (iface_idx < 0) {
		IFACE_IDX_FAILED(iface);
		return -1;
	}

	if (parse_addr(s_addr, &addr, &plen)) {
		printf("Invalid address format\n");
		return -1;
	}

	return f(iface_idx, addr, plen);
}

static int add_addr(const char *iface, const char *addr)
{
	return manage_addr(iface, addr, nlr_add_addr);
}

static int del_addr(const char *iface, const char *addr)
{
	return manage_addr(iface, addr, nlr_del_addr);
}

static int get_addr(const char *iface)
{
	int iface_idx = -1, err;
	char *name;
	struct in_addr in;
	struct nlr_addr *addr, *p;

	if (iface) {
		iface_idx = nlr_iface_idx(iface);
		if (iface_idx < 0) {
			IFACE_IDX_FAILED(iface);
			return -1;
		}
	}

	addr = nlr_get_addr(iface_idx, &err);

	for (p = addr; p; p = p->pnext) {
		name = iface ? (char *)iface : nlr_iface_name(p->iface_idx);
		if (!name) {
			printf("Failed to determine name of iface #%d\n", p->iface_idx);
			continue;
		}

		in.s_addr = p->addr;
		printf("%s %s/%d\n", name, inet_ntoa(in), p->prefix_len);

		if (name != iface)
			free(name);
	}

	nlr_addr_free(addr);

	return err;
}

const char *nlr_iface_type2str(enum nlr_iface_type type)
{
	static const char *t[] = {
		[NLR_IFACE_TYPE_LOOPBACK] = "loopback",
		[NLR_IFACE_TYPE_ETHERNET] = "ethernet",
		[NLR_IFACE_TYPE_WIRELESS] = "wireless",
		[NLR_IFACE_TYPE_BRIDGE] = "bridge",
		[NLR_IFACE_TYPE_VLAN] = "vlan",
		[NLR_IFACE_TYPE_BONDING] = "bonding",
		[NLR_IFACE_TYPE_TUNNEL] = "tunnel",
	};
	static char buf[32];

	if (type >= 0 && type < sizeof(t)/sizeof(t[0]) && t[type])
		return t[type];

	snprintf(buf, sizeof(buf), "(%d)", type);
	return buf;
}

static int get_iface_info(const char *iface_name)
{
	struct nlr_iface *iface, *p;
	int iface_idx = -1, err, n;
	char *master, *link;

	if (iface_name) {
		iface_idx = nlr_iface_idx(iface_name);
		if (iface_idx < 0) {
			IFACE_IDX_FAILED(iface_name);
			return -1;
		}
	}

	iface = nlr_iface(iface_idx, &err);

	for (p = iface, n = 12; p; p = p->pnext) {
		printf("\niface ");
		if (p->link_idx >= 0) {
			/* Mimic ip link show format */
			link = nlr_iface_name(p->link_idx);
			if (link) {
				printf("%s@%s\n", p->name, link);
				free(link);
			} else {
				printf("%s@%d\n", p->name, p->link_idx);
			}
		} else {
			printf("%s\n", p->name);
		}

		if (p->master_idx >= 0) {
			master = nlr_iface_name(p->master_idx);
			printf("%*s: %s\n", n, "master", master);
			free(master);
		}

		printf("%*s: %d\n", n, "idx", p->idx);

		printf("%*s: %s\n", n, "type", nlr_iface_type2str(p->type));

		if (p->mtu > 0)
			printf("%*s: %d\n", n, "mtu", p->mtu);

		printf("%*s: %s\n", n, "admin-state",
			p->is_up ? "up" : "down");

		printf("%*s: %s\n", n, "carrier", p->carrier_on ? "yes" : "no");

		if (p->addr[0] && p->addr[1] && p->addr[2] && p->addr[3]
			&& p->addr[4] && p->addr[5]) {
			printf("%*s: %02x:%02x:%02x:%02x:%02x:%02x\n", n,
				"addr", p->addr[0], p->addr[1], p->addr[2],
				p->addr[3], p->addr[4], p->addr[5]
			);
		}

		if (stats) {
			printf("%*s: %ld\n", n, "tx-bytes", p->stats.tx_bytes);
			printf("%*s: %ld\n", n, "tx-packets",
				p->stats.tx_packets
			);
			printf("%*s: %ld\n", n, "rx-bytes", p->stats.rx_bytes);
			printf("%*s: %ld\n", n, "rx-packets",
				p->stats.rx_packets
			);
		}
	}

	nlr_iface_free(iface);

	return err;
}

/*
 * For known code (>=0) return its name, for unknown code return
 * its numeric value. @t[] -- possible names, sparse array
 * (indexed by code, some elements are NULL). @n -- number of
 * elements in @t[].
 */
static const char *code2name(int code, const char *t[], int n)
{
	static char buf[4];

	if (code < 0 || code >= n || !t[code]) {
		snprintf(buf, sizeof(buf), "%d", code);
		return buf;
	}
	return t[code];
}

/*
 * Return code by its name. If name is unknown, return <0.
 * Support code numeric value in place of name, in this case
 * return a value.
 */
static int name2code(const char *name, const char *t[], int n)
{
	int i;
	char c;

	for (i = 0; i < n && (!t[i] || strcmp(t[i], name)); i++);
	if (i < n)
		return i;
	return i < n || sscanf(name, "%u%c", &i, &c) == 1 ? i : -1;
}

/*
 * Helper function used by code_names() only. Support NULL in place
 * of @e1 and @e2. While sorting NULLs will be placed ath the end of
 * an array.
 */
static int code_names_qsort_cmp(const void *p1, const void *p2)
{
	const char *s1 = *(const char **)p1, *s2 = *(const char **)p2;
	if (!s2)
		return -1;
	if (!s1)
		return 1;
	return strcmp(s1, s2);
}

/*
 * Return alphabetically sorted dynamically allocated array
 * of code names (terminated with NULL). Used by help functions.
 */
static char **code_names(const char *t[], int n)
{
	char **r;

	r = malloc((n + 1) * sizeof(t[0]));
	if (!r)
		return NULL;
	memcpy(r, t, n * sizeof(t[0]));
	qsort(r, n, sizeof(t[0]), code_names_qsort_cmp);
	r[n] = NULL;
	return r;
}

/* Wrappers on code-name functions */
/* Return number of elements in static array */
#define COUNT_OF(a) (sizeof(a)/sizeof((a)[0]))
#define CODE2NAME(code, t) (code2name(code, t, COUNT_OF(t)))
#define NAME2CODE(name, t) name2code(name, t, COUNT_OF(t))
#define CODE_NAMES(t) code_names(t, COUNT_OF(t))

static const char *route_type_name[] = {
	[RTN_UNICAST] = "unicast",
	[RTN_BROADCAST] = "broadcast",
	[RTN_LOCAL] = "local",
	[RTN_UNREACHABLE] = "unreachable",
	[RTN_BLACKHOLE] = "blackhole",
	[RTN_PROHIBIT] = "prohibit",
};

static const char *route_scope_name[] = {
	[RT_SCOPE_HOST] = "host",
	[RT_SCOPE_LINK] = "link",
	[RT_SCOPE_UNIVERSE] = "universe",
	[RT_SCOPE_NOWHERE] = "nowhere",
};

static const char *route_proto_name[] = {
	[RTPROT_KERNEL] = "kernel", /* Route installed by kernel */
	[RTPROT_STATIC] = "static", /* Route installed by admin */
	[RTPROT_BOOT] = "boot", /* Route installed during boot */
	[RTPROT_DHCP] = "dhcp", /* Route installed by DHCP client */
	[RTPROT_BGP] = "bgp", /* BGP routes */
	[RTPROT_ISIS] = "isis", /* ISIS routes */
	[RTPROT_OSPF] = "ospf", /* OSPF routes*/
};

static const char *route_table_name[] = {
	[RT_TABLE_MAIN] = "main",
	[RT_TABLE_LOCAL] = "local",
};


static void print_route(struct nlr_route *r)
{
	struct in_addr in;
	const char *oif;

	/* Mimic "ip route" output */

	if (r->dest) {
		in.s_addr = r->dest;
		if (r->dest_plen == 32)
			printf("%s", inet_ntoa(in));
		else
			printf("%s/%d", inet_ntoa(in), r->dest_plen);
		if (r->gw) {
			in.s_addr = r->gw;
			printf(" via %s", inet_ntoa(in));
		}
	} else {
		in.s_addr = r->gw;
		printf("default via %s", inet_ntoa(in));
	}

	/*
	if (r->table != RT_TABLE_MAIN && r->table != RT_TABLE_LOCAL) {
		printf(" table %s", CODE2NAME(r->table, route_table_name));
	}
	*/

	/*
	if (r->type != RTN_UNICAST) {
		printf(" type %s", CODE2NAME(r->type, route_type_name));
	}
	*/

	oif = nlr_iface_name(r->oif);
	printf(" dev %s", oif);
	free((void *)oif);

	printf(" proto %s", CODE2NAME(r->proto, route_proto_name));

	if (r->scope != RT_SCOPE_UNIVERSE)
		printf(" scope %s", CODE2NAME(r->scope, route_scope_name));

	if (r->prefsrc) {
		in.s_addr = r->prefsrc;
		printf(" src %s", inet_ntoa(in));
	}

	if (r->flags & RTNH_F_LINKDOWN)
		printf(" linkdown");
	printf("\n");
}

/* 24 --> 255.255.255.0 */
static unsigned netmask(int plen)
{
	if (!plen)
		return 0;
	return htonl(~0 << (32 - plen));
}

static void init_route_filter(struct nlr_route *filter)
{
	filter->dest = INADDR_NONE;
	filter->dest_plen = -1;
	filter->gw = INADDR_NONE;
	filter->table = -1;
	filter->type = -1;
	filter->scope = -1;
	filter->proto = -1;
}

static int get_route(const char *s_addr)
{
	struct nlr_route *r, *p, *q, filter;
	int err;
	in_addr_t addr;

	addr = inet_addr(s_addr);
	if (addr == INADDR_ANY) {
		printf("Invalid address format\n");
		return -1;
	}
	init_route_filter(&filter);
	//filter.table = RT_TABLE_MAIN;
	r = nlr_get_routes(&filter, &err);
	if (err) {
		nlr_free_routes(r);
		return -1;
	}
	if (!r)
		return 0;
	for (p = r, q = NULL; p; p = p->pnext) {
		if ((addr & netmask(p->dest_plen)) == p->dest) {
			if (!q || p->dest_plen > q->dest_plen)
				q = p;
		}
	}
	if (q)
		print_route(q);
	nlr_free_routes(r);
	return 0;
}

static int show_routes(char *w[])
{
	int err;
	struct nlr_route *h, *r, filter;
	int i, mimic_iproute = 1;
	char **t;

	init_route_filter(&filter);

	if (!w || !w[0])
		goto w_processing_done;

	if (!strcmp(w[0], "all")) {
		if (w[1]) {
			printf("Expecting nothing after \"all\"\n");
			return -1;
		}
		mimic_iproute = 0;
		goto w_processing_done;
	}

	for (i = 0; w[i]; i += 2) {
		if (!w[i + 1]) {
			printf("Option \"%s\" expected value!\n", w[i]);
			return -1;
		}
		if (!strcmp(w[i], "dest")) {
			if (filter.dest != INADDR_NONE) {
			}
			if (parse_addr(w[i + 1], &filter.dest,
				&filter.dest_plen)) {
				printf("Invalid format of destination address\n");
				return -1;
			}
		} else if (!strcmp(w[i], "gw")) {
			if (filter.gw != INADDR_NONE) {
			}
			filter.gw = inet_addr(w[i + 1]);
			if (filter.gw == INADDR_NONE) {
				printf("Invalid format of gateway address\n");
				return -1;
			}
		} else if (!strcmp(w[i], "type")) {
			if (filter.type >= 0) {
			}
			filter.type = NAME2CODE(w[i + 1], route_type_name);
			if (filter.type < 0) {
				printf("Unknown route type. Use numeric value or one of:\n");
				t = CODE_NAMES(route_type_name);
				if (t) {
					for (i = 0; t[i]; i++)
						printf("* %s\n", t[i]);
					free(t);
				}
				return -1;
			}
		} else if (!strcmp(w[i], "scope")) {
			if (filter.scope >= 0) {
			}
			filter.scope = NAME2CODE(w[i + 1], route_scope_name);
			if (filter.scope < 0) {
				printf("Unknown route scope. Use numeric value or one of:\n");
				t = CODE_NAMES(route_scope_name);
				if (t) {
					for (i = 0; t[i]; i++)
						printf("* %s\n", t[i]);
					free(t);
				}
				return -1;
			}
		} else if (!strcmp(w[i], "proto")) {
			if (filter.proto >= 0) {
			}
			filter.proto = NAME2CODE(w[i + 1], route_proto_name);
			if (filter.proto < 0) {
				printf("Unknown route protocol. Use numeric value or one of:\n");
				t = CODE_NAMES(route_proto_name);
				if (t) {
					for (i = 0; t[i]; i++)
						printf("* %s\n", t[i]);
					free(t);
				}
				return -1;
			}
		} else if (!strcmp(w[i], "table")) {
			if (filter.table >= 0) {
			}
			filter.table = NAME2CODE(w[i + 1], route_table_name);
			if (filter.table < 0) {
				printf("Unknown route table. Use numeric value or one of:\n");
				t = CODE_NAMES(route_table_name);
				if (t) {
					for (i = 0; t[i]; i++)
						printf("* %s\n", t[i]);
					free(t);
				}
				return -1;
			}
		} else {
			printf("Unknown option: \"%s\"\n", w[i]);
			return -1;
		}
	}
w_processing_done:

	h = nlr_get_routes(&filter, &err);
	if (err)
		return -1;

	if (mimic_iproute) {
		/*
		 * If user doesn't specify a filter, then mimic "ip route
		 * output".
		 */
		for (r = h; r; r = r->pnext) {
			if (r->table != RT_TABLE_MAIN
				&& r->table != RT_TABLE_LOCAL)
				continue;
			if (r->type != RTN_UNICAST && r->type != RTN_LOCAL)
				continue;
			if (r->scope != RT_SCOPE_UNIVERSE
				&& r->scope != RT_SCOPE_LINK)
				continue;
			print_route(r);
		}
	} else {
		for (r = h; r; r = r->pnext)
			print_route(r);
	}

	nlr_free_routes(h);
	return err;
}

static int manage_route(const char *dest, const char *gw,
		       int (*f)(in_addr_t, int, in_addr_t))
{
	in_addr_t dest_addr, gw_addr;
	int dest_plen;

	if (parse_addr(dest, &dest_addr, &dest_plen)) {
		printf("Invalid format of dest addr\n");
		return -1;
	}
	gw_addr = inet_addr(gw);
	if (gw_addr == INADDR_NONE) {
		printf("Invalid gw addr\n");
		return -1;
	}
	return f(dest_addr, dest_plen, gw_addr);
}

static int add_route(const char *dest, const char *gw)
{
	return manage_route(dest, gw, nlr_add_route);
}

static int del_route(const char *dest, const char *gw)
{
	return manage_route(dest, gw, nlr_del_route);
}

static void help(void)
{
	printf("\nUsage: [OPTIONS] OBJECT CMD [CMD_OPTIONS]" \
	       "\nOptions: -d -- debug, -h -- help" \
	       " -s -- stats (show more detailed info),"
	       "\n$ ip link [show [IFACE]]" \
	       "\n$ ip link add IFACE type bridge" \
	       "\n$ ip link add IFACE type vlan MASTER_IFACE VLAN_ID" \
	       "\n$ ip link del IFACE" \
	       "\n$ ip link set IFACE up|down"
	       "\n$ ip link set IFACE addr hh:hh:hh:hh:hh:hh" \
	       "\n$ ip link set IFACE master MASTER" \
	       "\n$ ip link set IFACE nomaster" \
	       "\n$ ip addr [show [IFACE]]" \
	       "\n$ ip addr add|del IFACE ADDR/BITS" \
	       "\n$ ip route [show [dest ADDR/BITS] [proto PROTO] [type TYPE] [scope SCOPE] [table TABLE] [gw ADDR]]" \
	       "\n$ ip route show all" \
	       "\n$ ip route get ADDR" \
	       "\n$ ip route add|del DEST/BITS via GW" \
	       "\n"
	);
}

int main(int argc, char *argv[])
{
	int r, i, logmask;
	const char *obj, *cmd, *iface;
	int c, debug;

	if (!argv[1] || !strcmp(argv[1], "-h")) {
		help();
		return 0;
	}

	/* Parse common options */
	debug = 0;
	while ((c = getopt(argc, (char **)argv, "dsh")) != -1) {
		switch (c) {
		case 'd':
			debug = 1;
			break;
		case 'h':
			help();
			return 0;
		case 's':
			stats = 1;
			break;
		case '?':
		default:
			fprintf(stderr, "\nInvalid options. Use '-h' to get help.\n");
			return -1;
		}
	}

	i = optind;
	if (i == argc)
		return 0;
	logmask = LOG_MASK(LOG_ERR);
	if (debug) {
		logmask |= LOG_MASK(LOG_INFO) | LOG_MASK(LOG_DEBUG);
		/* Enable debugging in the libnel */
		setenv("LIBNEL_DEBUG", "1", 0);
	}
	openlog(NULL, LOG_PERROR, LOG_USER);
	setlogmask(logmask);

	if (nlr_init()) {
		fprintf(stderr, "nlroute init failed\n");
		return -1;
	}

	obj = argv[i];
	if (!obj) {
		fprintf(stderr, "Missing object.\n");
		goto fin;
	}

	cmd = argv[i + 1];
	argv = argv + i + 2;
	if (!cmd) {
		cmd = "show";
		argv--;
	}

	r = 1;
	if (!strcmp(obj, "link")) {
		if (!strcmp(cmd, "show")) {
			if (argv[0] && argv[1])
				goto fin;
			r = get_iface_info(argv[0]);
		} else if (!strcmp(cmd, "set")) {
			iface = argv[0];
			if (!iface || !argv[1])
				goto fin;
			if (!strcmp(argv[1], "up")) {
				if (argv[2])
					goto fin;
				r = set_iface(iface, 1);
			} else if (!strcmp(argv[1], "down")) {
				if (argv[2])
					goto fin;
				r = set_iface(iface, 0);
			} else if (!strcmp(argv[1], "addr")) {
				if (!argv[2] || argv[3])
					goto fin;
				r = set_iface_addr(iface, argv[2]);
			} else if (!strcmp(argv[1], "master")) {
				if (!argv[2] || argv[3])
					goto fin;
				r = set_iface_master(iface, argv[2]);
			} else if (!strcmp(argv[1], "nomaster")) {
				if (argv[2])
					goto fin;
				r = set_iface_master(iface, NULL);
			}
		} else if (!strcmp(cmd, "add")) {
			iface = argv[0];
			if (!iface || !argv[1] || strcmp(argv[1], "type"))
				goto fin;
			if (!strcmp(argv[2], "bridge")) {
				if (argv[3])
					goto fin;
				r = add_bridge(iface);
			} else if (!strcmp(argv[2], "vlan")) {
				if (!argv[3] || !argv[4])
					goto fin;
				r = add_vlan(iface, argv[3], argv[4]);
			}
		} else if (!strcmp(cmd, "del")) {
			iface = argv[0];
			if (!iface || argv[1])
				goto fin;
			r = del_iface(iface);
		}
	} else if (!strcmp(obj, "addr")) {
		if (!strcmp(cmd, "show")) {
			if (!argv[0] && argv[1])
				goto fin;
			r = get_addr(argv[0]);
		} else if (!strcmp(cmd, "add")) {
			if (!argv[0] || !argv[1] || argv[2])
				goto fin;
			r = add_addr(argv[0], argv[1]);
		} else if (!strcmp(cmd, "del")) {
			if (!argv[0] || !argv[1] || argv[2])
				goto fin;
			r = del_addr(argv[0], argv[1]);
		}
	} else if (!strcmp(obj, "route")) {
		if (!strcmp(cmd, "show")) {
			r = show_routes(argv);
		} else if (!strcmp(cmd, "get")) {
			if (!argv[0] || argv[1])
				goto fin;
			r = get_route(argv[0]);
		} else if (!strcmp(cmd, "add")) {
			if (!argv[0] || strcmp(argv[1], "via") || !argv[2] || argv[3])
				goto fin;
			r = add_route(argv[0], argv[2]);
		} else if (!strcmp(cmd, "del")) {
			if (!argv[0] || strcmp(argv[2], "via") || !argv[2] || argv[3])
				goto fin;
			r = del_route(argv[0], argv[2]);
		}
	}

fin:
	if (r < 0)
		printf("%s\n", r ? "Failed" : "Done");
	else if (r > 0)
		fprintf(stderr, "Invalid args. See help (-h).\n");

	nlr_fin();

	closelog();

	return r;
}
