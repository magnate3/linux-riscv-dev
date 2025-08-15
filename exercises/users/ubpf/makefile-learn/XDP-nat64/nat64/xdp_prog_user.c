/* SPDX-License-Identifier: GPL-2.0-only
   Copyright (c) 2019-2022 */

static const char *__doc__ = "XDP redirect helper\n"
	" - Allows to populate/query tx_port and redirect_params maps\n";

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <getopt.h>
#include <stdbool.h>

#include <locale.h>
#include <unistd.h>
#include <time.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#include <net/if.h>
#include <linux/if_ether.h>
#include <linux/if_link.h> /* depend on kernel-headers installed */

#include "../common/common_params.h"
#include "../common/common_user_bpf_xdp.h"
#include "../common/common_libbpf.h"

#include "../common/xdp_stats_kern_user.h"

static const struct option_wrapper long_options[] = {

	{{"help",        no_argument,		NULL, 'h' },
	 "Show help", false},

	{{"dev",         required_argument,	NULL, 'd' },
	 "Operate on device <ifname>", "<ifname>", true},
	{{"redirect-dev",         required_argument,	NULL, 'r' },
	 "Redirect to device <ifname>", "<ifname>", true},

	{{"src-mac", required_argument, NULL, 'L' },
	 "Source MAC address of <dev>", "<mac>", true },

	{{"dest-mac", required_argument, NULL, 'R' },
	 "Destination MAC address of <redirect-dev>", "<mac>", true },

	{{"filename",    required_argument,	NULL,  1  },
	 "Load program from <file>", "<file>"},

	{{"quiet",       no_argument,		NULL, 'q' },
	 "Quiet mode (no output)"},

	{{0, 0, NULL,  0 }, NULL, false}
};

static int static_redirect_non_ip (int map_fd, __u8* dst, __u32* ifindex)
{
	if (bpf_map_update_elem (map_fd, dst, ifindex, 0) < 0) {
		fprintf(stderr,
			"WARN: Failed to update bpf map file: err(%d):%s\n",
			errno, strerror(errno));
		return -1;
	}
	if (verbose)
	{
		printf ("Redirecting dst address: %d to ifindex: %d\n", *dst, *ifindex);
	}

	return 0;
}


#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

const char *pin_basedir =  "/sys/fs/bpf";

int main(int argc, char **argv)
{
	int len;
	int map_fd;
	char pin_dir[PATH_MAX];

	struct config cfg = {
		.ifindex   = -1,
		.redirect_ifindex   = -1,
	};

	/* Cmdline options can change progsec */
	parse_cmdline_args(argc, argv, long_options, &cfg, __doc__);
	len = snprintf(pin_dir, PATH_MAX, "%s/%s", pin_basedir, cfg.ifname);
	if (len < 0) {
		fprintf(stderr, "ERR: creating pin dirname\n");
		return EXIT_FAIL_OPTION;
	}
	if (verbose)
	{
		printf("map dir: %s\n", pin_dir);
	}

	/* Open the static_redirect_8b map corresponding to the cfg.ifname interface */
	map_fd = open_bpf_map_file(pin_dir, "static_redirect_8b", NULL);
	if (map_fd < 0) 
	{
		return EXIT_FAIL_BPF;
	}
	__u8 nIPdst;
	__u32 nIPifindex;
	int cur_pos = 0;
	while (cur_pos < strlen(cfg.filename))
	{
		nIPdst = cfg.filename[cur_pos] - 48;// converting ascii to int
		cur_pos += 2;
		int sub_pos = 0;
		char substr[10];
		while (sub_pos < 5)
		{
			substr[sub_pos] = cfg.filename[cur_pos];
			cur_pos++;
			sub_pos++;
		}
		nIPifindex = if_nametoindex(substr);

		if (static_redirect_non_ip(map_fd, &nIPdst, &nIPifindex) < 0) 
		{
			printf("cant write static redirect map\n");
			fprintf(stderr, "can't write static redirect map\n");
			return 1;
		}
		cur_pos++;
	}
	return EXIT_OK;
}
