#include <assert.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <sys/socket.h>
#include <net/if.h>
#include <asm/types.h>

#include <arpa/inet.h>
#include <libmnl/libmnl.h>
#include <linux/if.h>
#include <linux/if_link.h>
#include <linux/rtnetlink.h>
#include <linux/dcbnl.h>

#define NUMPRI  (8)
#define BUFLEN (8192L)
struct dcbnetlink_state {
    struct mnl_socket *nl;
    //int ifindex;
    char iface[IFNAMSIZ];
    uint64_t rates[NUMPRI];
    char buf[BUFLEN];
};

int
send_dcbnetlink_msg(struct dcbnetlink_state *dcbstate);

int
init_dcbnetlink(struct dcbnetlink_state *dcbstate, char *iface);
