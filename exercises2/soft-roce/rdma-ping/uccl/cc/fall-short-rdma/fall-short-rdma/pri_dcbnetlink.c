
#include "pri_dcbnetlink.h"

char *usage_str = "Usage: %s [OPTION]\n"
    "\t-r\tIgnore RECV\n"
    "\t-i\tThe network interface to use\n";

void
usage(int argc, char **argv)
{
    assert(argc > 0);
    fprintf(stderr, usage_str, argv[0]);
}


int
send_dcbnetlink_msg(struct dcbnetlink_state *dcbstate)
{
    struct nlmsghdr *nlh;
    struct dcbmsg *dcb;
    struct nlattr *ieee_nested;

    /* Build the base message. */
    nlh = mnl_nlmsg_put_header(dcbstate->buf);
    nlh->nlmsg_type = RTM_GETDCB;
    nlh->nlmsg_flags = NLM_F_REQUEST;
    nlh->nlmsg_seq = time(NULL);
    dcb = mnl_nlmsg_put_extra_header(nlh, sizeof(*dcb));
    dcb->dcb_family = AF_UNSPEC;
    dcb->cmd = DCB_CMD_IEEE_SET;
    dcb->dcb_pad = 0;
    mnl_attr_put_strz(nlh, DCB_ATTR_IFNAME, dcbstate->iface);
    ieee_nested = mnl_attr_nest_start_check(nlh, BUFLEN, DCB_ATTR_IEEE);
    assert(sizeof(dcbstate->rates) == sizeof(uint64_t) * NUMPRI);
    mnl_attr_put(nlh, DCB_ATTR_IEEE_MAXRATE, sizeof(dcbstate->rates), dcbstate->rates);
    mnl_attr_nest_end(nlh, ieee_nested);

    /* Try to send the message */
    if (mnl_socket_sendto(dcbstate->nl, nlh, nlh->nlmsg_len) < 0) {
        fprintf(stderr, "Error: mnl_socket_send\n");
        return (-1);
    }

    return (0);
}

int
init_dcbnetlink(struct dcbnetlink_state *dcbstate, char *iface)
{
    struct mnl_socket *nl;
    //int ifindex = 0; 

    /* Initialize the DCB state. */
    memset(dcbstate, 0, sizeof(*dcbstate));

    /* Get the ifindex we will be configuring. */
    //ifindex = if_nametoindex(iface);
    //if (!ifindex) {
    //    fprintf(stderr, "Invalid device %s\n", iface);
    //    return (-1);
    //}

    /* Open the netlink socket. */
    nl = mnl_socket_open(NETLINK_ROUTE);
    if (nl == NULL) {
        fprintf(stderr, "Error: mnl_socket_open\n");
        return (-1);
    }

    /* Bind the netlink socket. */
    if (mnl_socket_bind(nl, 0, MNL_SOCKET_AUTOPID) < 0) {
        fprintf(stderr, "Error: mnl_socket_bind\n");
        return (-1);
    }


    /* Save the state. */
    dcbstate->nl = nl;
    strncpy(dcbstate->iface, iface, IFNAMSIZ);

    return (0);
}

int
run_dcbnetlink_test(char *iface, bool ign_recv)
{
    struct dcbnetlink_state dcbstate;
    int i;

    if (init_dcbnetlink(&dcbstate, iface) < 0) {
        fprintf(stderr, "Error initializing DCB NETLINK state\n");
        return (-1);
    }

    /* Set the rate-limits */
    for (i = 0; i < NUMPRI; i++) {
        dcbstate.rates[i] = 7 * 1000 * 1000; //7 Gbps
    }

    if (send_dcbnetlink_msg(&dcbstate) < 0) {
        fprintf(stderr, "Error sending DCB NETLINK message\n");
        return (-1);
    }

    if (ign_recv) {
        printf("Ignoring RECV.\n");
    } else {
        printf("Checking RECV.\n");
        /* TODO */
    }

    return (0);
}

/*int
main(int argc, char **argv)
{
    int c;
    bool ign_recv = false;
    char *iface = NULL;

    // Parse the command line options.  Not that relevant to this program. 
    while ((c = getopt(argc, argv, "i:rh")) != -1) {
        switch (c) {
        case 'i':
            iface = optarg;
            break;
        case 'h':
            usage(argc, argv);
            exit(1);
            break;
        case 'r':
            ign_recv = true;
            break;
        default:
            break;
        }
    }

    // Validate argumnets
    if (iface == NULL) {
        usage(argc, argv);
        exit(1);
    }

    // Run a DCB NETLINK test. 
    if (run_dcbnetlink_test(iface, ign_recv) < 0) {
        fprintf(stderr, "ERROR running DCB NETLINK test!\n");
        exit(1);
    }


    return (0);
}*/

