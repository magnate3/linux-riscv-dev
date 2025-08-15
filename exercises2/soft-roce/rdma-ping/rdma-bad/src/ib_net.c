//
// Created by michele on 07.02.24.
//

#include "ib_net.h"

struct ibv_device *ib_device_find_by_name (const char *name)
{
    struct ibv_device **dev_list = ibv_get_device_list (NULL);
    if (!dev_list)
    {
        fprintf (stderr, "Failed to get IB devices list\n");
        return NULL;
    }

    struct ibv_device *dev = NULL;
    for (int i = 0; dev_list[i] != NULL; i++)
    {
        if (strcmp (dev_list[i]->name, name) == 0)
        {
            dev = dev_list[i];
            break;
        }
    }

    ibv_free_device_list (dev_list);
    return dev;
}

int ib_get_local_info (struct ibv_context *restrict context, int ib_port, int gidx, struct ibv_qp *restrict qp, struct ib_node_info *restrict out)
{
    struct ibv_port_attr port_info;
    if (ibv_query_port (context, ib_port, &port_info))
    {
        LOG (stderr, "Couldn't get port info\n");
        return 1;
    }

    out->lid = port_info.lid;

    if (port_info.link_layer != IBV_LINK_LAYER_ETHERNET && !out->lid)
    {
        LOG (stderr, "Couldn't get local LID\n");
        return 1;
    }

    if (ibv_query_gid (context, ib_port, gidx, &out->gid))
    {
        LOG (stderr, "Couldn't get local GID\n");
        return 1;
    }

    out->qpn = qp->qp_num;
    out->psn = lrand48 () & 0xffffff;

    return 0;
}

void ib_print_node_info (struct ib_node_info *info)
{
    char gid_str[33];
    inet_ntop (AF_INET6, &info->gid, gid_str, sizeof (gid_str));
    printf ("Address:  LID 0x%04x, QPN 0x%06x, PSN 0x%06x, GID %s\n", info->lid, info->qpn, info->psn, gid_str);
}