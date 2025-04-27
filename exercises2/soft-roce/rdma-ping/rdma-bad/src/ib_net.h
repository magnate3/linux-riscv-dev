//
// Created by michele on 07.02.24.
//

#pragma once

#include "../common/common.h"
#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct ib_node_info {
    int lid;
    int qpn;
    int psn;
    union ibv_gid gid;
};

struct ibv_device *ib_device_find_by_name (const char *name);

/**
 * Retrieve information of the local IB device used for the communication.
 * If successful, the data will be stored in the out parameter.
 *
 * @param context the IB device context
 * @param ib_port the IB port
 * @param gidx the GID index
 * @param qp the queue pair
 * @param out the variable where the information will be stored
 * @return 0 if successful, 1 otherwise
 */
int ib_get_local_info (struct ibv_context *context, int ib_port, int gidx, struct ibv_qp *qp, struct ib_node_info *out);

/**
 * Print the information of the IB node.
 *
 * @param info the information to be printed
 */
void ib_print_node_info (struct ib_node_info *info);
