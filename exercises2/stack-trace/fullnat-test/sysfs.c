/* Copyright 2015 yiyuanzhong@gmail.com (Yiyuan Zhong)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <net/net_namespace.h>

#include "fullnat.h"

static int one = 1;
static int zero = 0;
static struct ctl_table_header *g_header;

static struct ctl_table sysfs_table[] __read_mostly = {
    {
        .procname = "tcp6_mode",
        .data = &g_mode_tcp6,
        .maxlen = sizeof(int),
        .mode = 0644,
        .proc_handler = proc_dointvec_minmax,
        .extra1 = &zero,
        .extra2 = &zero,
    },
    {
        .procname = "tcp_mode",
        .data = &g_mode_tcp,
        .maxlen = sizeof(int),
        .mode = 0644,
        .proc_handler = proc_dointvec_minmax,
        .extra1 = &zero,
        .extra2 = &one,
    },
    {
        .procname = "udp6_mode",
        .data = &g_mode_udp6,
        .maxlen = sizeof(int),
        .mode = 0644,
        .proc_handler = proc_dointvec_minmax,
        .extra1 = &zero,
        .extra2 = &zero,
    },
    {
        .procname = "udp_mode",
        .data = &g_mode_udp,
        .maxlen = sizeof(int),
        .mode = 0644,
        .proc_handler = proc_dointvec_minmax,
        .extra1 = &zero,
        .extra2 = &zero,
    },
    {}
};

int sysfs_initialize(void)
{
    g_header = register_net_sysctl(&init_net, "net/fullnat", sysfs_table);
    if (!g_header) {
        return -ENOMEM;
    }

    return 0;
}

void sysfs_shutdown(void)
{
    unregister_net_sysctl_table(g_header);
}
