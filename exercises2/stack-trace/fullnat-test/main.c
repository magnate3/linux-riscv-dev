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

#include "fullnat.h"

#include <linux/module.h>

int g_mode_tcp6 = 0;
int g_mode_udp6 = 0;
int g_mode_tcp = 0;
int g_mode_udp = 0;

static int __init fullnat_initialize(void)
{
    int ret;

    ret = procfs_initialize();
    if (ret) {
        return ret;
    }

    ret = sysfs_initialize();
    if (ret) {
        procfs_shutdown();
        return ret;
    }

    ret = rip_initialize();
    if (ret) {
        sysfs_shutdown();
        procfs_shutdown();
        return ret;
    }

    return 0;
}

static void __exit fullnat_shutdown(void)
{
    rip_shutdown();
    sysfs_shutdown();
    procfs_shutdown();
}

module_init(fullnat_initialize);
module_exit(fullnat_shutdown);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("yiyuanzhong@gmail.com (Yiyuan Zhong)");
