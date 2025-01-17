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

#include <linux/version.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/proc_fs.h>
#include <linux/uaccess.h>
#include <net/net_namespace.h>

#include "fullnat.h"

#define PROCFS_FILENAME "fullnat"
static int procfs_seq_show(struct seq_file *p, void *v)
{
    seq_printf(p, "tcp6_mode=%d\n", g_mode_tcp6);
    seq_printf(p, "udp6_mode=%d\n", g_mode_udp6);
    seq_printf(p, "tcp_mode=%d\n", g_mode_tcp);
    seq_printf(p, "udp_mode=%d\n", g_mode_udp);
    return 0;
}

static int procfs_open(struct inode *inode, struct file *file)
{
    if (!try_module_get(THIS_MODULE)) {
        return -EPERM;
    }

    return single_open(file, procfs_seq_show, NULL);
}

static int procfs_close(struct inode *inode, struct file *file)
{
    single_release(inode, file);
    module_put(THIS_MODULE);
    return 0;
}

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 6, 0)
static struct file_operations g_fileops __read_mostly = {
    .open    = procfs_open,
    .read    = seq_read,
    .llseek  = seq_lseek,
    .release = procfs_close,
};

#else
static struct proc_ops g_fileops = {
   .proc_open = procfs_open,
   .proc_read = seq_read,
   .proc_lseek  = seq_lseek,
   //.proc_write = proc_write,
   .proc_release = procfs_close,
};
#endif
int procfs_initialize(void)
{
    if (!proc_create_data(PROCFS_FILENAME,
                          S_IRUGO,
                          init_net.proc_net,
                          &g_fileops,
                          NULL)) {

        return -ENOMEM;
    }

    return 0;
}
void procfs_shutdown(void)
{
    remove_proc_entry(PROCFS_FILENAME, init_net.proc_net);
}
