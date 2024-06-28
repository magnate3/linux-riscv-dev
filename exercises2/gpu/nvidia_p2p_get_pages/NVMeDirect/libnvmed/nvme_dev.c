////////////////////////////////////////////////////////////////////////
//
// Copyright 2014 PMC-Sierra, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You may
// obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0 Unless required by
// applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the
// License.
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
//
//   Author: Logan Gunthorpe
//
//   Date:   Oct 23 2014
//
//   Description:
//     NVME device wrapper
//
////////////////////////////////////////////////////////////////////////

#include "nvme_dev.h"
#include <stdbool.h>
#include <linux/nvme.h>
#include "include/nvme_ioctl.h"

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <sys/time.h>
#include <stdio.h>

static int find_dev(dev_t dev)
{
    static struct dev_cache {
        dev_t dev;
        int fd;
    } cache[] = {[16] = {.dev=-1}};
struct dev_cache *c;
    for (c = cache; c->dev != -1; c++)
        if (c->dev == dev)
            return c->fd;

    const char *dir = "/dev";
    DIR *dp = opendir(dir);
    if (!dp)
        return -errno;

    struct dirent *entry;
    while ((entry = readdir(dp)) != NULL) {
        if (entry->d_type != DT_UNKNOWN  && entry->d_type != DT_BLK)
            continue;

        struct stat64 st;
        if (fstatat(dirfd(dp), entry->d_name, &st, 0))
            continue;

        if (!S_ISBLK(st.st_mode))
            continue;

        if (st.st_rdev != dev)
            continue;


        int ret = openat(dirfd(dp), entry->d_name, O_RDONLY);

        for (c = cache; c->dev != -1; c++) {
            if (c->dev == 0) {
                c->dev = dev;
                c->fd = ret;
            }
        }

        closedir(dp);
        return ret;
    }

    errno = ENOENT;
    closedir(dp);
    return -1;
}

int nvme_dev_find(dev_t dev)
{
    int devfd = find_dev(dev);
    if (devfd < 0)
        return -EPERM;

    if (ioctl(devfd, NVME_IOCTL_ID, 0) < 0)
        return -ENXIO;

    return devfd;
}


int nvme_dev_gpu_read(int devfd, int slba, int nblocks,
                      const struct pin_buf *dest,
                      unsigned long offset)
{
    int ret = 0;
    struct nvme_user_gpu_io iocmd = {
        .opcode = nvme_cmd_read,
        .slba = slba,
        .nblocks = nblocks-1,
        .gpu_mem_handle = dest->handle,
        .gpu_mem_offset = offset,
    };
//    fprintf(stderr, "NVME_IOCTL_SUBMIT_GPU_IO: slba: %x offset -> %x\n",slba, offset);
    ret = ioctl(devfd, NVME_IOCTL_SUBMIT_GPU_IO, &iocmd);
    return ret;
}

int nvme_dev_gpu_write(int devfd, int slba, int nblocks,
                      const struct pin_buf *src,
                      unsigned long offset)
{
    int ret = 0;
    struct nvme_user_gpu_io iocmd = {
        .opcode = nvme_cmd_write,
        .slba = slba,
        .nblocks = nblocks-1,
        .gpu_mem_handle = src->handle,
        .gpu_mem_offset = offset,
    };
//    fprintf(stderr, "NVME_IOCTL_SUBMIT_GPU_IO: slba: %x offset -> %x\n",slba, offset);
    ret = ioctl(devfd, NVME_IOCTL_SUBMIT_GPU_IO, &iocmd);
    return ret;
}


