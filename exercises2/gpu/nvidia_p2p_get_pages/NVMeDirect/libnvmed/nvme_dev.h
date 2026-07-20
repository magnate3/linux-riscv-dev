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

#ifndef P2MTR_NVME_DEV_H
#define P2MTR_NVME_DEV_H

#include "pinpool.h"
//#include "include/nvme_ioctl.h"

#include <sys/stat.h>
#include <stdlib.h>
#include <linux/types.h>

struct nvme_dev_sector {
    unsigned long slba;
    unsigned long count;
};

int nvme_dev_find(dev_t dev);

//int nvme_dev_read(int devfd, int slba, int nblocks, void *dest);
//int nvme_dev_write(int devfd, int slba, int nblocks, const void *src);
int nvme_dev_gpu_read(int devfd, int slba, int nblocks,
                      const struct pin_buf *dest,
                      unsigned long offset);
int nvme_dev_gpu_write(int devfd, int slba, int nblocks,
                       const struct pin_buf *src,
                       unsigned long offset);

#endif
