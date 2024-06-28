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
//     File mapper, which offers multiple ways to map a file to CPU or
//     GPU memory.
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_FILEMAP_H__
#define __LIBDONARD_FILEMAP_H__

#include "pinpool.h"

#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif

struct filemap {
    const char *filename;
    void *data;
    size_t length;

    struct pin_buf *pinbuf;

    enum {
        FILEMAP_TYPE_LOCAL = 0x1,
        FILEMAP_TYPE_CUDA = 0x2,
    } type;

    enum {
        FILEMAP_FALLBACK_NOPERM_DEV = 1,
        FILEMAP_FALLBACK_DEV_NOT_NVME = 2,
        FILEMAP_FALLBACK_NOPERM_FIBMAP = 3,
        FILEMAP_FALLBACK_IOCTL_ERROR = 4,
        FILEMAP_FALLBACK_PINBUF_TOO_SMALL = 5,
    } map_error;

    void (*free)(struct filemap *fm);
};




const char * filemap_map_error_string(int map_error);

static inline void filemap_free(struct filemap *fm)
{
    fm->free(fm);
}

struct filemap *filemap_open_local(const char *filename);
struct filemap *filemap_alloc_local(int fd, const char *fname);

/*
 * Map a file by DMAing it's contents to local memory,
 *  using the user_io IOCTL from an nvme device.
 *  In practice, this is not a smart thing to do, it's
 *  only here as a test/example for the IOCTL.
 */

struct filemap *filemap_open_local_nvme(const char *filename);
struct filemap *filemap_alloc_local_nvme(int fd, const char *fname);

struct filemap *filemap_open_cuda(const char *filename);
struct filemap *filemap_alloc_cuda(int fd, const char *fname);

struct filemap *filemap_open_cuda_nvme(const char *filename);
struct filemap *filemap_alloc_cuda_nvme(int fd, const char *fname);

int filemap_write_cuda_nvme(struct filemap *fmap_src, int fd_dst);

#ifdef __cplusplus
}
#endif


#endif
