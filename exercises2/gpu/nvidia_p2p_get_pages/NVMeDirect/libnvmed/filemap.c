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
//   Author: Hung-Wei Tseng
//
//   Date:   Dec 01 2014
//
//   Description:
//     Updated the filemap_alloc_cuda_nvme function to support files
//     that are larger than the pinbuffer size.
//
////////////////////////////////////////////////////////////////////////


#include "filemap.h"
#include "pinpool.h"
#include "nvme_dev.h"

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

#include <cuda_runtime.h>

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

static void free_munmap(struct filemap *fm)
{
    munmap(fm->data, fm->length);
    if (fm->filename != NULL)
        free((void *) fm->filename);
    free(fm);
}

static void copy_filename(struct filemap *fm, const char *fname)
{
    if (fname == NULL) {
        fm->filename = NULL;
        return;
    }

    fm->filename = malloc(strlen(fname)+1);
    if (fm->filename == NULL)
        return;

    strcpy((char *) fm->filename, fname);
}

struct filemap *filemap_alloc_local(int fd, const char *fname)
{
    struct stat64 stats;
    if (fstat(fd, &stats))
        return NULL;

    struct filemap *fm = malloc(sizeof(*fm));
    if (fm == NULL)
        return NULL;

    fm->map_error = 0;
    fm->length = stats.st_size;
    fm->type = FILEMAP_TYPE_LOCAL;
    copy_filename(fm, fname);

    fm->data = mmap(NULL, fm->length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fm->data == (void *) -1)
        goto error_exit_free;

    fm->free = free_munmap;

    return fm;

error_exit_free:
    free(fm);
    return NULL;
}

struct filemap *filemap_open_local(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return NULL;

    struct filemap *ret = filemap_alloc_local(fd, fname);
    close(fd);
    return ret;
}

static void free_local_nvme(struct filemap *fm)
{
    free(fm->data);
    if (fm->filename != NULL)
        free((void *) fm->filename);
    free(fm);
}

/*
 * Map a file by DMAing it's contents to local memory,
 *  using the user_io IOCTL from an nvme device.
 *  In practice, this is not a smart thing to do. It's
 *  only here as a test/example for the NVME IOCTL.
 */
struct filemap *filemap_alloc_local_nvme(int fd, const char *fname)
{
    int map_error;
    struct stat64 st;
    int i;
    if (fstat(fd, &st))
        return NULL;

    struct nvme_dev_sector *slist = NULL;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0) {
        if (devfd == -ENXIO)
            map_error = FILEMAP_FALLBACK_DEV_NOT_NVME;
        else
            map_error = FILEMAP_FALLBACK_NOPERM_DEV;
        goto fallback;
    }

    struct filemap *fm = malloc(sizeof(*fm));
    if (fm == NULL)
        return NULL;

    fm->map_error = 0;
    fm->length = st.st_size;
    fm->type = FILEMAP_TYPE_LOCAL;
    fm->free = free_local_nvme;
    copy_filename(fm, fname);

    int sector_count = nvme_dev_get_sector_list(fd, &st, &slist, 0);

    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        goto free_and_fallback;
    }


    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    fm->data = malloc(num_blocks * st.st_blksize);
    if (fm->data == NULL)
        goto exit_error_free;

    unsigned char *dest = fm->data;
    for (i = 0; i < sector_count; i++) {
        if (nvme_dev_read(devfd, slist[i].slba, slist[i].count, dest)) {
            map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
            free(fm->data);
            goto free_slist_and_fallback;
        }

        dest += slist[i].count * 512;
    }

    free(slist);
    return fm;

exit_error_free:
    free(slist);
    free(fm);
    return NULL;

free_slist_and_fallback:
    free(slist);

free_and_fallback:
    free(fm);

fallback:
    errno = 0;
    struct filemap *ret = filemap_alloc_local(fd, fname);
    ret->map_error = map_error;
    return ret;
}

struct filemap *filemap_open_local_nvme(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return NULL;

    struct filemap *ret = filemap_alloc_local_nvme(fd, fname);
    close(fd);
    return ret;
}

static void free_cuda(struct filemap *fm)
{
    cudaFree(fm->data);
    if (fm->filename != NULL)
        free((void *) fm->filename);
    free(fm);
}

struct filemap *filemap_alloc_cuda(int fd, const char *fname)
{
    struct stat64 stats;
    if (fstat(fd, &stats))
        return NULL;

    struct filemap *fm = malloc(sizeof(*fm));
    if (fm == NULL)
        return NULL;

    fm->map_error = 0;
    fm->length = stats.st_size;
    fm->type = FILEMAP_TYPE_CUDA;
    fm->free = free_cuda;
    copy_filename(fm, fname);

    if (cudaMalloc(&fm->data, fm->length) != cudaSuccess) {
        errno = ENOMEM;
        free(fm);
        return NULL;
    }

    void *data = mmap(NULL, fm->length, PROT_READ, MAP_PRIVATE, fd, 0);
    if (fm->data == (void *) -1)
        goto error_exit_free;


    cudaMemcpy(fm->data, data, fm->length, cudaMemcpyHostToDevice);

    munmap(data, fm->length);

    return fm;

error_exit_free:
    cudaFree(fm->data);
    free(fm);
    return NULL;
}

struct filemap *filemap_open_cuda(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return NULL;

    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    close(fd);
    return ret;
}

static void free_cuda_nvme_large(struct filemap *fm)
{
    if (fm->filename != NULL)
        free((void *) fm->filename);

    cudaFree(fm->data);

    free(fm);
}

static struct filemap *alloc_cuda_nvme_large(int fd, const char *fname,
                                             int devfd, struct filemap *fm,
                                             unsigned long num_blocks,
                                             struct nvme_dev_sector *slist,
                                             int sector_count)
{
    int map_error = 0;
    int i;
    if (cudaMalloc(&fm->data, fm->length) != cudaSuccess) {
        errno = ENOMEM;
        goto free_and_fallback;
    }

    fm->free = free_cuda_nvme_large;

    unsigned long max_lbas = fm->pinbuf->bufsize / 512;

    unsigned char *current = fm->data;
    for (i = 0; i < sector_count; i++) {
        unsigned long slba = slist[i].slba;
        unsigned long count = slist[i].count;

        while (count) {
            unsigned long c = count;
            if (c > max_lbas)
                c = max_lbas;

            if (nvme_dev_gpu_read(devfd, slba, c, fm->pinbuf, 0))
                goto map_error_and_free;

            slba += c;
            count -= c;

            cudaMemcpy(current, fm->pinbuf->address, c*512,
                       cudaMemcpyDeviceToDevice);
            current += c*512;
        }
    }

    pinpool_free(fm->pinbuf);
    fm->pinbuf = NULL;

    return fm;

map_error_and_free:
    map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
    cudaFree(fm->data);

free_and_fallback:
    pinpool_free(fm->pinbuf);
    free(fm);

    errno = 0;
    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    ret->map_error = map_error;
    return ret;
}

static void free_cuda_nvme(struct filemap *fm)
{
    pinpool_free(fm->pinbuf);

    if (fm->filename != NULL)
        free((void *) fm->filename);

    free(fm);
}

struct filemap *filemap_alloc_cuda_nvme(int fd, const char *fname)
{
    int map_error=0;
    struct stat64 st;
    int i;
    if (fstat(fd, &st))
        return NULL;

    struct nvme_dev_sector *slist;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0) {
        if (devfd == -ENXIO)
            map_error = FILEMAP_FALLBACK_DEV_NOT_NVME;
        else
            map_error = FILEMAP_FALLBACK_NOPERM_DEV;
        goto fallback;
    }

    int sector_count = nvme_dev_get_sector_list(fd, &st, &slist, 0);

    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        goto fallback;
    }

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;

    struct filemap *fm = malloc(sizeof(*fm));
    if (fm == NULL) {
        free(slist);
        return NULL;
    }

    fm->map_error = 0;
    fm->length = st.st_size;
    fm->type = FILEMAP_TYPE_CUDA;
    fm->free = free_cuda_nvme;
    copy_filename(fm, fname);
    fm->pinbuf = pinpool_alloc();
    fm->data = fm->pinbuf->address;

    if (num_blocks * st.st_blksize > fm->pinbuf->bufsize) {
        struct filemap * rt = alloc_cuda_nvme_large(fd, fname, devfd, fm, num_blocks,
                                                    slist, sector_count);
        free(slist);
        return rt;
    }

    unsigned long offset = 0;
    for (i = 0; i < sector_count; i++) {
        if (nvme_dev_gpu_read(devfd, slist[i].slba, slist[i].count,
                              fm->pinbuf, offset))
            goto map_error_and_free;
        offset += slist[i].count * 512;
    }

    free(slist);
    return fm;

map_error_and_free:
    map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
    pinpool_free(fm->pinbuf);
    free(slist);
    free(fm);

fallback:
    errno = 0;
    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    ret->map_error = map_error;
    return ret;
}


struct filemap *filemap_open_cuda_nvme(const char *fname)
{
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return NULL;

    struct filemap *ret = filemap_alloc_cuda_nvme(fd, fname);
    close(fd);
    return ret;
}


int filemap_write_cuda_nvme(struct filemap *fmap, int fd)
{
    int ret;
    struct stat64 st;
    int i;
    if (fmap->type != FILEMAP_TYPE_CUDA) {
        errno = EINVAL;
        return -1;
    }

    if (fstat64(fd, &st))
        return -1;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0) {
        return -1;
    }

    if ((ret = posix_fadvise(fd, 0, fmap->length, POSIX_FADV_DONTNEED)))
        return ret;


    struct nvme_dev_sector *slist;
    int sector_count = nvme_dev_get_sector_list(fd, &st, &slist, 0);

    if (sector_count < 0)
        return -1;

    unsigned long offset = 0;
    for (i = 0; i < sector_count; i++) {
        if (nvme_dev_gpu_write(devfd, slist[i].slba, slist[i].count,
                               fmap->pinbuf, offset))
        {
            free(slist);
            return -1;
        }

        offset += slist[i].count * 512;
    }

    //Updated modification and access times
    futimes(fd, NULL);

    free(slist);
    return 0;
}

const char * filemap_map_error_string(int map_error)
{
    switch (map_error) {
    case 0:
        return "No Error";
    case FILEMAP_FALLBACK_NOPERM_DEV:
        return "Could not open block device.";
    case FILEMAP_FALLBACK_DEV_NOT_NVME:
        return "Block device is not an NVMe device.";
    case FILEMAP_FALLBACK_NOPERM_FIBMAP:
        return "No permissions to use FIBMAP.";
    case FILEMAP_FALLBACK_IOCTL_ERROR:
        return "NVMe IOCTL Failed.";
    case FILEMAP_FALLBACK_PINBUF_TOO_SMALL:
        return "Pinned buffer is too small for the file.";
    default:
        return "Unkown Error.";
    }

}
