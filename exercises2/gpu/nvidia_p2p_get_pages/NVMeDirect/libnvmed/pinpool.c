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
//     Pinned Memory Pool
//
////////////////////////////////////////////////////////////////////////

#include "pinpool.h"
#include "fifo.h"

#include <nvmed/nv_pinbuf.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <sys/mman.h>

#include <stdlib.h>
#include <errno.h>

struct pin_buf_priv {
    struct pin_buf pub;
    CUDA_POINTER_ATTRIBUTE_P2P_TOKENS tokens;
};

static struct fifo *free_buf_fifo;
static struct pin_buf_priv *all_bufs;
static int buf_count;
static int devfd;

static int next_highest_pow2(int v)
{
    if (v < 128) return 128;

    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;

    return v;
}

static int pin(struct pin_buf_priv *pb)
{
    struct nv_gpu_mem gpumem = {
        .address = (__u64) pb->pub.address,
        .size = pb->pub.bufsize,
        .p2pToken = pb->tokens.p2pToken,
        .vaSpaceToken = pb->tokens.vaSpaceToken,
    };

    int ret = ioctl(devfd, NV_PIN_GPU_MEMORY, &gpumem);

    pb->pub.handle = gpumem.handle;

    return ret;
}

static int unpin(struct pin_buf_priv *pb)
{
    struct nv_gpu_mem gpumem = {
        .address = (__u64) pb->pub.address,
        .size = pb->pub.bufsize,
        .p2pToken = pb->tokens.p2pToken,
        .vaSpaceToken = pb->tokens.vaSpaceToken,
    };

    int ret = ioctl(devfd, NV_UNPIN_GPU_MEMORY, &gpumem);

    return ret;
}

static int init_pin_buf(struct pin_buf_priv *pb, size_t bufsize)
{
    pb->pub.mmap = NULL;
    pb->pub.bufsize = bufsize;
    int err;
    if ((err = cudaMalloc(&pb->pub.address, bufsize)) != cudaSuccess) {
        errno = ENOMEM;
//        fprintf(stderr, "cudaMalloc failed %s\n",  cudaGetErrorString(err));
        return -ENOMEM;
    }

    if (cuPointerGetAttribute(&pb->tokens, CU_POINTER_ATTRIBUTE_P2P_TOKENS,
                              (CUdeviceptr) pb->pub.address) != CUDA_SUCCESS)
    {
        errno = EIO;
        goto free_buf;
    }

    if (pin(pb))
        goto free_buf;

    return 0;

free_buf:
    cudaFree(pb->pub.address);
//    fprintf(stderr, "free cuda pinned memory\n");
    return -errno;
}

static void free_pin_buf(struct pin_buf_priv *pb)
{
    if (unpin(pb))
        return;

    cudaFree(pb->pub.address);
}

int pinpool_init(int count, size_t bufsize)
{
    int i;

    const int max_pin_size = 192*1024*1024;
    if (count == PINPOOL_MAX_BUFS)
        count = max_pin_size / bufsize;

    devfd = open("/dev/nv_pinbuf", O_RDWR);
    if (devfd < 0)
    {
        fprintf(stderr,"open pinpool failed\n");
        return -1;
    }

    free_buf_fifo = fifo_new(next_highest_pow2(count));
    if (!free_buf_fifo)
        goto close_dev;

    buf_count = count;
    all_bufs = malloc(count * sizeof(*all_bufs));
    if (all_bufs == NULL)
        goto free_fifo;

    for (i = 0; i < count; i++) {
        if (init_pin_buf(&all_bufs[i], bufsize))
            goto free_bufs;

        fifo_push(free_buf_fifo, &all_bufs[i].pub);
    }

    return 0;

free_bufs:
    fprintf(stderr,"free_bufs\n");
    i--;
    for (; i >= 0; i--)
        free_pin_buf(&all_bufs[i]);

    free(all_bufs);

free_fifo:
    fprintf(stderr,"create fifo failed\n");
    fifo_free(free_buf_fifo);

close_dev:
    close(devfd);
    return -1;
}

void pinpool_deinit(void)
{
    int i;
    for (i = 0; i < buf_count; i++)
        free_pin_buf(&all_bufs[i]);

    free(all_bufs);

    fifo_free(free_buf_fifo);

    close(devfd);
}

struct pin_buf *pinpool_alloc(void)
{
    return fifo_pop(free_buf_fifo);
}

void pinpool_free(struct pin_buf *p)
{
    if (p->mmap != NULL)
        pinpool_munmap(p);

    fifo_push(free_buf_fifo, p);
}

void *pinpool_mmap(struct pin_buf *p)
{
    int ret = ioctl(devfd, NV_SELECT_MMAP_MEMORY, p->handle);
    if (ret) {
        errno = -ret;
        return NULL;
    }

    void *addr = mmap(NULL, p->bufsize, PROT_READ | PROT_WRITE, MAP_SHARED, devfd, 0);
    if (addr == MAP_FAILED)
        return NULL;
    return addr;
}

int pinpool_munmap(struct pin_buf *p)
{
    if (p->mmap == NULL)
        return -EINVAL;

    int ret = munmap(p->mmap, p->bufsize);
    p->mmap = NULL;
    return ret;
}
