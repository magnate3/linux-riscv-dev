////////////////////////////////////////////////////////////////////////
//
// Copyright 2017 ESCAL, NC State University.
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
//   Author: Hung-Wei Tseng
//
//   Date:   April 25 2017
//
//   Description:
//   Front-end interface of DirectNVMe
//
//
////////////////////////////////////////////////////////////////////////
#include "pinpool.h"
#include "filemap.h"
#include "nvme_dev.h"
#include "nvmed.h"
#include <stdio.h>

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
#include <math.h>

#include <string.h>
#include <stdlib.h>

#include <pthread.h>
#include <linux/fiemap.h>
#include <nvmed/nv_pinbuf.h>
#include <linux/types.h>
#define FALLOC_FL_NO_HIDE_STALE  0x4

// Defining the number of I/O threads for each process.
#ifndef NUM_OF_THREADS
#define NUM_OF_THREADS 4
#endif

// The NVMe_Chunk must be smaller than the (2^(mdts))*512 Bytes of NVMe SSD. 
#ifndef NVMED_CHUNK
// NVMED_CHUNK should be 2^(NVMe SSD's mdts+3)
//#define NVMED_CHUNK 16384
#define NVMED_CHUNK 4096
#define NVMED_WRITE_CHUNK NVMED_CHUNK
#define INTEND_MDTS NVMED_CHUNK
#define DEVICE_MDTS NVMED_CHUNK
#endif
// Pinbuf size of the GPU -- K20 for 192MB. 
#define PINBUF_SIZE 192*1024*1024
#define GPU_DEVICE_MEM 5*1024*1024*1024
#ifdef DEBUG
#define DEBUG_PRINT(fmt, ...) fprintf(stderr, fmt,  __VA_ARGS__);
#else
#endif
int pinpoolfd;
struct filemap *filemap_cuda_nvmed(int fd,void *gpuMemPtr, size_t size, long file_offset, cudaStream_t stream);
static void free_cuda_nvme(struct filemap *fm);
int filemap_write_cuda_nvmed(struct filemap *fmap, int fd);

// Following is not open for public use
struct sector {
    unsigned long long slba;
    unsigned long long count;
};

#define FE_COUNT	8000
#define FE_FLAG_LAST	(1 <<  0)
#define FE_FLAG_UNKNOWN	(1 <<  1)
#define FE_FLAG_UNALLOC	(1 <<  2)
#define FE_FLAG_NOALIGN	(1 <<  8)

#define EXTENT_UNKNOWN (FE_FLAG_UNKNOWN | FE_FLAG_UNALLOC | FE_FLAG_NOALIGN)

struct fe_s {
	__u64 logical;
	__u64 physical;
	__u64 length;
	__u64 reserved64[2];
	__u32 flags;
	__u32 reserved32[3];
};

struct fm_s {
	__u64 start;
	__u64 length;
	__u32 flags;
	__u32 mapped_extents;
	__u32 extent_count;
	__u32 reserved;
};

struct fs_s {
	struct fm_s fm;
	struct fe_s fe[FE_COUNT];
};

#define FIEMAP	_IOWR('f', 11, struct fm_s)
#define SECTOR_OFFSET 9

// This function creates a list of block addresses for the requesting file.
static int get_sector_list(int fd, struct stat64 *st, struct sector **slist_p, size_t offset, size_t length)
{
    int blk_size = st->st_blksize >> SECTOR_OFFSET;
    size_t processed_size = 0;
    // Calculate the length of the whole file if length is not set
    if(length == 0)
    {
        length = st->st_size;
        length-=offset;
    }
    // Calculate the number of blocks for this file in the file system level.
    unsigned long num_blocks = (st->st_size + st->st_blksize - 1) / st->st_blksize;
    int i, j, err;
    int list_count = 1;
    // Allocate the list of block addresses
    struct sector *slist;
    *slist_p = slist = (struct sector *)malloc((st->st_blocks / (st->st_blksize >> SECTOR_OFFSET)*sizeof(struct sector)));
#ifdef DEBUG
    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);    
#endif
    struct fs_s fs;
    int chunk_offset = (int)log2(NVMED_CHUNK);
    memset(&fs, 0, sizeof(fs));
    fs.fm.length = length;
    fs.fm.flags  = FIEMAP_FLAG_SYNC;
    fs.fm.start = offset;
    fs.fm.extent_count = FE_COUNT;
    #ifdef DEBUG
    fprintf(stderr, "offset: 0x%x\n", offset);
    #endif
    __u64 chunk_num, length_of_the_extent, copied_length_of_the_extent;
    // Hopefully the file system supports FIEMAP so that we can get block addresses faster
    if (!(err = ioctl(fd, FIEMAP, &fs)))
    {
         #ifdef DEBUG
         fprintf(stderr, "FIEMAP %d\n",fs.fm.mapped_extents);
         #endif
         for (j = 0; j < fs.fm.mapped_extents; j++) 
         {
             copied_length_of_the_extent = 0;
             length_of_the_extent = fs.fe[j].length;
             slist->slba = fs.fe[j].physical >> SECTOR_OFFSET;
             #ifdef DEBUG
             fprintf(stderr, "log=%p phy=%p len=%llu flags=0x%x\n", fs.fe[j].logical,
		     fs.fe[j].physical, fs.fe[j].length, fs.fe[j].flags);
             #endif
             if(j == 0 && offset != 0)
             {
                 length_of_the_extent -= (offset-fs.fe[j].logical);
                 // The first sector, starting offset is within the extend.
                 if(list_count == 1) 
                 {
                     slist->slba = (fs.fe[j].physical + (offset - fs.fe[j].logical)) >> SECTOR_OFFSET;
                 }
             }
             chunk_num = slist->slba >> chunk_offset;
             while(copied_length_of_the_extent < length_of_the_extent)
             {
                 slist->count = (((++chunk_num) << chunk_offset) - slist->slba);
                 if(copied_length_of_the_extent + (slist->count << SECTOR_OFFSET) >= length_of_the_extent)
                 {
                     slist->count = (length_of_the_extent - copied_length_of_the_extent) >> SECTOR_OFFSET;
                     copied_length_of_the_extent += (slist->count << SECTOR_OFFSET);
                     processed_size += (slist->count << SECTOR_OFFSET);
                     #ifdef DEBUG
                     fprintf(stderr, "FIEMAP start: %p length: %llu copied %llu extent_length %llu %d\n", slist->slba, slist->count << SECTOR_OFFSET, copied_length_of_the_extent, length_of_the_extent, list_count);
                     #endif
                     if(processed_size >= length)
                     {
                         slist->count -= ((processed_size - length) >> SECTOR_OFFSET);
                         return list_count;
                     }
                     list_count++;
                     slist++;
                     break;
                        // The very last piece in the extent!
                 }
                 copied_length_of_the_extent += (slist->count << SECTOR_OFFSET);
                 #ifdef DEBUG
                 fprintf(stderr, "FIEMAP start: %p length: %llu copied %llu extent_length %llu\n", slist->slba, slist->count << SECTOR_OFFSET, copied_length_of_the_extent, length_of_the_extent);
                 #endif
                    
                 processed_size += (slist->count << SECTOR_OFFSET);
                 if(processed_size >= length)
                 {
                     slist->count -= ((processed_size - length) >> SECTOR_OFFSET);
                     return (list_count);
                 }
                 slist++;
                 slist->slba = chunk_num << chunk_offset;
                 list_count++;
            }
        }
    }
    else
    {

//        fprintf(stderr, "FIEMAP(%d) doesn't work, try FIBMAP\n");
        for (i = 0; i < num_blocks; i++) {
            unsigned long blknum = i;

            if (ioctl(fd, FIBMAP, &blknum) < 0)
                return -1;

            //Seems we can't transfer more than 65536 LBAs at once so
            // in that case we split it into multiple transfers
            if (i != 0 && blknum * blk_size == slist->slba + slist->count &&
                slist->count + blk_size <= 65536) {
                slist->count += blk_size;
                continue;
            }
            
            if (i != 0) {
                if(offset > 0)
                {
                    if((offset >> SECTOR_OFFSET) < slist->count) // starting address falls within this block
                    {
                        slist->slba += (offset >> SECTOR_OFFSET);
                        slist->count -= (offset >> SECTOR_OFFSET);
                        offset = 0;
                    }
                    else
                    {
                        offset -= (slist->count << SECTOR_OFFSET);
                        slist->slba = blknum * blk_size;
                        slist->count = blk_size;
                        continue;
                    }
                }
                processed_size += (slist->count << SECTOR_OFFSET);
                if(processed_size >= length)
                {
                    slist->count -=((processed_size-length) >> SECTOR_OFFSET);
                    return list_count;
                }

                slist++;
                list_count++;
            }

            slist->slba = blknum * blk_size;
            slist->count = blk_size;
        }
        if(processed_size < length)
        {
            if(processed_size + (slist->count << SECTOR_OFFSET) >= length)
            slist->count -=((processed_size + (slist->count << SECTOR_OFFSET) - length) >> SECTOR_OFFSET);
            return list_count;
        }
    }
    #ifdef DEBUG
    gettimeofday(&time_end, NULL);
    printf("HGProfile: FIEMAP %ld\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
    gettimeofday(&time_start, NULL);    
    #endif
    return list_count;
}

// Partition the pinbuf with the number of I/O threads
int nvmed_init(int threads)
{
    if(pinpool_init(threads, (PINBUF_SIZE)/threads))
    {
        fprintf(stderr,"Could not initialize pin pool");
        exit(1);
    }
    pinpoolfd = open("/dev/nv_pinbuf", O_RDWR);
    // Reset the semaphores
    ioctl(pinpoolfd, NV_NVMED_RESET, NULL);
    return 0;
}

// Deallocate the pinbuf
int nvmed_deinit()
{
    pinpool_deinit();
    close(pinpoolfd);
    return 0;
}

// Sending file data from the SSD to the GPU
size_t nvmed_send(int fd, void **gpuMemPtr, size_t size, unsigned long offset)
{
    #ifdef DEBUG
    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);
    #endif
    #ifdef PERFSTAT
    printf("nvmed_send\n");
    perfstats_print();
    #endif
    #ifdef DEBUG
    gettimeofday(&time_end, NULL);
    printf("HGProfile: open %ld\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
    #endif
    size_t ret;
    struct filemap *fm;
    if (fd < 0)
    {
        fprintf(stderr,"NVMeD file open error.\n");
        return -1;
    }
    if(offset % 512)
        fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");
    
#ifdef DEBUG
//    fprintf(stderr,"nvmed_send_st(%s,%p,%llu,%llu)\n",fname,*gpuMemPtr,size,offset);
#endif

    fm = filemap_cuda_nvmed(fd, *gpuMemPtr, size, offset, 0);

    if(*gpuMemPtr == NULL)
    {
        *gpuMemPtr = fm->data;
    }

    ret = fm->length;
    filemap_free(fm);
    #ifdef PERFSTAT
    printf("After nvmed_send\n");
    perfstats_print();
    #endif
    return ret;

}

// We partition the big output file in the nvmed_recv function. 
size_t nvmed_recv(int fd, void *gpuMemPtr, size_t size, unsigned long offset)
{
    struct pin_buf *buf = pinpool_alloc();
    struct filemap fm = {
        .data = gpuMemPtr,
        .length = size,
        .pinbuf = buf,
        .type = FILEMAP_TYPE_CUDA,
    };
    int errno;
    if(offset % 512)
        fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");

    if (errno = fallocate(fd, FALLOC_FL_NO_HIDE_STALE, 0, fm.length)) {

//    if (errno = posix_fallocate(fd, 0, fm.length)) {
            fprintf(stderr, "Could not fallocate the file %llu (%d), writing zeros instead: ",fm.length, errno);
            return -1;
    }

    if (filemap_write_cuda_nvmed(&fm, fd)) {
            fprintf(stderr, "Direct NVMeD writing error\n");
            return -1;
    }
    
    close(fd);
    pinpool_free(buf);
    return 0;
}


struct filemap *filemap_cuda_nvmed(int fd,void *gpuMemPtr, size_t size, long file_offset, cudaStream_t stream)
{
    int map_error;
    struct stat64 st;
    void *current;
    int i;
#ifdef DEBUG
    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);
    float *debug_array;
#endif
    if (fstat64(fd, &st))
        return NULL;
#ifdef DEBUG
        gettimeofday(&time_end, NULL);
        fprintf(stderr, "HGProfile: fstat %ld\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif

    struct sector *slist;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0) {
        if (devfd == -ENXIO)
        {
            fprintf(stderr, "NVMeD does not work on nvme device.\n");
            map_error = FILEMAP_FALLBACK_DEV_NOT_NVME;
        }
        else
        {
            fprintf(stderr, "NVMeD does not have device permission.\n");
            map_error = FILEMAP_FALLBACK_NOPERM_DEV;
        }
        goto fallback;
    }

    struct filemap *fm = malloc(sizeof(*fm));
    if (fm == NULL)
        return NULL;

    fm->map_error = 0;
    if(size == 0)
        fm->length = st.st_size;
    else
        fm->length = size;

    if(file_offset > 0 && size == 0)
        fm->length -= file_offset;

    if(gpuMemPtr == NULL)
    {
        // Make sure it's not larger than GPU device memory
        if ( fm->length > GPU_DEVICE_MEM)
        {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds available GPU memory size %llu\n",fm->length);
            free(fm);
            return NULL;
        }
        if (cudaMalloc(&fm->data, fm->length) != cudaSuccess) {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds current available GPU memory size, delay the task!\n");
            return fm;
        }
    }
    else
        fm->data = gpuMemPtr;

    fm->type = FILEMAP_TYPE_CUDA;
    fm->free = free_cuda_nvme;
#ifdef DEBUG
    gettimeofday(&time_start, NULL);
#endif

    int sector_count = get_sector_list(fd, &st, &slist, file_offset, size);
#ifdef DEBUG
    gettimeofday(&time_end, NULL);
    fprintf(stderr, "HGProfile: get %d sector list %ld\n",sector_count, ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif

    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        fprintf(stderr,"nvmed cannot get FILEMAP\n");
        goto free_and_fallback;
    }

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    fm->pinbuf = pinpool_alloc();


    current = fm->data;

#ifdef DEBUG
    fprintf(stderr,"nvmed_send(%p,%llu,%llu)\n",current,fm->length,file_offset);
#endif

    unsigned long offset = 0, copied = 0, this_sector=0;
    for (i = 0; i < sector_count; i++) {
        if(slist[i].slba == 0)
            break;
        // Make the LBA
        this_sector = slist[i].count * 512;
        // If the copied data is more than the BAR1 memory size, copying it out.
        if((offset + this_sector) > fm->pinbuf->bufsize)
        {
            if((copied + offset) > fm->length)
                break;
            else
            {
                cudaMemcpy(current, fm->pinbuf->address, offset, cudaMemcpyDeviceToDevice);
//                fprintf(stderr,"cudaMemcpy: %p <- %p, %llu\n",current, fm->pinbuf->address, offset);
//                fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);
            }
            copied += offset;
            current += offset;
            offset = 0;
        }
#ifdef DEBUG
        gettimeofday(&time_start, NULL);	
#endif
//      fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);

        if (map_error = nvme_dev_gpu_read(devfd, slist[i].slba, slist[i].count,
                              fm->pinbuf, offset))
        {
            fprintf(stderr, "GPU read ioctl error %s\n", strerror(map_error));
            map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
            pinpool_free(fm->pinbuf);
            goto free_and_fallback;
        }
#ifdef DEBUG
        gettimeofday(&time_end, NULL);
        fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);
        fprintf(stderr, "HGProfile: NVMeD read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
        offset += this_sector;
    }
    cudaMemcpy(current, fm->pinbuf->address, fm->length-copied, cudaMemcpyDeviceToDevice);
//        cudaMemcpyAsync(current, fm->pinbuf->address, fm->length-copied, cudaMemcpyDeviceToDevice, stream);
//        cudaStreamSynchronize(stream);
//    fprintf(stderr,"cudaMemcpy: %p <- %p, %llu, %llu\n",current, fm->pinbuf->address,  fm->length-copied, fm->pinbuf->bufsize);
#ifdef PERFSTAT
    fprintf(stdeer,"After CUDA memcpy\n");
    perfstats_print();
#endif
    pinpool_free(fm->pinbuf);

    return fm;

free_and_fallback:
    free(fm);

fallback:
    errno = 0;
//    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    fprintf(stderr,"NVMeD doesn't work\n");
//    ret->map_error = map_error;
    return NULL;
}

int filemap_write_cuda_nvmed(struct filemap *fm, int fd)
{
    int ret, i;
    struct stat64 st;
    // Hung-Wei: add a current pointer to the destination GPU memory address
    void *current;
    // Hung-Wei: maintain the original writing length
    unsigned long size=fm->length;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif

    if (fm->type != FILEMAP_TYPE_CUDA) {
        errno = EINVAL;
        return -1;
    }

    if (fstat64(fd, &st))
        return -1;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0) {
        return -1;
    }
    if ((ret = posix_fadvise(fd, 0, fm->length, POSIX_FADV_DONTNEED)))
        return ret;


//    struct nvme_dev_sector slist[st.st_blocks / (st.st_blksize / 512)];
    struct sector *slist;
    int sector_count = get_sector_list(fd, &st, &slist, 0, size);

    if (sector_count < 0)
        return -1;

    // Hung-Wei: maintaining a current pointer to the GPU source data address
    current = fm->data;

    unsigned long offset = 0, remaining=size, size_of_this_chunk=0;
    if(fm->pinbuf->bufsize < NVMED_CHUNK*512)
    {
        fprintf(stderr,"pinbuffer size too small");
        return -1;
    }
    for (i = 0; i < sector_count; i++) {
        // Hung-Wei: If everything in the pinbuffer is copied, copying it out.
        if(i == sector_count-1)
            size_of_this_chunk = fm->length - offset;
        else
            size_of_this_chunk = slist[i].count * 512;
        cudaMemcpy(fm->pinbuf->address, current, size_of_this_chunk, cudaMemcpyDeviceToDevice);
        current += size_of_this_chunk;
#ifdef DEBUG
        gettimeofday(&time_start, NULL);	
//	printf("HGProfile: writing %x %d %x %d\n",);	
#endif
        if (nvme_dev_gpu_write(devfd, slist[i].slba, slist[i].count,
                               fm->pinbuf, 0))
        {
            return -1;
        }
        offset += size_of_this_chunk;
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
	printf("HGProfile: write %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
    }

    //Updated modification and access times
    futimes(fd, NULL);

    return 0;
}


static void free_cuda_nvme(struct filemap *fm)
{
    pinpool_free(fm->pinbuf);

//    if (fm->filename != NULL)
//        free((void *) fm->filename);

    free(fm);
}


