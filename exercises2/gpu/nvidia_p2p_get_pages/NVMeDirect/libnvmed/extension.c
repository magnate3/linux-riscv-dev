size_t nvmed_send_direct(int fd, void **gpuMemPtr, size_t size, unsigned long offset)
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

    fm = filemap_direct_nvmed(fd, *gpuMemPtr, size, offset);

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

struct filemap *filemap_direct_nvmed(int fd,void *gpuMemPtr, size_t size, long file_offset)
{
    int map_error;
    struct stat64 st;
    void *current;
//#ifdef DEBUG
    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);
//#endif
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
        if ( fm->length > 5032116224)
        {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds available GPU memory size %llu\n",fm->length);
            free(fm);
            return NULL;
        }
            fprintf(stderr, "NULL point error!\n");
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
        fprintf(stderr,"nvmed cannot get FIBMAP\n");
        goto free_and_fallback;
    }

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
//    fm->pinbuf = pinpool_alloc();


    current = fm->data;

#ifdef DEBUG
    fprintf(stderr,"nvmed_send(%p,%llu,%llu)\n",current,fm->length,file_offset);
#endif

    unsigned long offset = 0, copied = 0, this_sector=0;
    for (int i = 0; i < sector_count; i++) {
        if(slist[i].slba == 0)
            break;
        this_sector = slist[i].count * 512;
        // If the copied data is more than the BAR1 memory size, copying it out.
/*        if((offset + this_sector) > fm->pinbuf->bufsize)
        {
            if((copied + offset) > fm->length)
                break;
            else
            {
                cudaMemcpy(current, fm->pinbuf->address, offset, cudaMemcpyDeviceToDevice);
                fprintf(stderr,"cudaMemcpy: %p <- %p, %llu\n",current, fm->pinbuf->address, offset);
                fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);
            }
            copied += offset;
            current += offset;
            offset = 0;
        }
*/
//#ifdef DEBUG
        gettimeofday(&time_start, NULL);	
//#endif
//      fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);

        if (map_error = nvme_dev_direct_read(devfd, slist[i].slba, slist[i].count,
                              (__u64)fm->data, offset))
        {
            fprintf(stderr, "Direct read ioctl error %s\n", strerror(map_error));
            map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
//            pinpool_free(fm->pinbuf);
            goto free_and_fallback;
        }
//#ifdef DEBUG
        gettimeofday(&time_end, NULL);
        fprintf(stderr, "HGProfile: direct read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
//#endif
        fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);
        offset += this_sector;
    }
//    cudaMemcpy(current, fm->pinbuf->address, fm->length-copied, cudaMemcpyDeviceToDevice);
//        cudaMemcpyAsync(current, fm->pinbuf->address, fm->length-copied, cudaMemcpyDeviceToDevice, stream);
//        cudaStreamSynchronize(stream);
//    fprintf(stderr,"cudaMemcpy: %p <- %p, %llu, %llu\n",current, fm->pinbuf->address,  fm->length-copied, fm->pinbuf->bufsize);
#ifdef PERFSTAT
    fprintf(stdeer,"After CUDA memcpy\n");
    perfstats_print();
#endif
//    pinpool_free(fm->pinbuf);

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
#if 0
volatile int number_delayed_tasks = 0;

struct nvmed_delayed_task
{
    int devfd;
    int fd;
    void *host_memory_address;
    void *gpu_memory_address;
    int status;
    void* (*func)(void *arg); // callback func
    void* args;
    int number_of_remaining_ssd_host_requests;
    int number_of_remaining_host_device_requests;
    struct filemap *fm;
    struct fifo *ssd_host_queue;
    struct fifo *host_device_queue;
    struct nvmed_request* requests;
    int threaded;
    void **gpuMemPtr;
};

struct nvmed_request
{
    int devfd;
    void *host_memory_address;
    void *gpu_memory_address;
    unsigned long slba;
    unsigned long count;
    unsigned long size;
};

struct nvmed_thread_parameter
{
    struct fifo *nvmed_incoming_requests_fifo;
    int *number_of_remaining_requests;
    struct fifo *nvmed_completed_requests_fifo;
    void *task;
};

inline void atomic_increment(volatile int *pw)
{
	  __asm (
	       "lock\n\t"
	       "incl %0":
	       "=m"(*pw): // output (%0)
	       "m"(*pw): // input (%1)
	       "cc" // clobbers
	       );
}

inline void atomic_decrement(volatile int *pw)
{
	  __asm (
	       "lock\n\t"
	       "decl %0":
	       "=m"(*pw): // output (%0)
	       "m"(*pw): // input (%1)
	       "cc" // clobbers
	       );
}
struct filemap *filemap_cuda_nvmed(int fd, const char *fname, void *gpuMemPtr, size_t size, long file_offset, cudaStream_t stream);
//struct filemap *filemap_cuda_nvmed_threaded(int fd, const char *fname, void *gpuMemPtr, size_t size, long file_offset);
int filemap_write_cuda_nvmed(struct filemap *fmap, int fd);
int filemap_write_cuda_nvmed_threaded(struct filemap *fm, int fd);
void *nvmed_send_ssd_host(void *x);
void *nvmed_read_thread(void *t);
void *nvmed_send_ssd_host_thread(void *t);
void *nvmed_send_host_device_thread(void *t);
__thread struct fifo *delayed_task_queue;
struct nvmed_delayed_task* create_delayed_task(int fd, struct filemap *fm, void **gpuMemPtr, void* (*func)(void *arg), void *args);
//void *nvmed_send_ssd_host(struct nvmed_delayed_task *delayed_task);
void *nvmed_wakeup_delayed_task(void *t);
struct filemap *filemap_local_nvmed(int fd, const char *fname,void *memPtr, size_t size, long file_offset, cudaStream_t stream);
int filemap_write_local_nvmed(struct filemap *fmap, int fd);
inline int submit_request(struct fifo *queue, struct nvmed_request *request, int devfd, size_t slba, size_t count, size_t size, void *host_memory_address, void *gpu_memory_address)
{
    request->devfd = devfd;
    request->host_memory_address = host_memory_address;
    request->gpu_memory_address = gpu_memory_address;
    request->slba = slba;
    request->count = count;
    request->size = size;
    fifo_push(queue, request);
}


int pinpool_reset()
{
    if(!pinpoolfd)
        pinpoolfd = open("/dev/donard_pinbuf", O_RDWR);
    ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RESET, NULL);
    return 0;
}

int nvmed_init_mp(int threads)
{
    if(threads == 0)
    {
#if INTEND_MDTS > NVMED_CHUNK
        if(pinpool_init(NUM_OF_THREADS, (INTEND_MDTS*512)))
#else
        if(pinpool_init(NUM_OF_THREADS, (NVMED_CHUNK*512)))
#endif
        {
            fprintf(stderr,"Could not initialize pin pool");
            exit(1);
        }
    }
    else
    {
        if(pinpool_init(1, (192*1024*1024)/threads))
        {
            fprintf(stderr,"Could not initialize pin pool");
            exit(1);
        }
    }
//    delayed_task_queue = fifo_new(16);
    return 0;
}


size_t nvmed_send_callback(const char *fname, void **gpuMemPtr,size_t size,unsigned long offset, cudaStream_t stream, int threaded, void* (*func)(void *arg), void *args)
{
    int fd = open(fname, O_RDONLY);
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
    fprintf(stderr,"nvmed_send(%s,%p,%llu,%llu)\n",fname,*gpuMemPtr,size,offset);
#endif
    if(threaded == 1)
    {
        // Acquire exclusive lock to prevent performance loss by delayed tasks.
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_EXCLUDE, NULL);
        fm = filemap_cuda_nvmed_threaded(fd, fname, *gpuMemPtr, size, offset);
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RELEASE, NULL);
    }
    else
    {
        fm = filemap_cuda_nvmed(fd, fname, *gpuMemPtr, size, offset, stream);
    }

    if(*gpuMemPtr == NULL)
        *gpuMemPtr = fm->data;
//    else
//        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RELEASE, NULL);
    ret = fm->length;
    if(*gpuMemPtr == NULL) // Delayed
    {
        // Don't let the delayed task affects current transactions. 
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_EXCLUDE, NULL);
        struct nvmed_delayed_task *delayed_task = create_delayed_task(fd, fm, gpuMemPtr, func, args);
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RELEASE, NULL);
        do
        {
            ioctl(pinpoolfd, DONARD_IOCTL_NVMED_ACCQUIRE, NULL);
            nvmed_wakeup_delayed_task(delayed_task);
        }
        while (fm->data == NULL);
        free(delayed_task);
        *gpuMemPtr = fm->data;
    }
    filemap_free(fm);
    close(fd);
    return ret;
    return 0;
}

void *nvmed_send_async(const char *fname, void **gpuMemPtr,size_t size,unsigned long offset, cudaStream_t stream, int threaded, void* (*func)(void *arg), void *args)
{
    int fd = open(fname, O_RDONLY);
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
    fprintf(stderr,"nvmed_send(%s,%p,%llu,%llu)\n",fname,*gpuMemPtr,size,offset);
#endif
    if(threaded == 1)
    {
        // Acquire exclusive lock to prevent performance loss by delayed tasks.
        fm = filemap_cuda_nvmed_threaded(fd, fname, *gpuMemPtr, size, offset);
    }
    else
    {
        fm = filemap_cuda_nvmed(fd, fname, *gpuMemPtr, size, offset, stream);
    }

    if(*gpuMemPtr == NULL)
        *gpuMemPtr = fm->data;
//    else
//        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RELEASE, NULL);
    ret = fm->length;
    if(*gpuMemPtr == NULL) // Delayed
    {
        // Don't let the delayed task affects current transactions. 
//        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_EXCLUDE, NULL);
        struct nvmed_delayed_task *delayed_task = create_delayed_task(fd, fm, gpuMemPtr, func, args);
        if(!delayed_task_queue)
            delayed_task_queue = fifo_new(16);

//    fprintf(stderr, "pushing %p to queue\n",delayed_task);
        fifo_push(delayed_task_queue, delayed_task);
//        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RELEASE, NULL);
    }
    else
        close(fd);
    return NULL;
}

size_t nvmed_send(const char *fname, void **gpuMemPtr, size_t size, unsigned long offset)
{
            #ifdef DEBUG
                struct timeval time_start, time_end;
                gettimeofday(&time_start, NULL);
            #endif
#ifdef PERFSTAT
printf("nvmed_send\n");
perfstats_print();
#endif
    int fd = open(fname, O_RDONLY);
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
    fprintf(stderr,"nvmed_send_st(%s,%p,%llu,%llu)\n",fname,*gpuMemPtr,size,offset);
#endif

    fm = filemap_cuda_nvmed(fd, fname, *gpuMemPtr, size, offset, 0);

    if(*gpuMemPtr == NULL)
    {
        *gpuMemPtr = fm->data;
    }
    if(*gpuMemPtr == NULL) // Delayed
    {
        fprintf(stderr,"Insufficient GPU memory for NVMeD. Use memory-pipeline instead.\n");
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_EXCLUDE, NULL);
        struct nvmed_delayed_task *delayed_task = create_delayed_task(fd, fm, gpuMemPtr, NULL, NULL);
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_RELEASE, NULL);
        ioctl(pinpoolfd, DONARD_IOCTL_NVMED_ACCQUIRE, NULL);
        nvmed_wakeup_delayed_task(delayed_task);
        free(delayed_task);
        *gpuMemPtr = fm->data;
    }
    else
        close(fd);
    ret = fm->length;
    filemap_free(fm);
#ifdef PERFSTAT
printf("After nvmed_send\n");
perfstats_print();
#endif
    return ret;

}

size_t nvmed_send_stream(const char *fname, void **gpuMemPtr, size_t size, unsigned long offset, cudaStream_t stream)
{
    int fd = open(fname, O_RDONLY);
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
    fprintf(stderr,"nvmed_send(%s,%p,%llu,%llu)\n",fname,*gpuMemPtr,size,offset);
#endif

    fm = filemap_cuda_nvmed(fd, fname, *gpuMemPtr, size, offset, stream);

    if(*gpuMemPtr == NULL)
        *gpuMemPtr = fm->data;
    ret = fm->length;
    filemap_free(fm);
    close(fd);
    return ret;

}

size_t nvmed_send_threaded(const char *fname, void **gpuMemPtr, size_t size, unsigned long offset) 
{
    int fd = open(fname, O_RDONLY);
    size_t ret;
    struct filemap *fm;
    if (fd < 0)
    {
        fprintf(stderr,"nvmed file open error\n");
        return -1;
    }
    if(offset % 512)
        fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");

    fm = filemap_cuda_nvmed_threaded(fd, fname, *gpuMemPtr, size, offset);
    if(*gpuMemPtr == NULL)
        *gpuMemPtr = fm->data;

    ret = fm->length;
    if(*gpuMemPtr == NULL) // Delayed
        return ret;
    filemap_free(fm);
    close(fd);
    return ret;
}

size_t nvmed_send_local(const char *fname, void *memPtr, size_t size, unsigned long offset)
{
#ifdef PERFSTAT
printf("nvmed_send_local\n");
perfstats_print();
#endif
            #ifdef DEBUG
                struct timeval time_start, time_end;
                gettimeofday(&time_start, NULL);
            #endif
    int fd = open(fname, O_RDONLY);
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
    fprintf(stderr,"nvmed_send_st(%s,%p,%llu,%llu)\n",fname,memPtr,size,offset);
#endif

    fm = filemap_local_nvmed(fd, fname, memPtr, size, offset, 0);
#ifdef PERFSTAT
printf("After nvmed_send_local\n");
perfstats_print();
#endif

    close(fd);
    ret = fm->length;
    filemap_free(fm);
    return ret;

}
size_t nvmed_recv_local(const char *fname, void *memPtr, size_t size, unsigned long offset)
{
    struct filemap fm = {
        .data = memPtr,
        .length = size,
        .type = FILEMAP_TYPE_LOCAL,
    };
    if(offset % 512)
        fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");

    int fd = open(fname, O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (fd < 0) {
            fprintf(stderr, "Error opening file '%s': %s\n", fname,
                    strerror(errno));
            return -1;
    }

    if (fallocate(fd, FALLOC_FL_NO_HIDE_STALE, 0, fm.length)) {
            fprintf(stderr, "Could not fallocate the file, writing zeros instead: ");
            return -1;
    }

    if (filemap_write_local_nvmed(&fm, fd) < 0) {
            fprintf(stderr, "Direct NVMeD writing error\n");
            return -1;
    }
    fsync(fd);
    close(fd);
    return 0;
}

// We partition the big output file in the nvmed_recv function. 
size_t nvmed_recv(const char *fname, void *gpuMemPtr, size_t size, unsigned long offset)
{
    struct pin_buf *buf = pinpool_alloc();
    struct filemap fm = {
        .data = gpuMemPtr,
        .length = size,
        .pinbuf = buf,
        .type = FILEMAP_TYPE_CUDA,
    };
    if(offset % 512)
        fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");

    int fd = open(fname, O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (fd < 0) {
            fprintf(stderr, "Error opening file '%s': %s\n", fname,
                    strerror(errno));
            return -1;
    }

    if (fallocate(fd, FALLOC_FL_NO_HIDE_STALE, 0, fm.length)) {
            fprintf(stderr, "Could not fallocate the file, writing zeros instead: ");
            return -1;
    }

    if (filemap_write_cuda_nvmed(&fm, fd)) {
            fprintf(stderr, "Direct NVMeD writing error\n");
            return -1;
    }
    
    close(fd);
    pinpool_free(buf);
    nvmed_wakeup_queue();
//    int devfd = open("/dev/donard_pinbuf", O_RDWR);
//    ioctl(devfd, DONARD_IOCTL_NVMED_ACCQUIRE, NULL);
    return 0;
}

// We partition the big output file in the nvmed_recv function. 
size_t nvmed_recv_threaded(const char *fname, void *gpuMemPtr, size_t size, unsigned long offset)
{
//    struct pin_buf *buf = pinpool_alloc();
    int bytes=0, remaining = 0;
    struct filemap fm = {
        .data = gpuMemPtr,
        .length = size,
//        .pinbuf = buf,
        .type = FILEMAP_TYPE_CUDA,
    };

    int fd = open(fname, O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (fd < 0) {
            fprintf(stderr, "Error opening file '%s': %s\n", fname,
                    strerror(errno));
            return -1;
    }

    if (fallocate(fd, FALLOC_FL_NO_HIDE_STALE, 0, fm.length)) {
            fprintf(stderr, "Could not fallocate the file, writing zeros instead: ");
            return -1;
    }
    if(offset % 512)
            fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");

    if (filemap_write_cuda_nvmed_threaded(&fm, fd)) {
            fprintf(stderr, "Direct NVMeD writing error\n");
            return -1;
    }
    
    bytes = size;
    close(fd);
    nvmed_wakeup_queue();
//    nvmed_wakeup_delayed_task(NULL);
//    pinpool_free(buf);
    return 0;
}



static void free_cuda_nvme_threaded(struct filemap *fm)
{
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



struct filemap *filemap_cuda_nvmed_threaded(int fd, const char *fname, void *gpuMemPtr, size_t size, long file_offset)
{
    int map_error;
    struct stat64 st;

    struct nvmed_request *requests;
    int i;
    pthread_t threads[NUM_OF_THREADS];
    struct fifo *nvmed_requests_fifo;
    struct nvmed_thread_parameter thread_info;
    volatile int number_of_remaining_requests;
    void *current;
    struct sector current_sector;

#ifdef DEBUG
    struct timeval time_start, time_end;
    gettimeofday(&time_start, NULL);	
#endif

    if (fstat64(fd, &st))
        return NULL;

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
    if(size == 0)
        fm->length = st.st_size;
    else
        fm->length = size;

    if(file_offset > 0 && size == 0)
        fm->length -= file_offset;

    if(gpuMemPtr == NULL)
    {
        if ( fm->length > 5032116224)
        {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds available GPU memory size\n");
            free(fm);
            return NULL;
        }
        if (cudaMalloc(&fm->data, fm->length) != cudaSuccess) {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds available GPU memory size, delay the task!\n");
//            free(fm);
            return fm;
        }
    }
    else
        fm->data = gpuMemPtr;

    fm->type = FILEMAP_TYPE_CUDA;
    fm->free = free_cuda_nvme_threaded;
    copy_filename(fm, fname);

    // Hung-Wei: Allocate GPU global memory for the input data if NULL is passed

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    requests = (struct nvmed_request *)calloc(num_blocks, sizeof(struct nvmed_request));
    number_of_remaining_requests = 1;
    int number_of_fifo_entries = (int)pow(2,ceil(log2(num_blocks*65536/NVMED_CHUNK)));
    nvmed_requests_fifo = fifo_new(number_of_fifo_entries);
    thread_info.number_of_remaining_requests = &number_of_remaining_requests;
    thread_info.nvmed_incoming_requests_fifo = nvmed_requests_fifo;
    current = fm->data;

//    for(int i = 0; i < NUM_OF_THREADS; i++)
    for(int i = 0; i < NUM_OF_READ_THREADS; i++)
    {
            pthread_create(&threads[i], NULL, nvmed_read_thread, &thread_info);
    }
#ifdef DEBUG
    fprintf(stderr,"nvmed_send(%p,%llu,%llu)\n",current,fm->length,file_offset);
#endif
    memset(&current_sector, 0, sizeof(struct sector));

    int sector_count = generate_requests(fd, &st, file_offset, fm->length, nvmed_requests_fifo, requests, NULL, fm->data, &number_of_remaining_requests);
    atomic_decrement(&number_of_remaining_requests);
    
    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        goto free_and_fallback;
    }
//    for(int i = 0; i < NUM_OF_THREADS; i++)
    for(int i = 0; i < NUM_OF_READ_THREADS; i++)
        pthread_join(threads[i], NULL);

    fifo_free(nvmed_requests_fifo);
    free(requests);
    return fm;

free_and_fallback:
    free(requests);
    free(fm);

fallback:
    errno = 0;
    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    ret->map_error = map_error;
    return ret;
}

void *nvmed_read_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *request;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
            #if INTEND_MDTS>NVMED_CHUNK
                int remaining_data;
            #endif
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    pinbuf = pinpool_alloc();
    while(*number_of_remaining_requests > 0)
    {
        if((request = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
//            fprintf(stderr,"nvmed_read_task(%p,%p,%llu,%d)\n",request->host_memory_address, request->gpu_memory_address, request->size, request->count);
            #endif
            #if INTEND_MDTS>NVMED_CHUNK
            offset = 0;
//            fprintf(stderr,"nvmed_read_task(%p,%p,%llu,%d)\n",request->host_memory_address, request->gpu_memory_address, request->size, request->count);
            for(remaining_data = request->count;remaining_data > 0; remaining_data-=NVMED_CHUNK)
            {
//                fprintf(stderr,"nvmed_read_task(%p,%d,%d)\n",request->slba, NVMED_CHUNK-1, offset);
                if (nvme_dev_gpu_read(request->devfd, request->slba, NVMED_CHUNK-1, pinbuf, offset))
                {
                    int map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
                    pinpool_free(pinbuf);
                    fprintf(stderr,"nvmed read error first\n");
                    return NULL;
                }
                request->slba+=NVMED_CHUNK;
                offset+=(NVMED_CHUNK << 9);
            }
            if(remaining_data < 0)
            {
                if (nvme_dev_gpu_read(request->devfd, request->slba, NVMED_CHUNK+remaining_data-1, pinbuf, offset))
                {
                    int map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
                    pinpool_free(pinbuf);
                    fprintf(stderr,"nvmed read error here\n");
                    return NULL;
                }
            }
                
            #else

            if (nvme_dev_gpu_read(request->devfd, request->slba, request->count, pinbuf, offset))
            {
                int map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
                pinpool_free(pinbuf);
                fprintf(stderr,"nvmed read error\n");
                return NULL;
            }
            #endif
            cudaMemcpy(request->gpu_memory_address, pinbuf->address, request->size , cudaMemcpyDeviceToDevice);
            atomic_decrement(number_of_remaining_requests);
            #ifdef DEBUG
            gettimeofday(&time_end, NULL);
            printf("HGProfile: threaded read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            #endif
        }
    }
    pinpool_free(pinbuf);
    fifo_close(nvmed_requests_fifo);
    return NULL;
}
/*
void *nvmed_copy_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *task;
    struct fifo *nvmed_copy_tasks_fifo = t;
    unsigned long offset = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(number_of_remaining_requests > 0)
    {
        if((task = (struct nvmed_request *)fifo_pop(nvmed_copy_tasks_fifo))!=NULL)
        {
#ifdef DEBUG
    gettimeofday(&time_start, NULL);	
#endif
            cudaMemcpy(task->dest, task->pinbuf->address, task->size , cudaMemcpyDeviceToDevice);
            pinpool_free(task->pinbuf);
            atomic_decrement(&number_of_remaining_requests);
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
	printf("HGProfile: threaded gpu %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif

        }
    }
    fifo_close(nvmed_copy_tasks_fifo);
    fifo_close(nvmed_requests_fifo);
    return NULL;
}
*/

/*
 * Map a file by DMAing it's contents to local memory,
 *  using the user_io IOCTL from an nvme device.
 *  In practice, this is not a smart thing to do. It's
 *  only here as a test/example for the NVME IOCTL.
 */
static void free_local_nvme(struct filemap *fm)
{
//    free(fm->data);
//    if (fm->filename != NULL)
//        free((void *) fm->filename);
    free(fm);
}

void *nvmed_host_pipeline_read_thread(void *t);
void *nvmed_host_pipeline_copy_thread(void *t);

unsigned long nvmed_host_pipeline_send(const char *fname, void **gpuMemPtr, size_t size, size_t offset, void *hostMemPtr)
{
    int map_error;
    struct stat64 st;
    void *current;
    void *current_gpu_memory_pointer;
//    void *hostMemPtr;
    int i;
    pthread_t threads[NUM_OF_THREADS];
    struct fifo *nvmed_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo;
    struct nvmed_request *requests;
    struct nvmed_thread_parameter thread_info;
    volatile int number_of_remaining_requests;
    size_t ret;
    int number_of_fifo_entries;
    struct sector current_sector;
    int sector_count = 0;
    int set_free = 0;
    #ifdef DEBUG
    struct timeval time_start, time_end;
    #endif
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return -1;

    if (fstat64(fd, &st))
        return NULL;

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    number_of_fifo_entries = (int)pow(2,ceil(log2(num_blocks*65536/NVMED_CHUNK)));
    int blk_size = st.st_blksize / 512;

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
    if(size == 0)
        fm->length = st.st_size;
    else
        fm->length = size;
    fm->type = FILEMAP_TYPE_LOCAL;
    fm->free = free_local_nvme;
    copy_filename(fm, fname);

    // Hung-Wei: Allocate GPU global memory for the input data if NULL is passed
    if(*gpuMemPtr == NULL)
    {
//        fprintf(stderr, "Send via delayed task\n");
//        create_delayed_task(devfd, fd, fm, NULL);
//        *gpuMemPtr = fm->data;
//        ret = fm->length;
//        fm->free = free_cuda_nvme_threaded;
//        filemap_free(fm);
//        free(fm);
//        return ret;

        if (cudaMalloc(&fm->data, fm->length) != cudaSuccess) {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds available GPU memory size\n");
            free(fm);
            return -1;
        }
    }
    else
        fm->data = *gpuMemPtr;
    
    if (fm->data == NULL)
        goto exit_error_free;
    if(hostMemPtr == NULL)
    {
        cudaMallocHost((void **)&hostMemPtr, fm->length);
        set_free = 1;
    }
    requests = (struct nvmed_request *)calloc(number_of_fifo_entries, sizeof(struct nvmed_request));

    number_of_remaining_requests = 1;
    nvmed_requests_fifo = fifo_new(number_of_fifo_entries);
    nvmed_memcpy_requests_fifo = fifo_new(number_of_fifo_entries);
    thread_info.number_of_remaining_requests = &number_of_remaining_requests;
    thread_info.nvmed_incoming_requests_fifo = nvmed_requests_fifo;
    thread_info.nvmed_completed_requests_fifo = nvmed_memcpy_requests_fifo;
//    for(int i = 0; i < NUM_OF_THREADS; i++)
    for(int i = 0; i < NUM_OF_READ_THREADS; i++)
    {
#ifdef singleStage
        if(i < 1)
            pthread_create(&threads[i], NULL, nvmed_host_pipeline_copy_thread, &thread_info);
        else
#endif
            pthread_create(&threads[i], NULL, nvmed_host_pipeline_read_thread, &thread_info);
    }

    sector_count = generate_requests(fd, &st, offset, fm->length, nvmed_requests_fifo, requests, hostMemPtr, fm->data, &number_of_remaining_requests);
    atomic_decrement(&number_of_remaining_requests);
//    fprintf(stderr,"host: %p, gpu: %p\n",current, current_gpu_memory_pointer);


    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        goto free_and_fallback;
    }

//    for(int i = 0; i < NUM_OF_THREADS; i++)
    for(int i = 0; i < NUM_OF_READ_THREADS; i++)
        pthread_join(threads[i], NULL);

    ret = fm->length;
    free(requests);
    filemap_free(fm);
    close(fd);
    if(set_free)
        cudaFreeHost(hostMemPtr);
    return ret;

exit_error_free:
    free(requests);
    free(fm);
    return 0;

free_and_fallback:
    free(requests);
    free(fm);

fallback:
    errno = 0;
    return 0;
}



void *nvmed_host_pipeline_read_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *task;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo = thread_info->nvmed_completed_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((task = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
            fprintf(stderr,"nvmed_read_task(%p, %p,%p,%llu,%d)\n",task->slba, task->host_memory_address, task->gpu_memory_address, task->size, task->count);
            #endif
            if (nvme_dev_read(task->devfd, task->slba, task->count, task->host_memory_address)) {
//                map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
//                goto free_and_fallback;
                    fprintf(stderr,"NVMe Read Error\n");
            }
            #ifdef DEBUG
            gettimeofday(&time_end, NULL);
            printf("HGProfile: pipeline read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            gettimeofday(&time_start, NULL);	
            #endif
#define singleStage 1
#ifdef singleStage
        cudaMemcpy(task->gpu_memory_address, task->host_memory_address, task->size , cudaMemcpyHostToDevice);
        atomic_decrement(number_of_remaining_requests);
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
//	printf("HGProfile: pipeline memcpy %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
#else
        fifo_push(nvmed_memcpy_requests_fifo, task);
#endif
        }
    }
    fifo_close(nvmed_requests_fifo);
    return NULL;
}

void *nvmed_host_pipeline_copy_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *task;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo = thread_info->nvmed_completed_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((task = (struct nvmed_request *)fifo_pop(nvmed_memcpy_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);
            #endif
            cudaMemcpy(task->gpu_memory_address, task->host_memory_address, task->size, cudaMemcpyHostToDevice);
            atomic_decrement(number_of_remaining_requests);
            #ifdef DEBUG
    	    gettimeofday(&time_end, NULL);
    	    //	printf("HGProfile: pipeline cudaMemcpy %d %p %d %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)), task->gpu_memory_address, task->size, task->slba);
	    printf("HGProfile: pipeline cudaMemcpy %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
	    #endif
        }
    }
    fifo_close(nvmed_requests_fifo);
    fifo_close(nvmed_memcpy_requests_fifo);
    return NULL;
}

void *nvmed_write_thread(void *t);
int filemap_write_cuda_nvmed(struct filemap *fmap, int fd)
{
    int ret;
    struct stat64 st;
    // Hung-Wei: add a current pointer to the destination GPU memory address
    void *current;
    // Hung-Wei: maintain the original writing length
    unsigned long size=fmap->length;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif

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


//    struct nvme_dev_sector slist[st.st_blocks / (st.st_blksize / 512)];
    struct sector *slist;
    int sector_count = get_sector_list(fd, &st, &slist, 0, size);

    if (sector_count < 0)
        return -1;

    // Hung-Wei: maintaining a current pointer to the GPU source data address
    current = fmap->data;

    unsigned long offset = 0, remaining=size, size_of_this_chunk=0;
    if(fmap->pinbuf->bufsize < NVMED_CHUNK*512)
    {
        fprintf(stderr,"pinbuffer size too small");
        return -1;
    }
    for (int i = 0; i < sector_count; i++) {
        // Hung-Wei: If everything in the pinbuffer is copied, copying it out.
        if(i == sector_count-1)
            size_of_this_chunk = fmap->length - offset;
        else
            size_of_this_chunk = slist[i].count * 512;
        cudaMemcpy(fmap->pinbuf->address, current, size_of_this_chunk, cudaMemcpyDeviceToDevice);
        current += size_of_this_chunk;
#ifdef DEBUG
        gettimeofday(&time_start, NULL);	
//	printf("HGProfile: writing %x %d %x %d\n",);	
#endif
        if (nvme_dev_gpu_write(devfd, slist[i].slba, slist[i].count,
                               fmap->pinbuf, 0))
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

int filemap_write_cuda_nvmed_threaded(struct filemap *fm, int fd)
{
    int ret;
    struct stat64 st;
    // Hung-Wei: add a current pointer to the destination GPU memory address
    void *current;
    pthread_t threads[NUM_OF_THREADS];
    struct fifo *nvmed_requests_fifo;
    struct nvmed_request *requests;
    struct nvmed_thread_parameter thread_info;
    // Hung-Wei: maintain the original writing length
    unsigned long size=fm->length;
    volatile int number_of_remaining_requests;
    int current_request = 0;
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
    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    number_of_remaining_requests = 1;
    int number_of_fifo_entries = (int)pow(2,ceil(log2(num_blocks*65536/NVMED_CHUNK)));
    nvmed_requests_fifo = fifo_new(number_of_fifo_entries);
    requests = (struct nvmed_request *)calloc(number_of_fifo_entries, sizeof(struct nvmed_request));
    thread_info.number_of_remaining_requests = &number_of_remaining_requests;
    thread_info.nvmed_incoming_requests_fifo = nvmed_requests_fifo;
    current = fm->data;
    for(int i = 0; i < NUM_OF_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, nvmed_write_thread, &thread_info);
    }

    int sector_count = generate_requests(fd, &st, 0, fm->length, nvmed_requests_fifo, requests, NULL, fm->data, &number_of_remaining_requests);
    atomic_decrement(&number_of_remaining_requests);

    if (sector_count < 0)
        return -1;

    for(int i = 0; i < NUM_OF_THREADS; i++)
        pthread_join(threads[i], NULL);
    
    free(requests);
    //Updated modification and access times
    futimes(fd, NULL);

    return 0;
}

void *nvmed_write_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *request;
//    struct fifo *nvmed_requests_fifo = t;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    pinbuf = pinpool_alloc();
    unsigned long offset = 0;
    int remaining;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((request = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
//            fprintf(stderr,"nvmed_recv_task_thread(%p,%llu,%x)\n",request->gpu_memory_address,request->size,request->slba);
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
            #endif
            cudaMemcpy(pinbuf->address, request->gpu_memory_address, request->size , cudaMemcpyDeviceToDevice);
            #ifdef DEBUG
    	    gettimeofday(&time_end, NULL);
    	    printf("HGProfile: threaded gpu %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            gettimeofday(&time_start, NULL);	
            #endif
#if INTEND_MDTS > NVMED_CHUNK
        offset = 0;
        for(remaining = request->count; remaining > 0; remaining -= NVMED_CHUNK)
        {
//            fprintf(stderr,"nvmed_recv_task_thread(%p,%d)\n",request->slba,offset);
        if (nvme_dev_gpu_write(request->devfd, request->slba, NVMED_CHUNK-1,
                               pinbuf, offset))
        {
            return -1;
        }
        request->slba+=NVMED_CHUNK;
        offset += NVMED_CHUNK << 9;
        }
        if(remaining < 0)
        {
        if (nvme_dev_gpu_write(request->devfd, request->slba, NVMED_CHUNK-1+remaining,
                               pinbuf, offset))
            return -1;
        }
#else
            if (nvme_dev_gpu_write(request->devfd, request->slba, request->count, pinbuf, offset))
            {
                int map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
                pinpool_free(pinbuf);
                fprintf(stderr,"nvmed write error\n");
                return NULL;
            }
#endif
            #ifdef DEBUG
    	    gettimeofday(&time_end, NULL);
    	    printf("HGProfile: write %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
    	    #endif
            atomic_decrement(number_of_remaining_requests);
        }
    }
    pinpool_free(pinbuf);
    fifo_close(nvmed_requests_fifo);
    return NULL;
}

struct filemap *filemap_alloc_cuda_nvmed(int fd, const char *fname, size_t size, unsigned long file_offset)
{
    int map_error;
    struct stat64 st;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    // Hung-Wei: add a current pointer to the destination GPU memory address
    void *current;
    if (fstat64(fd, &st))
        return NULL;

    struct sector *slist;
//    slist = (struct sector *)malloc((st.st_blocks / (st.st_blksize / 512)*sizeof(struct sector)));

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
    if(size == 0)
        fm->length = st.st_size;
    else
        fm->length = size;
    fm->type = FILEMAP_TYPE_CUDA;
    fm->free = free_cuda_nvme;
    copy_filename(fm, fname);

    int sector_count = get_sector_list(fd, &st, &slist, file_offset, size);

    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        goto free_and_fallback;
    }

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    fm->pinbuf = pinpool_alloc();

    // Hung-Wei: Allocate GPU global memory for the input data
    if (cudaMalloc(&fm->data, fm->length) != cudaSuccess) {
        errno = ENOMEM;
        fprintf(stderr, "File exceeds available GPU memory size\n");
        free(fm);
        return NULL;
    }

    current = fm->data;

    unsigned long offset = 0, copied = 0, size_of_this_sector=0;
    for (int i = 0; i < sector_count; i++) {
        // How big is the upcoming sector
        size_of_this_sector = slist[i].count * 512;
       // If the copied data is more than the pinbuffer size, copying it out.
        if((offset + size_of_this_sector) > fm->pinbuf->bufsize) {
            if((copied + offset) >= fm->length)
                break;
            else
            {
#ifdef DEBUG
        gettimeofday(&time_start, NULL);	
#endif
                cudaMemcpy(current, fm->pinbuf->address, offset, cudaMemcpyDeviceToDevice);
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
	printf("HGProfile: copy overhead %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
            }
            current += offset;
            copied += offset;
            offset = 0;
        }
#ifdef DEBUG
    gettimeofday(&time_start, NULL);	
#endif
        if (nvme_dev_gpu_read(devfd, slist[i].slba, slist[i].count,
                              fm->pinbuf, offset))
        {
            map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
            pinpool_free(fm->pinbuf);
            goto free_and_fallback;
        }
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
	printf("HGProfile: read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
        // Hung-Wei: update the offset
        offset += size_of_this_sector;
        //Original donard
        //offset += slist[i].count * 512;
    }
    // copy the last pieces into the destionation
    cudaMemcpy(current, fm->pinbuf->address, fm->length-copied, cudaMemcpyDeviceToDevice);

    return fm;

free_and_fallback:
    free(fm);

fallback:
    errno = 0;
    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    ret->map_error = map_error;
    return ret;
}

struct nvmed_delayed_task *create_delayed_task(int fd, struct filemap *fm, void **gpuMemPtr, void* (*func)(void *arg), void *args)
{
    pthread_t threadd;
    struct nvmed_delayed_task *delayed_task = (struct nvmed_delayed_task *)calloc(1, sizeof(struct nvmed_delayed_task));
//    delayed_task->devfd = devfd;
    delayed_task->fd = fd;
    delayed_task->fm = fm;
    delayed_task->threaded = 1;
    delayed_task->gpuMemPtr = gpuMemPtr;
    delayed_task->func = func;
    delayed_task->args = args;
//    pthread_create(&threadd, NULL, nvmed_send_ssd_host, delayed_task);
//    pthread_detach(threadd);

    nvmed_send_ssd_host(delayed_task);
    if(!delayed_task_queue)
    {
        fprintf(stderr, "creating delayed task queue\n");
        delayed_task_queue = fifo_new(16);
    }

//    fprintf(stderr, "pushing %p to queue\n",delayed_task);
    fifo_push(delayed_task_queue, delayed_task);
    atomic_increment(&number_delayed_tasks);

    return delayed_task;
}

//void *nvmed_send_ssd_host(struct nvmed_delayed_task *delayed_task)
void *nvmed_send_ssd_host(void *x)
{
    int i;
    struct nvmed_delayed_task *delayed_task = (struct nvmed_delayed_task *)x;
    int threaded = delayed_task->threaded;
    int fd = delayed_task->fd;
    pthread_t threads[NUM_OF_THREADS];
    struct fifo *ssd_host_queue;
    struct fifo *host_device_queue;
    struct nvmed_request *requests;
    struct nvmed_thread_parameter *thread_info;
    int *number_of_remaining_requests = &delayed_task->number_of_remaining_ssd_host_requests;
    struct filemap *fm = delayed_task->fm;
    size_t ret;
    int number_of_fifo_entries;
    struct sector current_sector;
    int sector_count = 0;
    int map_error;
    struct stat64 st;
    void *current;
    int blk_size;
    unsigned long num_blocks;
    #ifdef DEBUG
    struct timeval time_start, time_end;
    #endif
    if (fd < 0)
        return NULL;

    if (fstat64(fd, &st))
        return NULL;

    int devfd = nvme_dev_find(st.st_dev);
    if (devfd < 0) {
        if (devfd == -ENXIO)
        {
            fprintf(stderr, "NVMeD does not work on nvme device.\n");
//            map_error = FILEMAP_FALLBACK_DEV_NOT_NVME;
        }
        else
        {
            fprintf(stderr, "NVMeD does not have device permission.\n");
//            map_error = FILEMAP_FALLBACK_NOPERM_DEV;
        }
        return NULL;
    }
    delayed_task->devfd = devfd;

    num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    number_of_fifo_entries = (int)pow(2,ceil(log2(num_blocks*65536/NVMED_CHUNK)));
    blk_size = st.st_blksize / 512;

    if (fm == NULL)
        return NULL;

    fm->free = free_local_nvme;

    requests = (struct nvmed_request *)calloc(number_of_fifo_entries, sizeof(struct nvmed_request));

    *number_of_remaining_requests = 1;
    ssd_host_queue = fifo_new(number_of_fifo_entries);
    host_device_queue = fifo_new(number_of_fifo_entries);
    delayed_task->ssd_host_queue = ssd_host_queue;
    delayed_task->host_device_queue = host_device_queue;
    delayed_task->requests = requests;
    thread_info = (struct nvmed_thread_parameter *)calloc(1, sizeof(struct nvmed_thread_parameter));
    thread_info->number_of_remaining_requests = number_of_remaining_requests;
    thread_info->nvmed_incoming_requests_fifo = ssd_host_queue;
    thread_info->nvmed_completed_requests_fifo = host_device_queue;
    if(threaded)
    {
//        for(int i = 0; i < NUM_OF_THREADS; i++)
        for(int i = 0; i < NUM_OF_READ_THREADS; i++)
        {
            pthread_create(&threads[i], NULL, nvmed_send_ssd_host_thread, thread_info);
            pthread_detach(threads[i]);
        }
    }
    if(delayed_task->host_memory_address == NULL)
        delayed_task->host_memory_address = (void *)malloc(fm->length);
//        cudaMallocHost((void **)&delayed_task->host_memory_address,fm->length);
    #ifdef DEBUG
    fprintf(stderr, "Created delayed task with %p for %llu\n",delayed_task->host_memory_address,fm->length);
    #endif
    current = delayed_task->host_memory_address;

    unsigned long offset = 0, copied = 0, size_of_this_sector=0;
    int current_request=0;
    unsigned long remaining_blocks;
    // Generate SSD<->host transfer requests;
    for (int i = 0; i < num_blocks; i++) {
        unsigned long blknum = i;

        if (ioctl(fd, FIBMAP, &blknum) < 0)
            return NULL;

        //Seems we can't transfer more than 65536 LBAs at once so
        // in that case we split it into multiple transfers
        if (i != 0 && blknum * blk_size == current_sector.slba + current_sector.count &&
            current_sector.count + blk_size <= NVMED_CHUNK) {
            current_sector.count += blk_size;
            continue;
        }

        if (i != 0) {
            // Ready to enqueue the jobs. 
            atomic_increment(number_of_remaining_requests);
            int size_of_this_sector = current_sector.count*512;
            sector_count++;
            if(copied + size_of_this_sector > fm->length)
            {
                requests[current_request].size = fm->length - copied;
                submit_request(ssd_host_queue, &requests[current_request], devfd, 
                                current_sector.slba, 
                               (fm->length - copied)>> 9, fm->length - copied, current, NULL);
                current += requests[current_request].size;
                copied += requests[current_request].size;
                break;
            }
            submit_request(ssd_host_queue, &requests[current_request], devfd, 
                           current_sector.slba, 
                           current_sector.count, size_of_this_sector, current, NULL);
            copied += requests[current_request].size;
            current += requests[current_request].size;
            current_request++;
        }
        memset(&current_sector, 0, sizeof(struct sector));
        current_sector.slba = blknum * blk_size;
        current_sector.count = blk_size;
    }

    if(copied < fm->length)
    {
            int size_of_this_sector = current_sector.count*512;
            if(copied + size_of_this_sector > fm->length)
            {
                requests[current_request].size = fm->length - copied;
                submit_request(ssd_host_queue, &requests[current_request], devfd, 
                               current_sector.slba, 
                               (fm->length - copied)>>9, fm->length - copied, current, NULL);
            }
            else
            submit_request(ssd_host_queue, &requests[current_request], devfd, 
                           current_sector.slba, 
                           current_sector.count, size_of_this_sector, current, NULL);
//            fprintf(stderr,"nvmed_send_task(%p,%p,%llu)\n",current,current_gpu_memory_pointer,requests[current_request].size);
            current+=requests[current_request].size;
            copied+=requests[current_request].size;
            current_request++;        
    }
    else
        atomic_decrement(number_of_remaining_requests);
#ifdef DEBUG
    fprintf(stderr, "Processed %d (%p) requests from %p for %llu\n", *number_of_remaining_requests, number_of_remaining_requests, requests, copied);
#endif
    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
    fprintf(stderr, "Less than 0 sectors\n");
        goto free_and_fallback;
    }
    if(threaded == 0)
    {
//        pthread_create(&threads[0], NULL, nvmed_send_ssd_host_thread, thread_info);
//        pthread_detach(threads[0]);
        nvmed_send_ssd_host_thread(thread_info);
    }
//    ret = fm->length;
//    free(requests);
//    filemap_free(fm);
//    close(fd);
//    fprintf(stderr, "pushing %p to queue\n",delayed_task);


//    fifo_push(delayed_task_queue, delayed_task);
    atomic_increment(&number_delayed_tasks);
    close(fd);
    return fm;

exit_error_free:
    free(requests);
    free(fm);
    return NULL;

free_and_fallback:
    free(requests);
    free(fm);

fallback:
    errno = 0;
    return NULL;
}

void *nvmed_send_ssd_host_thread(void *t)
{
    struct nvmed_request *request;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo = thread_info->nvmed_completed_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
#ifdef DEBUG
    fprintf(stderr, "Sending from SSD to host %d (%p)\n", *number_of_remaining_requests,number_of_remaining_requests);
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((request = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
            #endif
//            fprintf(stderr,"nvmed_read_request(%p,%p,%llu,%d)\n",request->host_memory_address, request->gpu_memory_address, request->size, request->count);
            if (nvme_dev_read(request->devfd, request->slba, request->count, request->host_memory_address)) {
                    fprintf(stderr,"NVMe Read Error\n");
            }
            #ifdef DEBUG
            gettimeofday(&time_end, NULL);
            printf("HGProfile: pipeline read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            gettimeofday(&time_start, NULL);	
            #endif
            fifo_push(nvmed_memcpy_requests_fifo, request);
            atomic_decrement(number_of_remaining_requests);
        }
    }
    #ifdef DEBUG
    fprintf(stderr, "Sending from SSD to host %d (%p) Done\n", *number_of_remaining_requests,number_of_remaining_requests);
    #endif
    fifo_close(nvmed_requests_fifo);
    return NULL;
}

void *nvmed_wakeup_delayed_task(void *t)
{
    struct nvmed_delayed_task *delayed_task;
    struct nvmed_request *requests;
    struct nvmed_thread_parameter thread_info;
    int i;
    float *gpuMemPtr;
#define DEBUG 1
    #ifdef DEBUG
    struct timeval time_start, time_end;
    #endif
    if(t == NULL && number_delayed_tasks == 0)
        return NULL;
    else
    {
        if(t)
            delayed_task = (struct nvmed_delayed_task *)t;
        else if(!fifo_empty(delayed_task_queue))
        {
            delayed_task = (struct nvmed_delayed_task *)fifo_pop(delayed_task_queue);
//    #ifdef DEBUG
        fprintf(stderr, "Trying to wake up delayed task %p %d\n",delayed_task, number_delayed_tasks);
//    #endif
        }
        if(delayed_task == NULL)
            return NULL;
        requests = delayed_task->requests;
        #ifdef DEBUG
        fprintf(stderr, "Wake up delayed task %p\n",delayed_task);
        #endif
        if(cudaMalloc((void**) &gpuMemPtr, delayed_task->fm->length) != cudaSuccess)
        {
            #ifdef DEBUG
            fprintf(stderr, "Delay task %p again\n",delayed_task);
            #endif
//            if(t == NULL)
//                fifo_push(delayed_task_queue, delayed_task);
            return NULL;
        }
        delayed_task->fm->data = gpuMemPtr;
        // Why can't we just use cudaMemcpy :)
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
            #endif
#if 0   
        for(i =0; requests[i].host_memory_address != NULL; i++)
        {
            requests[i].gpu_memory_address = gpuMemPtr + ((requests[i].host_memory_address - delayed_task->host_memory_address)>>2);
            atomic_increment(&delayed_task->number_of_remaining_host_device_requests);
        }

        thread_info.number_of_remaining_requests = &delayed_task->number_of_remaining_host_device_requests;
        thread_info.nvmed_incoming_requests_fifo = delayed_task->host_device_queue;
        nvmed_send_host_device_thread(&thread_info);
#endif
        cudaMemcpy(gpuMemPtr, delayed_task->host_memory_address, delayed_task->fm->length, cudaMemcpyHostToDevice);
        free(delayed_task->host_memory_address);
            #ifdef DEBUG
            gettimeofday(&time_end, NULL);
            printf("HGProfile: wakeup memcpyHD %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            #endif
//        cudaFreeHost(delayed_task->host_memory_address);
        
//        pthread_mutex_lock(&nvmed_send_mutex);
//        *(delayed_task->gpuMemPtr) = gpuMemPtr;
//        pthread_cond_broadcast(&nvmed_send_cond);
//        pthread_mutex_unlock(&nvmed_send_mutex);
        #ifdef DEBUG
        fprintf(stderr, "Sending from host to device %d\n",delayed_task->number_of_remaining_host_device_requests);
        #endif
        free(requests);
    }
    if(delayed_task->func != NULL)
        delayed_task->func(delayed_task->args);
    // We have some more to wake up from the thread.
    if(t != NULL)
        return NULL;

    close(delayed_task->fd);
//    free(delayed_task->fm);
//    atomic_decrement(&number_delayed_tasks);
    #ifdef DEBUG
//    fprintf(stderr, "Remaining delayed tasks: %d\n",number_delayed_tasks);
    #endif
#undef DEBUG 
    return NULL;
}

void *nvmed_send_host_device_thread(void *t)
{
    struct nvmed_request *request;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
    #ifdef DEBUG
    fprintf(stderr, "Sending from host to device %d\n",*number_of_remaining_requests);
    #endif
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((request = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);
            #endif
            cudaMemcpy(request->gpu_memory_address, request->host_memory_address, request->size, cudaMemcpyHostToDevice);
            atomic_decrement(number_of_remaining_requests);
            #ifdef DEBUG
            gettimeofday(&time_end, NULL);
	    printf("HGProfile: pipeline cudaMemcpy %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
	    #endif
        }
    }
    fifo_close(nvmed_requests_fifo);
    return NULL;
}

// Wake up all the thread queued events. 
int nvmed_sync()
{
    size_t ret;
    if(!delayed_task_queue)
        return 0;
    ioctl(pinpoolfd, DONARD_IOCTL_NVMED_ACCQUIRE, NULL);
    while (!fifo_empty(delayed_task_queue))
    {
        nvmed_wakeup_delayed_task(NULL);
    }
}

int nvmed_barrier()
{
    int fd = open("/dev/donard_pinbuf", O_RDWR);
    ioctl(fd, DONARD_IOCTL_NVMED_ACCQUIRE, NULL);
    close(fd);
//    fprintf(stderr, "Barrier called\n");
    return 0;
}

int nvmed_wakeup_queue()
{
//        fprintf(stderr, "Wake up delayed tasks\n");
    //if(!pinpoolfd)
    //    pinpoolfd = open("/dev/donard_pinbuf", O_RDWR);
    int fd = open("/dev/donard_pinbuf", O_RDWR);
    ioctl(fd, DONARD_IOCTL_NVMED_WAKEUP, NULL);
    close(fd);
    return 0;    
}

int nvmed_process_lock()
{
//    if(!pinpoolfd)
        int fd = open("/dev/donard_pinbuf", O_RDWR);
        ioctl(fd, DONARD_IOCTL_NVMED_EXCLUDE, NULL);
        close(fd);
        return 0;
}
int nvmed_process_unlock()
{
//    if(!pinpoolfd)
        int fd = open("/dev/donard_pinbuf", O_RDWR);
        ioctl(fd, DONARD_IOCTL_NVMED_RELEASE, NULL);
        close(fd);
        return 0;
}

void *nvmed_host_pipeline_write_thread(void *t);
void *nvmed_host_pipeline_copy_write_thread(void *t);

unsigned long nvmed_host_pipeline_recv(const char *fname, void *gpuMemPtr, size_t size, size_t offset)
{
    int ret;
    struct stat64 st;
    // Hung-Wei: add a current pointer to the destination GPU memory address
    void *current;
    pthread_t threads[NUM_OF_THREADS];
    struct fifo *nvmed_requests_fifo;
    struct nvmed_request *requests;
    struct nvmed_thread_parameter thread_info;
    // Hung-Wei: maintain the original writing length
    volatile int number_of_remaining_requests;
    int current_request = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    struct filemap *fm = (struct filemap *)malloc(sizeof(struct filemap));
        fm->data = gpuMemPtr;
        fm->length = size;
        fm->type = FILEMAP_TYPE_CUDA;

    int fd = open(fname, O_WRONLY | O_TRUNC | O_CREAT, 0666);
    if (fd < 0) {
            fprintf(stderr, "Error opening file '%s': %s\n", fname,
                    strerror(errno));
            return -1;
    }

    if (fallocate(fd, FALLOC_FL_NO_HIDE_STALE, 0, fm->length)) {
            fprintf(stderr, "Could not fallocate the file, writing zeros instead: ");
            return -1;
    }

    if(offset % 512)
            fprintf(stderr,"NVMeD does not support offsets that does not align to 512B.\n");

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

    float *hostMemPtr = (float *)malloc(size);

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    number_of_remaining_requests = 1;
    int number_of_fifo_entries = (int)pow(2,ceil(log2(num_blocks*65536/NVMED_CHUNK)));
    nvmed_requests_fifo = fifo_new(number_of_fifo_entries);
    requests = (struct nvmed_request *)calloc(number_of_fifo_entries, sizeof(struct nvmed_request));
    thread_info.number_of_remaining_requests = &number_of_remaining_requests;
    thread_info.nvmed_incoming_requests_fifo = nvmed_requests_fifo;
    current = fm->data;
    for(int i = 0; i < NUM_OF_THREADS; i++)
    {
        pthread_create(&threads[i], NULL, nvmed_host_pipeline_write_thread, &thread_info);
    }

    int sector_count = generate_requests(fd, &st, 0, fm->length, nvmed_requests_fifo, requests, hostMemPtr, fm->data, &number_of_remaining_requests);
    atomic_decrement(&number_of_remaining_requests);

    if (sector_count < 0)
        return -1;

    for(int i = 0; i < NUM_OF_THREADS; i++)
        pthread_join(threads[i], NULL);
    
//    if(set_free)
    free(hostMemPtr);
    free(requests);
    //Updated modification and access times
    futimes(fd, NULL);
    fsync(fd);
//    fsync(fd);
    close(fd);

    return 0;
}

void *nvmed_host_pipeline_write_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *request;
//    struct fifo *nvmed_requests_fifo = t;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
//    pinbuf = pinpool_alloc();
    unsigned long offset = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((request = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
//            fprintf(stderr,"nvmed_recv_task_thread(%p,%llu,%x)\n",request->gpu_memory_address,request->size,request->slba);
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
            #endif
            cudaMemcpy(request->host_memory_address, request->gpu_memory_address, request->size , cudaMemcpyDeviceToHost);
            #ifdef DEBUG
    	    gettimeofday(&time_end, NULL);
    	    printf("HGProfile: threaded gpu %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            gettimeofday(&time_start, NULL);	
            #endif
            if (nvme_dev_write(request->devfd, request->slba, request->count, request->host_memory_address))
            {
                int map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
                pinpool_free(pinbuf);
                fprintf(stderr,"nvmed write error\n");
                return NULL;
            }
            #ifdef DEBUG
    	    gettimeofday(&time_end, NULL);
    	    printf("HGProfile: write %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
    	    #endif
            atomic_decrement(number_of_remaining_requests);
        }
    }
//    pinpool_free(pinbuf);
    fifo_close(nvmed_requests_fifo);
    return NULL;
}
void *nvmed_host_pipeline_pread_thread(void *t);
void *nvmed_host_pipeline_pread_copy_thread(void *t);

unsigned long nvmed_host_pipeline_pread(const char *fname, void **gpuMemPtr, size_t size, size_t offset, void *hostMemPtr)
{
    int map_error;
    struct stat64 st;
    void *current;
    void *current_gpu_memory_pointer;
//    void *hostMemPtr;
    int i;
    pthread_t threads[NUM_OF_THREADS];
    struct fifo *nvmed_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo;
    struct nvmed_request *requests;
    struct nvmed_thread_parameter thread_info;
    volatile int number_of_remaining_requests;
    size_t ret;
    int number_of_fifo_entries;
    struct sector current_sector;
    int sector_count = 0;
    int set_free = 0;
    #ifdef DEBUG
    struct timeval time_start, time_end;
    #endif
    int fd = open(fname, O_RDONLY);
    if (fd < 0)
        return -1;

    if (fstat64(fd, &st))
        return NULL;

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;
    number_of_fifo_entries = (int)pow(2,ceil(log2(num_blocks*65536/NVMED_CHUNK)));
    int blk_size = st.st_blksize / 512;
    if (size == 0)
        size = st.st_size;

    if(*gpuMemPtr == NULL)
    {
        if (cudaMalloc(gpuMemPtr, size) != cudaSuccess) {
            errno = ENOMEM;
            fprintf(stderr, "File exceeds available GPU memory size\n");
            return -1;
        }
    }
    if(hostMemPtr == NULL)
    {
        cudaMallocHost((void **)&hostMemPtr, size);
        set_free = 1;
    }
    requests = (struct nvmed_request *)calloc(number_of_fifo_entries, sizeof(struct nvmed_request));

    number_of_remaining_requests = 1;
    nvmed_requests_fifo = fifo_new(number_of_fifo_entries);
    nvmed_memcpy_requests_fifo = fifo_new(number_of_fifo_entries);
    thread_info.number_of_remaining_requests = &number_of_remaining_requests;
    thread_info.nvmed_incoming_requests_fifo = nvmed_requests_fifo;
    thread_info.nvmed_completed_requests_fifo = nvmed_memcpy_requests_fifo;
    for(int i = 0; i < NUM_OF_THREADS; i++)
    {
        if(i < 1)
            pthread_create(&threads[i], NULL, nvmed_host_pipeline_pread_copy_thread, &thread_info);
        else
            pthread_create(&threads[i], NULL, nvmed_host_pipeline_pread_thread, &thread_info);
    }
    i = 0;
    current = hostMemPtr;
    current_gpu_memory_pointer = *gpuMemPtr;
    for(ret = 0 ; ret < size ; ret += 33554432)
    {
        requests[i].devfd = fd;
        requests[i].slba = ret;
        requests[i].size = 33554432;
        if(size - ret < 33554432)
            requests[i].size = size - ret;
        requests[i].host_memory_address = current;
        requests[i].gpu_memory_address = current_gpu_memory_pointer;
        current_gpu_memory_pointer+=33554432;
        current+=33554432;
        fifo_push(&requests[i], nvmed_requests_fifo);
        i++;
        atomic_increment(&number_of_remaining_requests);
    }
    atomic_decrement(&number_of_remaining_requests);
//    fprintf(stderr,"host: %p, gpu: %p\n",current, current_gpu_memory_pointer);


    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        goto free_and_fallback;
    }

    for(int i = 0; i < NUM_OF_THREADS; i++)
        pthread_join(threads[i], NULL);

    free(requests);
    close(fd);
    if(set_free)
        cudaFreeHost(hostMemPtr);
    return ret;

exit_error_free:
    free(requests);
    return 0;

free_and_fallback:
    free(requests);

fallback:
    errno = 0;
    return 0;
}



void *nvmed_host_pipeline_pread_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *task;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo = thread_info->nvmed_completed_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((task = (struct nvmed_request *)fifo_pop(nvmed_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);	
            fprintf(stderr,"nvmed_read_task(%p, %p,%p,%llu,%d)\n",task->slba, task->host_memory_address, task->gpu_memory_address, task->size, task->count);
            #endif
            pread(task->devfd, task->host_memory_address, task->size, task->slba);
/*            if (nvme_dev_read(task->devfd, task->slba, task->count, task->host_memory_address)) {
//                map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
//                goto free_and_fallback;
                    fprintf(stderr,"NVMe Read Error\n");
            }*/
            #ifdef DEBUG
            gettimeofday(&time_end, NULL);
            printf("HGProfile: pipeline read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            gettimeofday(&time_start, NULL);	
            #endif
//#define singleStage 1
#ifdef singleStage
        cudaMemcpy(task->gpu_memory_address, task->host_memory_address, task->size , cudaMemcpyHostToDevice);
        atomic_decrement(number_of_remaining_requests);
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
//	printf("HGProfile: pipeline memcpy %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
#else
        fifo_push(nvmed_memcpy_requests_fifo, task);
#endif
        }
    }
    fifo_close(nvmed_requests_fifo);
    return NULL;
}

void *nvmed_host_pipeline_pread_copy_thread(void *t)
{
    struct pin_buf *pinbuf;
    struct nvmed_request *task;
    struct nvmed_thread_parameter *thread_info = (struct nvmed_thread_parameter *)t;
    struct fifo *nvmed_requests_fifo = thread_info->nvmed_incoming_requests_fifo;
    struct fifo *nvmed_memcpy_requests_fifo = thread_info->nvmed_completed_requests_fifo;
    volatile int *number_of_remaining_requests = thread_info->number_of_remaining_requests;
    unsigned long offset = 0;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
    while(*number_of_remaining_requests > 0)
    {
        if((task = (struct nvmed_request *)fifo_pop(nvmed_memcpy_requests_fifo))!=NULL)
        {
            #ifdef DEBUG
            gettimeofday(&time_start, NULL);
            #endif
            cudaMemcpy(task->gpu_memory_address, task->host_memory_address, task->size, cudaMemcpyHostToDevice);
            atomic_decrement(number_of_remaining_requests);
            #ifdef DEBUG
    	    gettimeofday(&time_end, NULL);
    	    //	printf("HGProfile: pipeline cudaMemcpy %d %p %d %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)), task->gpu_memory_address, task->size, task->slba);
	    printf("HGProfile: pipeline cudaMemcpy %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
	    #endif
        }
    }
    fifo_close(nvmed_requests_fifo);
    fifo_close(nvmed_memcpy_requests_fifo);
    return NULL;
}

struct filemap *filemap_local_nvmed(int fd, const char *fname,void *memPtr, size_t size, long file_offset, cudaStream_t stream)
{
    int map_error;
    struct stat64 st;
    void *current;
    int ret_val;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif
            #ifdef DEBUG
                gettimeofday(&time_start, NULL);
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

    if(memPtr == NULL)
    {
        fm->data = malloc(fm->length);
        if (fm->data == NULL) {
            return fm;
        }
    }
    else
        fm->data = memPtr;

    fm->type = FILEMAP_TYPE_CUDA;
    fm->free = free_cuda_nvme_threaded;
    copy_filename(fm, fname);
            #ifdef DEBUG
                gettimeofday(&time_start, NULL);
            #endif
#ifdef PERFSTAT
printf("Before get_sector_list\n");
perfstats_print();
#endif

    int sector_count = get_sector_list(fd, &st, &slist, file_offset, size);
            #ifdef DEBUG
                gettimeofday(&time_end, NULL);
                fprintf(stderr, "HGProfile: get %d sector list %ld\n",sector_count, ((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            #endif

    if (sector_count < 0) {
        map_error = FILEMAP_FALLBACK_NOPERM_FIBMAP;
        fprintf(stderr,"nvmed cannot get FIBMAP\n");
        goto free_and_fallback;
    }

    unsigned long num_blocks = (st.st_size + st.st_blksize - 1) / st.st_blksize;

    current = fm->data;

#ifdef DEBUG
    fprintf(stderr,"nvmed_send(%p,%llu,%llu)\n",current,fm->length,file_offset);
#endif

    unsigned long offset = 0, copied = 0, this_sector=0;
//perfstats_init();
        for (int i = 0; i < sector_count; i++) {
            if(slist[i].slba == 0)
                break;
            this_sector = slist[i].count * 512;
            #ifdef DEBUG
                gettimeofday(&time_start, NULL);	
            #endif
//            fprintf(stderr,"nvmed_send_task(%p,%llu,%llu)\n",slist[i].slba,slist[i].count,offset);
//perfstats_enable();
#ifdef PERFSTAT
printf("Before NVMe\n");
perfstats_print();
#endif
            if (ret_val = nvme_dev_read(devfd, slist[i].slba, slist[i].count,
                              current))
            {
                map_error = FILEMAP_FALLBACK_IOCTL_ERROR;
                fprintf(stderr, "Dev read ioctl error %x %d %d\n",slist[i].slba,slist[i].count,ret_val);
//                pinpool_free(fm->pinbuf);
                goto free_and_fallback;
            }
            #ifdef DEBUG
                gettimeofday(&time_end, NULL);
                fprintf(stderr, "HGProfile: NVMeD read %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
            #endif
//            offset += this_sector;
//perfstats_disable();
#ifdef PERFSTAT
printf("After NVMe\n");
perfstats_print();
#endif
            current += this_sector;
        }
//        cudaMemcpyAsync(current, fm->pinbuf->address, fm->length-copied, cudaMemcpyDeviceToDevice, stream);
//perfstats_deinit();
        cudaStreamSynchronize(stream);
//    pinpool_free(fm->pinbuf);

    return fm;

free_and_fallback:
    free(fm);

fallback:
    errno = 0;
    struct filemap *ret = filemap_alloc_cuda(fd, fname);
    ret->map_error = map_error;
    return ret;
}
int filemap_write_local_nvmed(struct filemap *fmap, int fd)
{
    int ret;
    struct stat64 st;
    // Hung-Wei: add a current pointer to the destination GPU memory address
    void *current;
    // Hung-Wei: maintain the original writing length
    unsigned long size=fmap->length;
#ifdef DEBUG
    struct timeval time_start, time_end;
#endif

    if (fmap->type != FILEMAP_TYPE_LOCAL) {
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


//    struct nvme_dev_sector slist[st.st_blocks / (st.st_blksize / 512)];
    struct sector *slist;
    int sector_count = get_sector_list(fd, &st, &slist, 0, size);

    if (sector_count < 0)
        return -1;

    // Hung-Wei: maintaining a current pointer to the GPU source data address
    current = fmap->data;

    unsigned long offset = 0, remaining=size, size_of_this_chunk=0;
    for (int i = 0; i < sector_count; i++) {
        if(i == sector_count-1)
            size_of_this_chunk = fmap->length - offset;
        else
            size_of_this_chunk = slist[i].count * 512;
//        current += size_of_this_chunk;
#ifdef DEBUG
        gettimeofday(&time_start, NULL);	
//	printf("HGProfile: writing %x %d %x %d\n",);	
#endif
        if (nvme_dev_write(devfd, slist[i].slba, slist[i].count,
                               current))
        {
            return -1;
        }
        current += size_of_this_chunk;
#ifdef DEBUG
    	gettimeofday(&time_end, NULL);
	printf("HGProfile: write %d\n",((time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec)));
#endif
    }

    //Updated modification and access times
    futimes(fd, NULL);
    fsync(fd);
//    fsync(fd);
//    fsync(fd);
    return 0;
}
#endif
