#ifndef _CONF_H_
#define _CONF_H_

#define MODE_SYNC 1
#define MODE_ASYNC 2
#define MODE_ONE 3
#define MODE MODE_SYNC

#define COPY_LESS 1
#define SIMPLE_POLL 1
#define SIMPLE_MAKE_WR 1

#define CUSTOM_MAKE_REQ_FN 0
#define KERNEL_SECTOR_SIZE   512
#define SECTORS_PER_PAGE  (PAGE_SIZE / KERNEL_SECTOR_SIZE)
#define DEVICE_BOUND 100
#define REQ_ARR_SIZE 10
#define MAX_REQ 1024
#define MERGE_REQ false
#define REQ_POOL_SIZE 1024

#define RDMA_BUFFER_SIZE (1024*1024)
#define CQE_SIZE 4096

#define DEBUG_OUT_REQ 0
#define MEASURE_LATENCY 0
#define LATENCY_BUCKET 100


//redefine
#if MEASURE_LATENCY
#define MAX_REQ 1
#endif

//check
#if MEASURE_LATENCY && MAX_REQ != 1
#error
#endif


#endif



