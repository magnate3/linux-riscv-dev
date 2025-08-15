#ifndef DPU_MEMBO_IOCTL_INCLUDE_H
#define DPU_MEMBO_IOCTL_INCLUDE_H

#include <dpu_membo.h>

#define DPU_MEMBO_IOCTL_MAGIC 'd'

#define DPU_MEMBO_IOCTL_ALLOC_RANKS_DIRECT _IOWR(DPU_MEMBO_IOCTL_MAGIC, 0, struct dpu_membo_allocation_context *)
#define DPU_MEMBO_IOCTL_ALLOC_RANKS_ASYNC _IOWR(DPU_MEMBO_IOCTL_MAGIC, 1, struct dpu_membo_allocation_context *)
#define DPU_MEMBO_IOCTL_SET_THRESHOLD _IOWR(DPU_MEMBO_IOCTL_MAGIC, 2, struct dpu_membo_dynamic_threshold_context *)
#define DPU_MEMBO_IOCTL_GET_USAGE _IOWR(DPU_MEMBO_IOCTL_MAGIC, 3, struct dpu_membo_usage_context *)

#endif
