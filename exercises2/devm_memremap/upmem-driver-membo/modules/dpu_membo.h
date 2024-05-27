#ifndef DPU_MEMBO_H
#define DPU_MEMBO_H

#include <linux/list.h>
#include <dpu_region.h>
#include <dpu_rank.h>

#define PAGES_PER_DPU_RANK (DPU_RANK_SIZE / PAGE_SIZE)
#define SECTIONS_PER_DPU_RANK (PAGES_PER_DPU_RANK / PAGES_PER_SECTION)

#define DPU_MEMBO_NAME "dpu_membo"

typedef struct membo_context {
    int nid;
    struct mutex mutex;
    struct list_head rank_list;
    struct list_head ltb_rank_list;
    struct dpu_rank_t *ltb_index;
    atomic_t nr_free_ranks;
    atomic_t nr_used_ranks;
    atomic_t nr_ltb_ranks;
    atomic_t nr_reserved_ranks;
    atomic_t nr_total_ranks;
} membo_context_t;

struct dpu_membo_fs {
    struct cdev cdev;
    struct device dev;
    bool is_opened;
    struct mutex mutex;
};

struct dpu_membo_allocation_context {
    int nr_req_ranks;
    int nr_alloc_ranks;
};

struct dpu_membo_dynamic_reservation_context {
    int node0_threshold;
    int node1_threshold;
};

struct dpu_membo_usage_context {
    int nr_used_ranks;
};

void membo_lock(int nid);
void membo_unlock(int nid);

void membo_fs_lock(void);
void membo_fs_unlock(void);

uint32_t dpu_membo_rank_alloc(struct dpu_rank_t **rank, int nid);
uint32_t dpu_membo_rank_free(struct dpu_rank_t **rank, int nid);

int request_mram_borrowing(int nid);
int request_mram_reclamation(int nid);

int init_membo_context(int nid);
void destroy_membo_context(int nid);
int membo_init(void);

int dpu_membo_create_device(void);
void dpu_membo_release_device(void);
int dpu_membo_dev_uevent(struct device *dev, struct kobj_uevent_env *env);

#endif
