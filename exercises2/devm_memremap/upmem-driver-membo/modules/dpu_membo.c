#include <linux/nodemask.h>
#include <linux/memory_hotplug.h>
#include <linux/memory.h>

#include <dpu_membo.h>
#include <dpu_membo_ioctl.h>
#include <dpu_rank.h>

bool membo_initialized = false;
membo_context_t *membo_context_list[MAX_NUMNODES];
struct dpu_membo_fs membo_fs;

int dpu_membo_dev_uevent(struct device *dev, struct kobj_uevent_env *env)
{
    add_uevent_var(env, "DEVMODE=%#o", 0666);
    return 0;
}

struct class *dpu_membo_class;

static int dpu_membo_open(struct inode *inode, struct file *filp)
{
    struct dpu_membo_fs *fs =
        container_of(inode->i_cdev, struct dpu_membo_fs, cdev);

    filp->private_data = fs;

    membo_fs_lock();
    if (fs->is_opened) {
        membo_fs_unlock();
        return 0;
    }

    fs->is_opened = true;
    membo_fs_unlock();

    return 0;
}

static int dpu_membo_release(struct inode *inode, struct file *filp)
{
    struct dpu_membo_fs *fs = filp->private_data;

    if (!fs)
        return 0;

    membo_fs_lock();
    fs->is_opened = false;
    membo_fs_unlock();

    return 0;
}

static int reclaim_one_rank(struct dpu_rank_t *rank)
{
    struct page *page = virt_to_page(rank->region->base);
    struct memory_block *mem = container_of(rank->dev.parent, struct memory_block, dev);
    unsigned long pfn;

    for (pfn = page_to_pfn(page); pfn < page_to_pfn(page) + PAGES_PER_SECTION * atomic_read(&rank->nr_ltb_sections); pfn += PAGES_PER_SECTION) {
        reclaim_mram_pages(pfn, PAGES_PER_SECTION, mem->group, &rank->region->dpu_dax_dev.pgmap);
    }

    atomic_set(&rank->nr_ltb_sections, 0);
    dpu_membo_rank_free(&rank, rank->nid);

    return 0;
}

static int direct_reclaim_ranks(int nr_ranks)
{
    struct dpu_rank_t *rank_iterator, *tmp;
    int node;
    int nr_req_target = nr_ranks;

    /* reclaim nr_req_ranks ranks */
    for_each_online_node(node)
        list_for_each_entry_safe (rank_iterator, tmp, &membo_context_list[node]->ltb_rank_list, list) {
            reclaim_one_rank(rank_iterator);

            if (--nr_req_target == 0)
                goto end;
        }

end:
    return 0;
}

static int direct_reclaim_ranks_node(int nr_ranks, int node)
{
    struct dpu_rank_t *rank_iterator, *tmp;
    int nr_req_target = nr_ranks;

    /* reclaim nr_req_ranks ranks */
    list_for_each_entry_safe (rank_iterator, tmp, &membo_context_list[node]->ltb_rank_list, list) {
        reclaim_one_rank(rank_iterator);

        if (--nr_req_target == 0)
            goto end;
    }

end:
    return 0;
}

static int reserve_ranks_for_allocation(int nr_ranks)
{
    struct dpu_rank_t *rank_iterator, *tmp;
    int node;
    int nr_req_target = nr_ranks;

    for_each_online_node(node)
        list_for_each_entry_safe (rank_iterator, tmp, &membo_context_list[node]->rank_list, list) {
            if (!rank_iterator->is_reserved) {
                /* the rank is reserved for allocation */
                rank_iterator->is_reserved = true;
                atomic_dec(&membo_context_list[rank_iterator->nid]->nr_free_ranks);
                if (--nr_req_target == 0)
                    goto end;
            }
        }
end:
    return 0;
}

static int dpu_membo_alloc_ranks_direct(unsigned long ptr)
{
    struct dpu_membo_allocation_context allocation_context;
    int node;
    int nr_free_ranks = 0;
    int nr_ltb_ranks = 0;
    int nr_reclamation_ranks = 0;
    int nr_reserved_ranks = 0;

    if (copy_from_user(&allocation_context, (void *)ptr, sizeof(allocation_context)))
        return -EFAULT;

    /* we must lock membo on each node */
    for_each_online_node(node)
        membo_lock(node);

    for_each_online_node(node)
        nr_free_ranks += atomic_read(&membo_context_list[node]->nr_free_ranks);

    for_each_online_node(node) {
        nr_ltb_ranks += atomic_read(&membo_context_list[node]->nr_ltb_ranks);
        nr_reserved_ranks += atomic_read(&membo_context_list[node]->nr_reserved_ranks);
    }

    pr_info("membo: threshold is: %d ranks\n", nr_reserved_ranks);
    if (allocation_context.nr_req_ranks <= nr_free_ranks) {
        pr_info("membo: allocation without direct reclamation\n");
        goto reserve_ranks;
    }

    if (allocation_context.nr_req_ranks <= nr_free_ranks + nr_ltb_ranks) {
        /* we can get enough ranks after relcaiming (nr_req_ranks - nr_free_ranks) ranks */
        nr_reclamation_ranks = allocation_context.nr_req_ranks - nr_free_ranks;
        pr_info("membo: trigger direct reclamation: %d ranks\n", nr_reclamation_ranks);
        direct_reclaim_ranks(nr_reclamation_ranks);
    } else {
        for_each_online_node(node)
            membo_unlock(node);
        return -EBUSY;
    }

reserve_ranks:
    reserve_ranks_for_allocation(allocation_context.nr_req_ranks);

    for_each_online_node(node)
        membo_unlock(node);
    return 0;
}

static int dpu_membo_alloc_ranks_async(unsigned long ptr)
{
    struct dpu_membo_allocation_context allocation_context;
    int node;
    int nr_free_ranks = 0;
    int nr_ltb_ranks = 0;

    if (copy_from_user(&allocation_context, (void *)ptr, sizeof(allocation_context)))
        return -EFAULT;

    /* we must lock membo on each node */
    for_each_online_node(node)
        membo_lock(node);

    for_each_online_node(node)
        nr_free_ranks += atomic_read(&membo_context_list[node]->nr_free_ranks);

    /* We can get enough ranks without doing the direct reclamation */
    if (allocation_context.nr_req_ranks <= nr_free_ranks) {
        allocation_context.nr_alloc_ranks = allocation_context.nr_req_ranks;
        goto reserve_ranks;
    }

    for_each_online_node(node)
        nr_ltb_ranks += atomic_read(&membo_context_list[node]->nr_ltb_ranks);

    if (nr_free_ranks + nr_ltb_ranks >= allocation_context.nr_req_ranks) {
        if (nr_free_ranks == 0) {
            direct_reclaim_ranks(1);
            allocation_context.nr_alloc_ranks = 1;
        } else {
            /* Give the free ranks to the user immediately */
            allocation_context.nr_alloc_ranks = nr_free_ranks;
        }
        goto reserve_ranks;
    } else {
        for_each_online_node(node)
            membo_unlock(node);
        return -EBUSY;
    }

reserve_ranks:
    if (copy_to_user((void *)ptr, &allocation_context, sizeof(allocation_context))) {
        for_each_online_node(node)
            membo_unlock(node);
        return -EFAULT;
    }

    reserve_ranks_for_allocation(allocation_context.nr_alloc_ranks);

    for_each_online_node(node)
        membo_unlock(node);

    return 0;
}

static int dpu_membo_get_usage(unsigned long ptr)
{
    struct dpu_membo_usage_context usage_context;
    int node;
    int nr_used_ranks = 0;

    for_each_online_node(node)
        membo_lock(node);

    for_each_online_node(node)
        nr_used_ranks += atomic_read(&membo_context_list[node]->nr_used_ranks);

    usage_context.nr_used_ranks = nr_used_ranks;

    if (copy_to_user((void *)ptr, &usage_context, sizeof(usage_context))) {
        for_each_online_node(node)
            membo_unlock(node);
        return -EFAULT;
    }

    for_each_online_node(node)
        membo_unlock(node);

    return 0;
}

static int dpu_membo_set_threshold(unsigned long ptr)
{
    struct dpu_membo_dynamic_reservation_context reservation_context;
    int node;

    if (copy_from_user(&reservation_context, (void *)ptr, sizeof(reservation_context)))
        return -EFAULT;

    for_each_online_node(node)
        membo_lock(node);

    atomic_set(&membo_context_list[0]->nr_reserved_ranks, reservation_context.node0_threshold);
    atomic_set(&membo_context_list[1]->nr_reserved_ranks, reservation_context.node1_threshold);

    for_each_online_node(node) {
        pg_data_t *pgdat = NODE_DATA(node);
        int nr_total_ranks = atomic_read(&membo_context_list[node]->nr_total_ranks);
        int nr_reserved_ranks = atomic_read(&membo_context_list[node]->nr_reserved_ranks);
        int nr_ltb_ranks = atomic_read(&membo_context_list[node]->nr_ltb_ranks);

        if (nr_ltb_ranks > nr_total_ranks - nr_reserved_ranks) {
            direct_reclaim_ranks_node(nr_ltb_ranks - (nr_total_ranks - nr_reserved_ranks), node);
        }

        atomic_set(&pgdat->membo_disabled, 0);
    }

    for_each_online_node(node)
        membo_unlock(node);

    return 0;
}

static long dpu_membo_ioctl(struct file *filp, unsigned int cmd,
        unsigned long arg)
{
    struct dpu_membo_fs *fs = filp->private_data;
    int ret = 0;

    if (!fs)
        return 0;

    switch (cmd) {
    case DPU_MEMBO_IOCTL_ALLOC_RANKS_DIRECT:
        ret = dpu_membo_alloc_ranks_direct(arg);
        break;
    case DPU_MEMBO_IOCTL_ALLOC_RANKS_ASYNC:
        ret = dpu_membo_alloc_ranks_async(arg);
        break;
    case DPU_MEMBO_IOCTL_SET_THRESHOLD:
        ret = dpu_membo_set_threshold(arg);
        break;
    case DPU_MEMBO_IOCTL_GET_USAGE:
        ret = dpu_membo_get_usage(arg);
        break;
    default:
        break;
    }

    return ret;
}

static struct file_operations dpu_membo_fops = {
    .owner = THIS_MODULE,
    .open = dpu_membo_open,
    .release = dpu_membo_release,
    .unlocked_ioctl = dpu_membo_ioctl,
};

extern int (*membo_request_mram_borrowing)(int nid);
extern int (*membo_request_mram_reclamation)(int nid);

void membo_lock(int nid)
{
    mutex_lock(&(membo_context_list[nid]->mutex));
}

void membo_unlock(int nid)
{
    mutex_unlock(&(membo_context_list[nid]->mutex));
}

void membo_fs_lock(void)
{
    mutex_lock(&membo_fs.mutex);
}

void membo_fs_unlock(void)
{
    mutex_unlock(&membo_fs.mutex);
}

static void dpu_membo_dev_release(struct device *dev)
{

}

int dpu_membo_create_device(void)
{
    int ret;

    ret = alloc_chrdev_region(&membo_fs.dev.devt, 0, 1, DPU_MEMBO_NAME);
    if (ret)
        return ret;

    cdev_init(&membo_fs.cdev, &dpu_membo_fops);
    membo_fs.cdev.owner = THIS_MODULE;

    device_initialize(&membo_fs.dev);

    membo_fs.dev.class = dpu_membo_class;
    membo_fs.dev.release = dpu_membo_dev_release;

    dev_set_drvdata(&membo_fs.dev, &membo_fs);
    dev_set_name(&membo_fs.dev, DPU_MEMBO_NAME);

    ret = cdev_device_add(&membo_fs.cdev, &membo_fs.dev);
	if (ret)
		goto out;

    mutex_init(&membo_fs.mutex);
    membo_fs.is_opened = false;

    return 0;
out:
    put_device(&membo_fs.dev);
    unregister_chrdev_region(membo_fs.dev.devt, 1);
    return ret;
}

void dpu_membo_release_device(void)
{
    cdev_device_del(&membo_fs.cdev, &membo_fs.dev);
    put_device(&membo_fs.dev);
    unregister_chrdev_region(membo_fs.dev.devt, 1);
}

static void init_membo_api(void)
{
    membo_request_mram_borrowing = request_mram_borrowing;
    membo_request_mram_reclamation = request_mram_reclamation;
}

int init_membo_context(int nid)
{
    membo_context_list[nid] = kzalloc(sizeof(membo_context_t), GFP_KERNEL);

    if (!membo_context_list[nid])
        return -ENOMEM;
    mutex_init(&membo_context_list[nid]->mutex);

    membo_lock(nid);
    INIT_LIST_HEAD(&membo_context_list[nid]->rank_list);
    INIT_LIST_HEAD(&membo_context_list[nid]->ltb_rank_list);

    membo_context_list[nid]->ltb_index = NULL;
    membo_context_list[nid]->nid = nid;

    atomic_set(&membo_context_list[nid]->nr_free_ranks, 0);
    atomic_set(&membo_context_list[nid]->nr_ltb_ranks, 0);
    atomic_set(&membo_context_list[nid]->nr_used_ranks, 0);
    atomic_set(&membo_context_list[nid]->nr_reserved_ranks, 0);
    atomic_set(&NODE_DATA(nid)->membo_is_direct_reclaim_activated, 0);

    membo_unlock(nid);

    return 0;
}

void destroy_membo_context(int nid)
{
    if (membo_context_list[nid])
        kfree(membo_context_list[nid]);
}

int membo_init(void)
{
    int node;

    for_each_online_node(node)
        membo_init_node(node);

    for_each_online_node(node)
        membo_manager_run(node);

    for_each_online_node(node)
        membo_reclaimer_run(node);

    init_membo_api();

    return 0;
}

static uint32_t expand_one_section(struct dpu_rank_t *rank, int section_id)
{
    struct page *page = virt_to_page(rank->region->base);
    struct memory_block *mem = container_of(rank->dev.parent, struct memory_block, dev);
    struct zone *zone = page_zone(page);

    borrow_mram_pages(page_to_pfn(page) + section_id * PAGES_PER_SECTION, PAGES_PER_SECTION, zone, mem->group);
    return 0;
}

static uint32_t reclaim_one_section(struct dpu_rank_t *rank, int section_id)
{
    struct page *page = virt_to_page(rank->region->base);
    struct memory_block *mem = container_of(rank->dev.parent, struct memory_block, dev);

    reclaim_mram_pages(page_to_pfn(page) + section_id * PAGES_PER_SECTION, PAGES_PER_SECTION, mem->group, &rank->region->dpu_dax_dev.pgmap);
    return 0;
}

uint32_t dpu_membo_rank_alloc(struct dpu_rank_t **rank, int nid)
{
    struct dpu_rank_t *rank_iterator;
    pg_data_t *pgdat = NODE_DATA(nid);

    *rank = NULL;

    list_for_each_entry (rank_iterator, &membo_context_list[nid]->rank_list, list) {
        if (rank_iterator->is_reserved)
            continue;
        if (dpu_rank_get(rank_iterator) == DPU_OK) {
            *rank = rank_iterator;

            /* Move the rank from rank_list to ltb_rank_list */
            list_del(&rank_iterator->list);
            list_add_tail(&rank_iterator->list, &membo_context_list[nid]->ltb_rank_list);

            /* Update counters */
            atomic_dec(&membo_context_list[nid]->nr_free_ranks);
            if (atomic_inc_return(&membo_context_list[nid]->nr_ltb_ranks) == 1)
                wakeup_membo_reclaimer(nid);
            atomic_inc(&pgdat->membo_nr_ranks);

            /* Update ltb allocation index */
            membo_context_list[nid]->ltb_index = rank_iterator;
            return DPU_OK;
        }
    }

    /* We can not find a free rank for the MEMBO allocation */
    return DPU_ERR_DRIVER;
}

uint32_t dpu_membo_rank_free(struct dpu_rank_t **rank, int nid)
{
    struct dpu_rank_t *target_rank;
    pg_data_t *pgdat = NODE_DATA(nid);

    target_rank = *rank;

    dpu_rank_put(target_rank);

    if (atomic_dec_return(&membo_context_list[nid]->nr_ltb_ranks) == 0)
        membo_context_list[nid]->ltb_index = NULL;
    else if (membo_context_list[nid]->ltb_index == target_rank)
        membo_context_list[nid]->ltb_index = list_entry(target_rank->list.prev, typeof(*target_rank), list);

    list_del(&target_rank->list);
    list_add_tail(&target_rank->list, &membo_context_list[nid]->rank_list);

    atomic_dec(&pgdat->membo_nr_ranks);

    return DPU_OK;
}

int request_mram_borrowing(int nid)
{
    struct dpu_rank_t *current_ltb_rank;

    membo_lock(nid);
    current_ltb_rank = membo_context_list[nid]->ltb_index;

    if (current_ltb_rank)
        if (atomic_read(&current_ltb_rank->nr_ltb_sections) != SECTIONS_PER_DPU_RANK)
            goto request_one_section;

    /* try to allocate a new rank for MEMBO */
    if (atomic_read(&membo_context_list[nid]->nr_ltb_ranks) >= atomic_read(&membo_context_list[nid]->nr_total_ranks) - atomic_read(&membo_context_list[nid]->nr_reserved_ranks)) {
        pr_info("Fail to borrow a rank\n");
        membo_unlock(nid);
        return -EBUSY;
    }

    if (dpu_membo_rank_alloc(&current_ltb_rank, nid) == DPU_OK)
        goto request_one_section;

    membo_unlock(nid);
    return -EBUSY;

request_one_section:
    expand_one_section(current_ltb_rank, atomic_read(&current_ltb_rank->nr_ltb_sections));
    atomic_inc(&current_ltb_rank->nr_ltb_sections);
    membo_unlock(nid);
    return 0;
}

int request_mram_reclamation(int nid)
{
    struct dpu_rank_t *current_ltb_rank;

    membo_lock(nid);
    current_ltb_rank = membo_context_list[nid]->ltb_index;

    if (!atomic_read(&membo_context_list[nid]->nr_ltb_ranks)) {
        membo_unlock(nid);
        return -EBUSY;
    }

    atomic_dec(&current_ltb_rank->nr_ltb_sections);
    reclaim_one_section(current_ltb_rank, atomic_read(&current_ltb_rank->nr_ltb_sections));

    if (!atomic_read(&current_ltb_rank->nr_ltb_sections))
        dpu_membo_rank_free(&current_ltb_rank, nid);

    membo_unlock(nid);
    return 0;
}

