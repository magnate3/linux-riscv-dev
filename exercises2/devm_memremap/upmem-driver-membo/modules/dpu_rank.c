/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2020 UPMEM. All rights reserved. */
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/slab.h>
#include <linux/cred.h>
#include <linux/uaccess.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/types.h>
#include <linux/version.h>
#include <linux/vmalloc.h>
#include <asm/cacheflush.h>
#include <linux/sched.h>
#include <linux/string.h>

#include <dpu_rank.h>
#include <dpu_rank_ioctl.h>
#include <dpu_region.h>
#include <dpu_control_interface.h>
#include <dpu_utils.h>
#include <dpu_rank_mcu.h>
#include <dpu_management.h>
#include <ufi/ufi.h>
#include <dpu_membo.h>

extern membo_context_t *membo_context_list[MAX_NUMNODES];

struct class *dpu_rank_class;

/*
 * Synchronize all allocations so that we can check if a DIMM is used.
 */
static DEFINE_MUTEX(rank_allocator_lock);

static void dpu_rank_allocator_lock(void)
{
	mutex_lock(&rank_allocator_lock);
}

static void dpu_rank_allocator_unlock(void)
{
	mutex_unlock(&rank_allocator_lock);
}

static struct page **get_page_array(struct dpu_rank_t *rank, int dpu_idx)
{
	uint32_t mram_size, nb_page_in_array;

	mram_size = rank->region->addr_translate.desc.memories.mram_size;
	nb_page_in_array = (mram_size / PAGE_SIZE);

	return &rank->xfer_dpu_page_array[dpu_idx * nb_page_in_array + dpu_idx];
}

/* Returns pages that must be put and free by calling function,
 * note that in case of success, the caller must release mmap_lock. */
static int pin_pages_for_xfer(struct device *dev, struct dpu_rank_t *rank,
			      struct dpu_transfer_mram *xfer,
			      unsigned int gup_flags, int dpu_idx)
{
	struct xfer_page *xferp;
	unsigned long nb_pages, nb_pages_expected;
	uint32_t off_page;
	int i;
	uint8_t *ptr_user =
		xfer->ptr[dpu_idx]; /* very important to keep this address,
					* since it will get overriden by
					* get_user_pages
					*/

	/* Allocation from userspace may not be aligned to
	 * page size, compute the offset of the base pointer
	 * to the previous page boundary.
	 */
	off_page = ((unsigned long)ptr_user & (PAGE_SIZE - 1));

	nb_pages_expected = ((xfer->size + off_page) / PAGE_SIZE);
	nb_pages_expected += (((xfer->size + off_page) % PAGE_SIZE) ? 1 : 0);

	xferp = &rank->xfer_pg[dpu_idx];
	memset(xferp, 0, sizeof(struct xfer_page));

	xferp->pages = get_page_array(rank, dpu_idx);
	xferp->off_first_page = off_page;
	xferp->nb_pages = nb_pages_expected;

	xfer->ptr[dpu_idx] = xferp;

	/* No page to pin or flush, bail early */
	if (nb_pages_expected == 0)
		return 0;

#if LINUX_VERSION_CODE > KERNEL_VERSION(3, 10, 0)
	/* Note: If needed, PageTransHuge returns true in case of a huge page */
	nb_pages = get_user_pages((unsigned long)ptr_user, xferp->nb_pages,
				  gup_flags, xferp->pages, NULL);
#else
	nb_pages = get_user_pages(current, current->mm, (unsigned long)ptr_user,
				  xferp->nb_pages, gup_flags, 0, xferp->pages,
				  NULL);
#endif
	if (nb_pages <= 0 || nb_pages != nb_pages_expected) {
		dev_err(dev, "cannot pin pages: nb_pages %ld/expected %ld\n",
			nb_pages, nb_pages_expected);
		return -EFAULT;
	}

	for (i = 0; i < nb_pages; ++i)
		flush_dcache_page(xferp->pages[i]);

	return nb_pages;
}

/*
 * This function fills xferp->pages whith all pages containing the buffer
 */
static int get_pages_for_xfer(struct device *dev, struct dpu_rank_t *rank,
			      struct dpu_transfer_mram *xfer, int dpu_idx)
{
	struct xfer_page *xferp;
	uint8_t *ptr_kernel = xfer->ptr[dpu_idx];
	uint32_t off_in_page;
	uint32_t len_copy_remaining = xfer->size;
	uint32_t len_copy_done = 0;
	uint32_t len_copy_in_page;
	uint32_t each_page;

	xferp = &rank->xfer_pg[dpu_idx];
	memset(xferp, 0, sizeof(struct xfer_page));

	/* Allocation may not be aligned to
	 * page size, compute the offset of the base pointer
	 * to the previous page boundary.
	 */
	xferp->off_first_page = ((unsigned long)ptr_kernel & (PAGE_SIZE - 1));
	xferp->nb_pages = ((xfer->size + xferp->off_first_page) / PAGE_SIZE);
	xferp->nb_pages +=
		(((xfer->size + xferp->off_first_page) % PAGE_SIZE) ? 1 : 0);
	if (xferp->nb_pages == 0) {
		return 0;
	}
	xferp->pages = get_page_array(rank, dpu_idx);
	xfer->ptr[dpu_idx] = xferp;

	for (each_page = 0; each_page < xferp->nb_pages; ++each_page) {
		off_in_page = !each_page ? xferp->off_first_page : 0;
		len_copy_in_page = min((uint32_t)(PAGE_SIZE - off_in_page),
				       len_copy_remaining);

		/* Beware if address is within the vmalloc range */
		if (is_vmalloc_addr(ptr_kernel + len_copy_done))
			xferp->pages[each_page] =
				vmalloc_to_page(ptr_kernel + len_copy_done);
		else
			xferp->pages[each_page] =
				virt_to_page(ptr_kernel + len_copy_done);

		len_copy_remaining -= len_copy_in_page;
		len_copy_done += len_copy_in_page;
	}

	// What for?
	for (each_page = 0; each_page < xferp->nb_pages; ++each_page)
		flush_dcache_page(xferp->pages[each_page]);

	return xferp->nb_pages;
}

/* Careful to release mmap_lock ! */
static int pin_pages_for_xfer_matrix(struct device *dev,
				     struct dpu_rank_t *rank,
				     struct dpu_transfer_mram *xfer_matrix,
				     unsigned int gup_flags)
{
	struct dpu_region_address_translation *tr;
	uint8_t ci_id, dpu_id, nb_cis, nb_dpus_per_ci;
	int idx;
	int ret;

	tr = &rank->region->addr_translate;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	nb_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 8, 0)
	down_read(&current->mm->mmap_sem);
#else
	down_read(&current->mm->mmap_lock);
#endif

	for_each_dpu_in_rank(idx, ci_id, dpu_id, nb_cis, nb_dpus_per_ci)
	{
		/* Here we work 'in-place' in xfer_matrix by replacing pointers
		 * to userspace buffers in struct dpu_transfer_mram * by newly
		 * allocated struct page ** representing the userspace buffer.
		 */
		if (!xfer_matrix->ptr[idx])
			continue;

		ret = pin_pages_for_xfer(dev, rank, xfer_matrix, gup_flags,
					 idx);
		if (ret < 0) {
			int i, j;

			for (i = idx - 1; i >= 0; --i) {
				if (xfer_matrix->ptr[i]) {
					struct xfer_page *xferp;

					xferp = xfer_matrix->ptr[i];

					for (j = 0; j < xferp->nb_pages; ++j)
						put_page(xferp->pages[j]);
				}
			}
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 8, 0)
			up_read(&current->mm->mmap_sem);
#else
			up_read(&current->mm->mmap_lock);
#endif
			return ret;
		}
	}

	return 0;
}

static int
get_kernel_pages_for_xfer_matrix(struct device *dev, struct dpu_rank_t *rank,
				 struct dpu_transfer_mram *xfer_matrix)
{
	struct dpu_region_address_translation *tr;
	uint8_t ci_id, dpu_id, nb_cis, nb_dpus_per_ci;
	int idx;
	int ret;

	tr = &rank->region->addr_translate;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	nb_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;

	for_each_dpu_in_rank(idx, ci_id, dpu_id, nb_cis, nb_dpus_per_ci)
	{
		/* Here we work 'in-place' in xfer_matrix by replacing pointers
		 * to kernel buffers in struct dpu_transfer_mram * by
		 * struct page ** representing the kernel buffer.
		 */
		if (!xfer_matrix->ptr[idx])
			continue;

		ret = get_pages_for_xfer(dev, rank, xfer_matrix, idx);
		if (ret < 0)
			return ret;
	}

	return 0;
}

static int dpu_rank_get_user_xfer_matrix(struct dpu_transfer_mram *xfer_matrix,
					 unsigned long ptr)
{
	/* Retrieve matrix transfer from userspace */
	if (copy_from_user(xfer_matrix, (void *)ptr, sizeof(*xfer_matrix)))
		return -EFAULT;

	return 0;
}

static int dpu_rank_write_to_rank(struct dpu_rank_t *rank, unsigned long ptr)
{
	struct device *dev = &rank->dev;
	struct dpu_region_address_translation *tr;
	struct dpu_transfer_mram xfer_matrix;
	int i, ret = 0;
	uint8_t ci_id, dpu_id, nb_cis, nb_dpus_per_ci;
	int idx;

	tr = &rank->region->addr_translate;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	nb_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;

	ret = dpu_rank_get_user_xfer_matrix(&xfer_matrix, ptr);
	if (ret)
		return ret;

	/* Pin pages of all the buffers in the transfer matrix, and start
	 * the transfer: from here we are committed to release mmap_lock.
	 */
	ret = pin_pages_for_xfer_matrix(dev, rank, &xfer_matrix, 0);
	if (ret)
		return ret;

	/* Launch the transfer */
	tr->write_to_rank(tr, rank->region->base, rank->channel_id,
			  &xfer_matrix);

	/* Free pages */
	for_each_dpu_in_rank(idx, ci_id, dpu_id, nb_cis, nb_dpus_per_ci)
	{
		if (xfer_matrix.ptr[idx]) {
			struct xfer_page *xferp;

			xferp = xfer_matrix.ptr[idx];

			for (i = 0; i < xferp->nb_pages; ++i)
				put_page(xferp->pages[i]);
		}
	}

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 8, 0)
	up_read(&current->mm->mmap_sem);
#else
	up_read(&current->mm->mmap_lock);
#endif

	return ret;
}

static int dpu_rank_read_from_rank(struct dpu_rank_t *rank, unsigned long ptr)
{
	struct device *dev = &rank->dev;
	struct dpu_region_address_translation *tr;
	struct dpu_transfer_mram xfer_matrix;
	int i, ret = 0;
	uint8_t ci_id, dpu_id, nb_cis, nb_dpus_per_ci;
	int idx;

	tr = &rank->region->addr_translate;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	nb_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;

	ret = dpu_rank_get_user_xfer_matrix(&xfer_matrix, ptr);
	if (ret)
		return ret;

		/* Pin pages of all the buffers in the transfer matrix, and start
	 * the transfer. Check if the buffer is writable and do not forget
	 * to fault in pages...
	 */
#if LINUX_VERSION_CODE > KERNEL_VERSION(3, 10, 0)
	ret = pin_pages_for_xfer_matrix(dev, rank, &xfer_matrix,
					FOLL_WRITE | FOLL_POPULATE);
#else
	ret = pin_pages_for_xfer_matrix(dev, rank, &xfer_matrix, FOLL_WRITE);
#endif
	if (ret)
		return ret;

	/* Launch the transfer */
	tr->read_from_rank(tr, rank->region->base, rank->channel_id,
			   &xfer_matrix);

	/* Free pages */
	for_each_dpu_in_rank(idx, ci_id, dpu_id, nb_cis, nb_dpus_per_ci)
	{
		if (xfer_matrix.ptr[idx]) {
			struct xfer_page *xferp;

			xferp = xfer_matrix.ptr[idx];

			for (i = 0; i < xferp->nb_pages; ++i)
				put_page(xferp->pages[i]);
		}
	}

#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 8, 0)
	up_read(&current->mm->mmap_sem);
#else
	up_read(&current->mm->mmap_lock);
#endif

	return ret;
}

static int dpu_rank_commit_commands(struct dpu_rank_t *rank, unsigned long ptr)
{
	struct dpu_region_address_translation *tr;
	uint32_t size_command;
	uint8_t nb_cis;

	tr = &rank->region->addr_translate;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	size_command = sizeof(uint64_t) * nb_cis;

	memset(rank->control_interface, 0, size_command);
	if (copy_from_user(rank->control_interface, (uint8_t *)ptr,
			   size_command))
		return -EFAULT;

	dpu_control_interface_commit_command(rank, rank->control_interface);

	return 0;
}

static int dpu_rank_update_commands(struct dpu_rank_t *rank, unsigned long ptr)
{
	struct dpu_region_address_translation *tr;
	uint32_t size_command;
	uint8_t nb_cis;

	tr = &rank->region->addr_translate;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	size_command = sizeof(uint64_t) * nb_cis;

	memset(rank->control_interface, 0, size_command);
	dpu_control_interface_update_command(rank, rank->control_interface);

	if (copy_to_user((uint8_t *)ptr, rank->control_interface, size_command))
		return -EFAULT;

	return 0;
}

uint32_t dpu_rank_get(struct dpu_rank_t *rank)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;

	dpu_region_lock(rank->region);

	if (rank->owner.is_owned && !rank->debug_mode) {
		dpu_region_unlock(rank->region);
		return DPU_ERR_DRIVER;
	}

	/* Do not init the rank when attached in debug mode */
	if (rank->owner.usage_count == 0) {
		uint8_t each_ci;

		dpu_rank_allocator_lock();
		if ((tr->init_rank) && (tr->init_rank(tr, rank->channel_id))) {
			dpu_rank_allocator_unlock();
			dpu_region_unlock(rank->region);
			pr_warn("Failed to allocate rank, error at initialization.\n");
			return DPU_ERR_DRIVER;
		}
		rank->owner.is_owned = 1;
		dpu_rank_allocator_unlock();

		/* Clear cached values */
		for (each_ci = 0; each_ci < DPU_MAX_NR_CIS; ++each_ci) {
			rank->runtime.control_interface.slice_info[each_ci]
				.structure_value = 0ULL;
			rank->runtime.control_interface.slice_info[each_ci]
				.slice_target.type = DPU_SLICE_TARGET_NONE;
			// TODO Quid of host_mux_mram_state ?
		}
	}
	rank->owner.usage_count++;

	dpu_region_unlock(rank->region);

	return DPU_OK;
}

void dpu_rank_put(struct dpu_rank_t *rank)
{
	struct dpu_region_address_translation *tr =
		&rank->region->addr_translate;
    pg_data_t *pgdat = NODE_DATA(rank->nid);

	dpu_region_lock(rank->region);

	rank->owner.usage_count--;
	if (rank->owner.usage_count == 0) {
		dpu_rank_allocator_lock();
		rank->owner.is_owned = 0;
		if (tr->destroy_rank)
			tr->destroy_rank(tr, rank->channel_id);
		dpu_rank_allocator_unlock();

		/*
         * Make sure we do not leave the region open whereas the rank
         * was freed.
         */
		rank->debug_mode = 0;
		rank->region->mode = DPU_REGION_MODE_UNDEFINED;
	}

    if (atomic_inc_return(&membo_context_list[rank->nid]->nr_free_ranks) == 1) {
        atomic_set(&pgdat->membo_disabled, 0);
    }

	dpu_region_unlock(rank->region);
}

static int dpu_rank_open(struct inode *inode, struct file *filp)
{
	struct dpu_rank_t *rank =
		container_of(inode->i_cdev, struct dpu_rank_t, cdev);

    membo_lock(rank->nid);

    if (!rank->is_reserved) {
        membo_unlock(rank->nid);
        return -EINVAL;
    }
	dev_dbg(&rank->dev, "opened rank_id %u\n", rank->id);

	filp->private_data = rank;

    if (dpu_rank_get(rank) != DPU_OK) {
        membo_unlock(rank->nid);
		return -EINVAL;
    }

    atomic_inc(&membo_context_list[rank->nid]->nr_used_ranks);

    membo_unlock(rank->nid);
	return 0;
}

static int dpu_rank_release(struct inode *inode, struct file *filp)
{
	struct dpu_rank_t *rank = filp->private_data;

	if (!rank)
		return 0;

    membo_lock(rank->nid);

	dev_dbg(&rank->dev, "closed rank_id %u\n", rank->id);

	dpu_rank_put(rank);

    rank->is_reserved = false;

    atomic_dec(&membo_context_list[rank->nid]->nr_used_ranks);

    membo_unlock(rank->nid);

	return 0;
}

static int dpu_rank_debug_mode(struct dpu_rank_t *rank, unsigned long mode)
{
	dpu_region_lock(rank->region);

	rank->debug_mode = mode;

	dpu_region_unlock(rank->region);

	return 0;
}

static long dpu_rank_ioctl(struct file *filp, unsigned int cmd,
			   unsigned long arg)
{
	struct dpu_rank_t *rank = filp->private_data;
	int ret = -EINVAL;

	if (!rank)
		return 0;

	dev_dbg(&rank->dev, "ioctl rank_id %u\n", rank->id);

	switch (cmd) {
	case DPU_RANK_IOCTL_WRITE_TO_RANK:
		ret = dpu_rank_write_to_rank(rank, arg);

		break;
	case DPU_RANK_IOCTL_READ_FROM_RANK:
		ret = dpu_rank_read_from_rank(rank, arg);

		break;
	case DPU_RANK_IOCTL_COMMIT_COMMANDS:
		ret = dpu_rank_commit_commands(rank, arg);

		break;
	case DPU_RANK_IOCTL_UPDATE_COMMANDS:
		ret = dpu_rank_update_commands(rank, arg);

		break;
	case DPU_RANK_IOCTL_DEBUG_MODE:
		ret = dpu_rank_debug_mode(rank, arg);

		break;
	default:
		break;
	}

	return ret;
}

/* This operation is backend specific, some will allow the mapping of
 * control interfaces and/or MRAMs.
 */
static int dpu_rank_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct dpu_rank_t *rank = filp->private_data;
	struct dpu_region_address_translation *tr;
	int ret = 0;

	tr = &rank->region->addr_translate;

	dpu_region_lock(rank->region);

	switch (rank->region->mode) {
	case DPU_REGION_MODE_UNDEFINED:
		if ((tr->capabilities & CAP_HYBRID) == 0) {
			ret = -EINVAL;
			goto end;
		}

		rank->region->mode = DPU_REGION_MODE_HYBRID;

		break;
	case DPU_REGION_MODE_SAFE:
	case DPU_REGION_MODE_PERF:
		/* TODO: Can we return a value that is not correct
                         * regarding man mmap ?
                         */
		dev_err(&rank->dev, "device already open"
				    " in perf or safe mode\n");
		ret = -EPERM;
		goto end;
	case DPU_REGION_MODE_HYBRID:
		break;
	}

end:
	dpu_region_unlock(rank->region);

	return ret ? ret : tr->mmap_hybrid(tr, filp, vma);
}

static struct file_operations dpu_rank_fops = { .owner = THIS_MODULE,
						.open = dpu_rank_open,
						.release = dpu_rank_release,
						.unlocked_ioctl =
							dpu_rank_ioctl,
						.mmap = dpu_rank_mmap };

static void dpu_rank_dev_release(struct device *dev)
{
	// TODO lacks attribute into dpu_rank_device to be update here,
	// mainly is_allocated ?
	// WARNING: here it is when the device is removed, not when userspace
	// releases fd.
}

static int dpu_init_ddr(struct dpu_region *region, struct dpu_rank_t *rank)
{
	struct dpu_region_address_translation *tr = &region->addr_translate;
	struct page *page;
	struct xfer_page *xferp;
	struct dpu_transfer_mram xfer_matrix;
	uint32_t nb_pages_per_mram, mram_size;
	int ret = 0, idx, i;
	uint8_t dpu_id, ci_id;
	uint8_t nb_cis, nb_dpus_per_ci;

	nb_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;
	nb_cis = tr->desc.topology.nr_of_control_interfaces;
	mram_size = tr->desc.memories.mram_size;
	nb_pages_per_mram = mram_size / PAGE_SIZE;

	xfer_matrix.size = mram_size;
	xfer_matrix.offset_in_mram = 0;

	/* GFP_ZERO is not necessary actually, but init with zero is cleaner */
	page = alloc_page(GFP_KERNEL | __GFP_ZERO);
	if (!page)
		return -ENOMEM;

	for_each_dpu_in_rank(idx, ci_id, dpu_id, nb_cis, nb_dpus_per_ci)
	{
		xferp = &rank->xfer_pg[idx];
		memset(xferp, 0, sizeof(struct xfer_page));

		xferp->pages = get_page_array(rank, idx);
		for (i = 0; i < nb_pages_per_mram; ++i)
			xferp->pages[i] = page;
		xferp->nb_pages = nb_pages_per_mram;

		xfer_matrix.ptr[idx] = xferp;
	}

	tr->write_to_rank(tr, region->base, rank->channel_id, &xfer_matrix);

	pr_info("ddr rank init done.\n");

	__free_page(page);

	return ret;
}

static int dpu_rank_create_device(struct device *dev_parent,
				  struct dpu_region *region,
				  struct dpu_rank_t *rank, bool must_init_mram)
{
	int ret;
	uint32_t mram_size, dpu_size_page_array;
	uint8_t nb_cis, nb_dpus_per_ci;

	ret = alloc_chrdev_region(&rank->dev.devt, 0, 1, DPU_RANK_NAME);
	if (ret)
		return ret;

	cdev_init(&rank->cdev, &dpu_rank_fops);
	rank->cdev.owner = THIS_MODULE;

	device_initialize(&rank->dev);

	nb_cis = region->addr_translate.desc.topology.nr_of_control_interfaces;
	nb_dpus_per_ci = region->addr_translate.desc.topology
				 .nr_of_dpus_per_control_interface;
	mram_size = region->addr_translate.desc.memories.mram_size;
	/* Userspace buffer is likely unaligned and need 1 more page */
	dpu_size_page_array =
		((mram_size / PAGE_SIZE) + 1) * sizeof(struct page *);

	rank->xfer_dpu_page_array =
		vmalloc(nb_cis * nb_dpus_per_ci * dpu_size_page_array);
	if (!rank->xfer_dpu_page_array) {
		goto free_device_ref_and_chrdev;
	}

	rank->owner.is_owned = 0;
	rank->owner.usage_count = 0;

	rank->region = region;

	rank->dev.class = dpu_rank_class;
	rank->dev.parent = dev_parent;
	dev_set_drvdata(&rank->dev, rank);
	rank->dev.release = dpu_rank_dev_release;
	dev_set_name(&rank->dev, DPU_RANK_PATH, rank->id);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
	ret = cdev_device_add(&rank->cdev, &rank->dev);
	if (ret)
		goto free_dpu_page_array;
#else
	ret = cdev_add(&rank->cdev, rank->dev.devt, 1);
	if (ret)
		goto free_dpu_page_array;

	ret = device_add(&rank->dev);
	if (ret)
		goto free_cdev;
#endif

	if (must_init_mram) {
		ret = dpu_init_ddr(region, rank);
		if (ret)
			goto free_cdev_dev;
	}

	return 0;

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
free_cdev_dev:
	cdev_device_del(&rank->cdev, &rank->dev);
#else
free_cdev_dev:
	device_del(&rank->dev);
free_cdev:
	cdev_del(&rank->cdev);
#endif
free_dpu_page_array:
	vfree(rank->xfer_dpu_page_array);
free_device_ref_and_chrdev:
	put_device(&rank->dev);
	unregister_chrdev_region(rank->dev.devt, 1);
	return ret;
}

int dpu_rank_copy_to_rank(struct dpu_rank_t *rank,
			  struct dpu_transfer_mram *xfer_matrix)
{
	int ret;
	struct device *dev = &rank->dev;
	struct dpu_region_address_translation *tr;

	tr = &rank->region->addr_translate;

	ret = get_kernel_pages_for_xfer_matrix(dev, rank, xfer_matrix);
	if (ret)
		return ret;

	tr->write_to_rank(tr, rank->region->base, rank->channel_id,
			  xfer_matrix);

	return 0;
}

int dpu_rank_copy_from_rank(struct dpu_rank_t *rank,
			    struct dpu_transfer_mram *xfer_matrix)
{
	int ret;
	struct device *dev = &rank->dev;
	struct dpu_region_address_translation *tr;

	tr = &rank->region->addr_translate;

	ret = get_kernel_pages_for_xfer_matrix(dev, rank, xfer_matrix);
	if (ret)
		return ret;

	tr->read_from_rank(tr, rank->region->base, rank->channel_id,
			   xfer_matrix);

	return 0;
}

int dpu_rank_init_device(struct device *dev, struct dpu_region *region,
			 bool must_init_mram)
{
	struct dpu_region_address_translation *tr;
	struct dpu_rank_t *rank;
	uint8_t nr_cis, each_ci;
	uint8_t nr_dpus_per_ci, each_dpu;
	int ret;

	tr = &region->addr_translate;

	rank = &region->rank;
	rank->channel_id = 0xff;
	rank->rank_index = DPU_RANK_INVALID_INDEX;

	/* Assume all DPUs are enabled */
	nr_cis = tr->desc.topology.nr_of_control_interfaces;
	nr_dpus_per_ci = tr->desc.topology.nr_of_dpus_per_control_interface;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		struct dpu_configuration_slice_info_t *slice_info =
			&rank->runtime.control_interface.slice_info[each_ci];

		slice_info->enabled_dpus = (1 << nr_dpus_per_ci) - 1;
		slice_info->all_dpus_are_enabled = true;
	}

	rank->dpus = kzalloc(nr_dpus_per_ci * nr_cis * sizeof(*(rank->dpus)),
			     GFP_KERNEL);
	if (!rank->dpus)
		return -ENOMEM;

	for (each_ci = 0; each_ci < nr_cis; ++each_ci) {
		for (each_dpu = 0; each_dpu < nr_dpus_per_ci; ++each_dpu) {
			uint8_t dpu_index =
				(nr_dpus_per_ci * each_ci) + each_dpu;
			struct dpu_t *dpu = rank->dpus + dpu_index;
			dpu->rank = rank;
			dpu->slice_id = each_ci;
			dpu->dpu_id = each_dpu;
			dpu->enabled = true;
		}
	}

	ret = dpu_rank_create_device(dev, region, rank, must_init_mram);
	if (ret)
		goto free_dpus;

    atomic_set(&rank->nr_ltb_sections, 0);
	list_add_tail(&rank->list, &(membo_context_list[rank->nid]->rank_list));
    rank->is_reserved = false;
    atomic_inc(&membo_context_list[rank->nid]->nr_free_ranks);
    atomic_inc(&membo_context_list[rank->nid]->nr_total_ranks);

	return 0;

free_dpus:
	kfree(rank->dpus);
	return ret;
}

void dpu_rank_release_device(struct dpu_region *region)
{
	struct dpu_rank_t *rank = &region->rank;

	pr_info("dpu_rank: releasing rank\n");

	list_del(&rank->list);

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4, 10, 0)
	cdev_device_del(&rank->cdev, &rank->dev);
#else
	device_del(&rank->dev);
	cdev_del(&rank->cdev);
#endif
	vfree(rank->xfer_dpu_page_array);
	put_device(&rank->dev);
	unregister_chrdev_region(rank->dev.devt, 1);
	kfree(rank->dpus);
}

bool dpu_is_dimm_used(struct dpu_rank_t *rank)
{
	struct dpu_rank_t *rank_iterator;
	const char *sn = rank->serial_number;
	int nr_ranks_used = 0;
    int node;

	/* We cannot relate the rank to a DIMM */
	if (!strcmp(sn, "")) {
		return true;
	}

    for_each_online_node(node)
        list_for_each_entry (rank_iterator, &(membo_context_list[node]->rank_list), list) {
            if ((!strcmp(rank_iterator->serial_number, sn)) &&
                (rank_iterator->owner.is_owned)) {
                nr_ranks_used++;
            }
        }

	return (nr_ranks_used == 0) ? false : true;
}
