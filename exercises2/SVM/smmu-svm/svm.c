// SPDX-License-Identifier: GPL-2.0
/*
 * Copyright (c) 2017-2018 Hisilicon Limited.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 */

#include <asm/esr.h>
#include <linux/mmu_context.h>

#include <linux/delay.h>
#include <linux/err.h>
#include <linux/interrupt.h>
#include <linux/io.h>
#include <linux/iommu.h>
#include <linux/miscdevice.h>
#include <linux/mman.h>
#include <linux/mmu_notifier.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_device.h>
#include <linux/platform_device.h>
#include <linux/ptrace.h>
#include <linux/security.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/sched.h>
#include <linux/hugetlb.h>
#include <linux/sched/mm.h>
#include <linux/msi.h>
#include <linux/acpi.h>

#define SVM_DEVICE_NAME "svm"
#define ASID_SHIFT		48

#define SVM_IOCTL_REMAP_PROC		0xfff4
#define SVM_IOCTL_UNPIN_MEMORY		0xfff5
#define SVM_IOCTL_PIN_MEMORY		0xfff7
#define SVM_IOCTL_GET_PHYS			0xfff9
#define SVM_IOCTL_LOAD_FLAG			0xfffa
#define SVM_IOCTL_SET_RC			0xfffc
#define SVM_IOCTL_PROCESS_BIND		0xffff

#define CORE_SID		0

#define SVM_IOCTL_RELEASE_PHYS32	0xfff3
#define SVM_REMAP_MEM_LEN_MAX		(16 * 1024 * 1024)
#define MMAP_PHY32_MAX (16 * 1024 * 1024)

static int probe_index;
static LIST_HEAD(child_list);
static DECLARE_RWSEM(svm_sem);
static struct rb_root svm_process_root = RB_ROOT;
static struct mutex svm_process_mutex;

struct core_device {
	struct device	dev;
	struct iommu_group	*group;
	struct iommu_domain	*domain;
	u8	smmu_bypass;
	struct list_head entry;
};

struct svm_device {
	unsigned long long	id;
	struct miscdevice	miscdev;
	struct device		*dev;
	phys_addr_t l2buff;
	unsigned long		l2size;
};

struct svm_bind_process {
	pid_t			vpid;
	u64			ttbr;
	u64			tcr;
	int			pasid;
	u32			flags;
#define SVM_BIND_PID		(1 << 0)
};

/*
 *svm_process is released in svm_notifier_release() when mm refcnt
 *goes down zero. We should access svm_process only in the context
 *where mm_struct is valid, which means we should always get mm
 *refcnt first.
 */
struct svm_process {
	struct pid		*pid;
	struct mm_struct	*mm;
	unsigned long		asid;
	struct rb_node		rb_node;
	struct mmu_notifier	notifier;
	/* For postponed release */
	struct rcu_head		rcu;
	int			pasid;
	struct mutex		mutex;
	struct rb_root		sdma_list;
	struct svm_device	*sdev;
	struct iommu_sva	*sva;
};

struct svm_sdma {
	struct rb_node node;
	unsigned long addr;
	int nr_pages;
	struct page **pages;
	atomic64_t ref;
};

struct svm_proc_mem {
	u32 dev_id;
	u32 len;
	u64 pid;
	u64 vaddr;
	u64 buf;
};

static char *svm_cmd_to_string(unsigned int cmd)
{
	switch (cmd) {
	case SVM_IOCTL_PROCESS_BIND:
		return "bind";
	case SVM_IOCTL_GET_PHYS:
		return "get phys";
	case SVM_IOCTL_SET_RC:
		return "set rc";
	case SVM_IOCTL_PIN_MEMORY:
		return "pin memory";
	case SVM_IOCTL_UNPIN_MEMORY:
		return "unpin memory";
	case SVM_IOCTL_REMAP_PROC:
		return "remap proc";
	case SVM_IOCTL_LOAD_FLAG:
		return "load flag";
	case SVM_IOCTL_RELEASE_PHYS32:
		return "release phys";
	default:
		return "unsupported";
	}

	return NULL;
}

/*
 * image word of slot
 * SVM_IMAGE_WORD_INIT: initial value, indicating that the slot is not used.
 * SVM_IMAGE_WORD_VALID: valid data is filled in the slot
 * SVM_IMAGE_WORD_DONE: the DMA operation is complete when the TS uses this address,
 * so, this slot can be freed.
 */
#define SVM_IMAGE_WORD_INIT	0x0
#define SVM_IMAGE_WORD_VALID	0xaa55aa55
#define SVM_IMAGE_WORD_DONE	0x55ff55ff

/*
 * The length of this structure must be 64 bytes, which is the agreement with the TS.
 * And the data type and sequence cannot be changed, because the TS core reads data
 * based on the data type and sequence.
 * image_word: slot status. For details, see SVM_IMAGE_WORD_xxx
 * pid: pid of process which ioctl svm device to get physical addr, it is used for
 * verification by TS.
 * data_type: used to determine the data type by TS. Currently, data type must be
 * SVM_VA2PA_TYPE_DMA.
 * char data[48]: for the data type SVM_VA2PA_TYPE_DMA, the DMA address is stored.
 */
struct svm_va2pa_slot {
	int image_word;
	int resv;
	int pid;
	int data_type;
	union {
		char user_defined_data[48];
		struct {
			unsigned long phys;
			unsigned long len;
			char reserved[32];
		};
	};
};

struct svm_va2pa_trunk {
	struct svm_va2pa_slot *slots;
	int slot_total;
	int slot_used;
	unsigned long *bitmap;
	struct mutex mutex;
};

struct svm_va2pa_trunk va2pa_trunk;

#define SVM_VA2PA_TRUNK_SIZE_MAX	0x3200000
#define SVM_VA2PA_MEMORY_ALIGN		64
#define SVM_VA2PA_SLOT_SIZE		sizeof(struct svm_va2pa_slot)
#define SVM_VA2PA_TYPE_DMA		0x1
#define SVM_MEM_REG			"va2pa trunk"
#define SVM_VA2PA_CLEAN_BATCH_NUM	0x80

struct device_node *svm_find_mem_reg_node(struct device *dev, const char *compat)
{
	int index = 0;
	struct device_node *tmp = NULL;
	struct device_node *np = dev->of_node;

	for (; ; index++) {
		tmp = of_parse_phandle(np, "memory-region", index);
		if (!tmp)
			break;

		if (of_device_is_compatible(tmp, compat))
			return tmp;

		of_node_put(tmp);
	}

	return NULL;
}

static int svm_parse_trunk_memory(struct device *dev, phys_addr_t *base, unsigned long *size)
{
	int err;
	struct resource r;
	struct device_node *trunk = NULL;

	trunk = svm_find_mem_reg_node(dev, SVM_MEM_REG);
	if (!trunk) {
		dev_err(dev, "Didn't find reserved memory\n");
		return -EINVAL;
	}

	err = of_address_to_resource(trunk, 0, &r);
	of_node_put(trunk);
	if (err) {
		dev_err(dev, "Couldn't address to resource for reserved memory\n");
		return -ENOMEM;
	}

	*base = r.start;
	*size = resource_size(&r);

	return 0;
}

static int svm_setup_trunk(struct device *dev, phys_addr_t base, unsigned long size)
{
	int slot_total;
	unsigned long *bitmap = NULL;
	struct svm_va2pa_slot *slot = NULL;

	if (!IS_ALIGNED(base, SVM_VA2PA_MEMORY_ALIGN)) {
		dev_err(dev, "Didn't aligned to %u\n", SVM_VA2PA_MEMORY_ALIGN);
		return -EINVAL;
	}

	if ((size == 0) || (size > SVM_VA2PA_TRUNK_SIZE_MAX)) {
		dev_err(dev, "Size of reserved memory is not right\n");
		return -EINVAL;
	}

	slot_total = size / SVM_VA2PA_SLOT_SIZE;
	if (slot_total < BITS_PER_LONG)
		return -EINVAL;

	bitmap = kvcalloc(slot_total / BITS_PER_LONG, sizeof(unsigned long), GFP_KERNEL);
	if (!bitmap) {
		dev_err(dev, "alloc memory failed\n");
		return -ENOMEM;
	}

	slot = ioremap(base, size);
	if (!slot) {
		kvfree(bitmap);
		dev_err(dev, "Ioremap trunk failed\n");
		return -ENXIO;
	}

	va2pa_trunk.slots = slot;
	va2pa_trunk.slot_used = 0;
	va2pa_trunk.slot_total = slot_total;
	va2pa_trunk.bitmap = bitmap;
	mutex_init(&va2pa_trunk.mutex);

	return 0;
}

static void svm_remove_trunk(struct device *dev)
{
	iounmap(va2pa_trunk.slots);
	kvfree(va2pa_trunk.bitmap);

	va2pa_trunk.slots = NULL;
	va2pa_trunk.bitmap = NULL;
}

static void svm_set_slot_valid(unsigned long index, unsigned long phys, unsigned long len)
{
	struct svm_va2pa_slot *slot = &va2pa_trunk.slots[index];

	slot->phys = phys;
	slot->len = len;
	slot->image_word = SVM_IMAGE_WORD_VALID;
	slot->pid = current->tgid;
	slot->data_type = SVM_VA2PA_TYPE_DMA;
	__bitmap_set(va2pa_trunk.bitmap, index, 1);
	va2pa_trunk.slot_used++;
}

static void svm_set_slot_init(unsigned long index)
{
	struct svm_va2pa_slot *slot = &va2pa_trunk.slots[index];

	slot->image_word = SVM_IMAGE_WORD_INIT;
	__bitmap_clear(va2pa_trunk.bitmap, index, 1);
	va2pa_trunk.slot_used--;
}

static void svm_clean_done_slots(void)
{
	int used = va2pa_trunk.slot_used;
	int count = 0;
	long temp = -1;
	phys_addr_t addr;
	unsigned long *bitmap = va2pa_trunk.bitmap;

	for (; count < used && count < SVM_VA2PA_CLEAN_BATCH_NUM;) {
		temp = find_next_bit(bitmap, va2pa_trunk.slot_total, temp + 1);
		if (temp == va2pa_trunk.slot_total)
			break;

		count++;
		if (va2pa_trunk.slots[temp].image_word != SVM_IMAGE_WORD_DONE)
			continue;

		addr = (phys_addr_t)va2pa_trunk.slots[temp].phys;
		put_page(pfn_to_page(PHYS_PFN(addr)));
		svm_set_slot_init(temp);
	}
}

static int svm_find_slot_init(unsigned long *index)
{
	int temp;
	unsigned long *bitmap = va2pa_trunk.bitmap;

	temp = find_first_zero_bit(bitmap, va2pa_trunk.slot_total);
	if (temp == va2pa_trunk.slot_total)
		return -ENOSPC;

	*index = temp;
	return 0;
}

static int svm_va2pa_trunk_init(struct device *dev)
{
	int err;
	phys_addr_t base;
	unsigned long size;

	err = svm_parse_trunk_memory(dev, &base, &size);
	if (err)
		return err;

	err = svm_setup_trunk(dev, base, size);
	if (err)
		return err;

	return 0;
}

static struct svm_process *find_svm_process(unsigned long asid)
{
	struct rb_node *node = svm_process_root.rb_node;

	while (node) {
		struct svm_process *process = NULL;

		process = rb_entry(node, struct svm_process, rb_node);
		if (asid < process->asid)
			node = node->rb_left;
		else if (asid > process->asid)
			node = node->rb_right;
		else
			return process;
	}

	return NULL;
}

static void insert_svm_process(struct svm_process *process)
{
	struct rb_node **p = &svm_process_root.rb_node;
	struct rb_node *parent = NULL;

	while (*p) {
		struct svm_process *tmp_process = NULL;

		parent = *p;
		tmp_process = rb_entry(parent, struct svm_process, rb_node);
		if (process->asid < tmp_process->asid)
			p = &(*p)->rb_left;
		else if (process->asid > tmp_process->asid)
			p = &(*p)->rb_right;
		else {
			WARN_ON_ONCE("asid already in the tree");
			return;
		}
	}

	rb_link_node(&process->rb_node, parent, p);
	rb_insert_color(&process->rb_node, &svm_process_root);
}

static void delete_svm_process(struct svm_process *process)
{
	rb_erase(&process->rb_node, &svm_process_root);
	RB_CLEAR_NODE(&process->rb_node);
}

static struct svm_device *file_to_sdev(struct file *file)
{
	return container_of(file->private_data,
			struct svm_device, miscdev);
}

static inline struct core_device *to_core_device(struct device *d)
{
	return container_of(d, struct core_device, dev);
}

static struct svm_sdma *svm_find_sdma(struct svm_process *process,
				unsigned long addr, int nr_pages)
{
	struct rb_node *node = process->sdma_list.rb_node;

	while (node) {
		struct svm_sdma *sdma = NULL;

		sdma = rb_entry(node, struct svm_sdma, node);
		if (addr < sdma->addr)
			node = node->rb_left;
		else if (addr > sdma->addr)
			node = node->rb_right;
		else if (nr_pages < sdma->nr_pages)
			node = node->rb_left;
		else if (nr_pages > sdma->nr_pages)
			node = node->rb_right;
		else
			return sdma;
	}

	return NULL;
}

static int svm_insert_sdma(struct svm_process *process, struct svm_sdma *sdma)
{
	struct rb_node **p = &process->sdma_list.rb_node;
	struct rb_node *parent = NULL;

	while (*p) {
		struct svm_sdma *tmp_sdma = NULL;

		parent = *p;
		tmp_sdma = rb_entry(parent, struct svm_sdma, node);
		if (sdma->addr < tmp_sdma->addr)
			p = &(*p)->rb_left;
		else if (sdma->addr > tmp_sdma->addr)
			p = &(*p)->rb_right;
		else if (sdma->nr_pages < tmp_sdma->nr_pages)
			p = &(*p)->rb_left;
		else if (sdma->nr_pages > tmp_sdma->nr_pages)
			p = &(*p)->rb_right;
		else {
			/*
			 * add reference count and return -EBUSY
			 * to free former alloced one.
			 */
			atomic64_inc(&tmp_sdma->ref);
			return -EBUSY;
		}
	}

	rb_link_node(&sdma->node, parent, p);
	rb_insert_color(&sdma->node, &process->sdma_list);

	return 0;
}

static void svm_remove_sdma(struct svm_process *process,
			    struct svm_sdma *sdma, bool try_rm)
{
	int null_count = 0;

	if (try_rm && (!atomic64_dec_and_test(&sdma->ref)))
		return;

	rb_erase(&sdma->node, &process->sdma_list);
	RB_CLEAR_NODE(&sdma->node);

	while (sdma->nr_pages--) {
		if (sdma->pages[sdma->nr_pages] == NULL) {
			pr_err("null pointer, nr_pages:%d.\n", sdma->nr_pages);
			null_count++;
			continue;
		}

		put_page(sdma->pages[sdma->nr_pages]);
	}

	if (null_count)
		dump_stack();

	kvfree(sdma->pages);
	kfree(sdma);
}

static int svm_pin_pages(unsigned long addr, int nr_pages,
			 struct page **pages)
{
	int err;

	err = get_user_pages_fast(addr, nr_pages, 1, pages);
	if (err > 0 && err < nr_pages) {
		while (err--)
			put_page(pages[err]);
		err = -EFAULT;
	} else if (err == 0) {
		err = -EFAULT;
	}

	return err;
}

static int svm_add_sdma(struct svm_process *process,
			unsigned long addr, unsigned long size)
{
	int err;
	struct svm_sdma *sdma = NULL;

	sdma = kzalloc(sizeof(struct svm_sdma), GFP_KERNEL);
	if (sdma == NULL)
		return -ENOMEM;

	atomic64_set(&sdma->ref, 1);
	sdma->addr = addr & PAGE_MASK;
	sdma->nr_pages = (PAGE_ALIGN(size + addr) >> PAGE_SHIFT) -
			 (sdma->addr >> PAGE_SHIFT);
	sdma->pages = kvcalloc(sdma->nr_pages, sizeof(char *), GFP_KERNEL);
	if (sdma->pages == NULL) {
		err = -ENOMEM;
		goto err_free_sdma;
	}

	/*
	 * If always pin the same addr with the same nr_pages, pin pages
	 * maybe should move after insert sdma with mutex lock.
	 */
	err = svm_pin_pages(sdma->addr, sdma->nr_pages, sdma->pages);
	if (err < 0) {
		pr_err("%s: failed to pin pages addr 0x%pK, size 0x%lx\n",
		       __func__, (void *)addr, size);
		goto err_free_pages;
	}

	err = svm_insert_sdma(process, sdma);
	if (err < 0) {
		err = 0;
		pr_debug("%s: sdma already exist!\n", __func__);
		goto err_unpin_pages;
	}

	return err;

err_unpin_pages:
	while (sdma->nr_pages--)
		put_page(sdma->pages[sdma->nr_pages]);
err_free_pages:
	kvfree(sdma->pages);
err_free_sdma:
	kfree(sdma);

	return err;
}

static int svm_pin_memory(unsigned long __user *arg)
{
	int err;
	struct svm_process *process = NULL;
	unsigned long addr, size, asid;

	if (!acpi_disabled)
		return -EPERM;

	if (arg == NULL)
		return -EINVAL;

	if (get_user(addr, arg))
		return -EFAULT;

	if (get_user(size, arg + 1))
		return -EFAULT;

	if ((addr + size <= addr) || (size >= (u64)UINT_MAX) || (addr == 0))
		return -EINVAL;

	asid = arm64_mm_context_get(current->mm);
	if (!asid)
		return -ENOSPC;

	mutex_lock(&svm_process_mutex);
	process = find_svm_process(asid);
	if (process == NULL) {
		mutex_unlock(&svm_process_mutex);
		err = -ESRCH;
		goto out;
	}
	mutex_unlock(&svm_process_mutex);

	mutex_lock(&process->mutex);
	err = svm_add_sdma(process, addr, size);
	mutex_unlock(&process->mutex);

out:
	arm64_mm_context_put(current->mm);

	return err;
}

static int svm_unpin_memory(unsigned long __user *arg)
{
	int err = 0, nr_pages;
	struct svm_sdma *sdma = NULL;
	unsigned long addr, size, asid;
	struct svm_process *process = NULL;

	if (!acpi_disabled)
		return -EPERM;

	if (arg == NULL)
		return -EINVAL;

	if (get_user(addr, arg))
		return -EFAULT;

	if (get_user(size, arg + 1))
		return -EFAULT;

	if (ULONG_MAX - addr < size)
		return -EINVAL;

	asid = arm64_mm_context_get(current->mm);
	if (!asid)
		return -ENOSPC;

	nr_pages = (PAGE_ALIGN(size + addr) >> PAGE_SHIFT) -
		   ((addr & PAGE_MASK) >> PAGE_SHIFT);
	addr &= PAGE_MASK;

	mutex_lock(&svm_process_mutex);
	process = find_svm_process(asid);
	if (process == NULL) {
		mutex_unlock(&svm_process_mutex);
		err = -ESRCH;
		goto out;
	}
	mutex_unlock(&svm_process_mutex);

	mutex_lock(&process->mutex);
	sdma = svm_find_sdma(process, addr, nr_pages);
	if (sdma == NULL) {
		mutex_unlock(&process->mutex);
		err = -ESRCH;
		goto out;
	}

	svm_remove_sdma(process, sdma, true);
	mutex_unlock(&process->mutex);

out:
	arm64_mm_context_put(current->mm);

	return err;
}

static void svm_unpin_all(struct svm_process *process)
{
	struct rb_node *node = NULL;

	while ((node = rb_first(&process->sdma_list)))
		svm_remove_sdma(process,
				rb_entry(node, struct svm_sdma, node),
				false);
}

static int svm_acpi_bind_core(struct core_device *cdev,	void *data)
{
	struct task_struct *task = NULL;
	struct svm_process *process = data;

	if (cdev->smmu_bypass)
		return 0;

	task = get_pid_task(process->pid, PIDTYPE_PID);
	if (!task) {
		pr_err("failed to get task_struct\n");
		return -ESRCH;
	}

	process->sva = iommu_sva_bind_device(&cdev->dev, task->mm, NULL);
	if (!process->sva) {
		pr_err("failed to bind device\n");
		return PTR_ERR(process->sva);
	}

	process->pasid = task->mm->pasid;
	put_task_struct(task);

	return 0;
}

static int svm_dt_bind_core(struct device *dev, void *data)
{
	struct task_struct *task = NULL;
	struct svm_process *process = data;
	struct core_device *cdev = to_core_device(dev);

	if (cdev->smmu_bypass)
		return 0;

	task = get_pid_task(process->pid, PIDTYPE_PID);
	if (!task) {
		pr_err("failed to get task_struct\n");
		return -ESRCH;
	}

	process->sva = iommu_sva_bind_device(dev, task->mm, NULL);
	if (!process->sva) {
		pr_err("failed to bind device\n");
		return PTR_ERR(process->sva);
	}

	process->pasid = task->mm->pasid;
	put_task_struct(task);

	return 0;
}

static void svm_dt_bind_cores(struct svm_process *process)
{
	device_for_each_child(process->sdev->dev, process, svm_dt_bind_core);
}

static void svm_acpi_bind_cores(struct svm_process *process)
{
	struct core_device *pos = NULL;

	list_for_each_entry(pos, &child_list, entry) {
		svm_acpi_bind_core(pos, process);
	}
}

static void svm_process_free(struct mmu_notifier *mn)
{
	struct svm_process *process = NULL;

	process = container_of(mn, struct svm_process, notifier);
	svm_unpin_all(process);
	arm64_mm_context_put(process->mm);
	kfree(process);
}

static void svm_process_release(struct svm_process *process)
{
	delete_svm_process(process);
	put_pid(process->pid);

	mmu_notifier_put(&process->notifier);
}

static void svm_notifier_release(struct mmu_notifier *mn,
					struct mm_struct *mm)
{
	struct svm_process *process = NULL;

	process = container_of(mn, struct svm_process, notifier);

	/*
	 * No need to call svm_unbind_cores(), as iommu-sva will do the
	 * unbind in its mm_notifier callback.
	 */

	mutex_lock(&svm_process_mutex);
	svm_process_release(process);
	mutex_unlock(&svm_process_mutex);
}

static struct mmu_notifier_ops svm_process_mmu_notifier = {
	.release	= svm_notifier_release,
	.free_notifier = svm_process_free,
};

static struct svm_process *
svm_process_alloc(struct svm_device *sdev, struct pid *pid,
		struct mm_struct *mm, unsigned long asid)
{
	struct svm_process *process = kzalloc(sizeof(*process), GFP_ATOMIC);

	if (!process)
		return ERR_PTR(-ENOMEM);

	process->sdev = sdev;
	process->pid = pid;
	process->mm = mm;
	process->asid = asid;
	process->sdma_list = RB_ROOT; //lint !e64
	mutex_init(&process->mutex);
	process->notifier.ops = &svm_process_mmu_notifier;

	return process;
}

static struct task_struct *svm_get_task(struct svm_bind_process params)
{
	struct task_struct *task = NULL;

	if (params.flags & ~SVM_BIND_PID)
		return ERR_PTR(-EINVAL);

	if (params.flags & SVM_BIND_PID) {
		struct mm_struct *mm = NULL;

		task = find_get_task_by_vpid(params.vpid);
		if (task == NULL)
			return ERR_PTR(-ESRCH);

		/* check the permission */
		mm = mm_access(task, PTRACE_MODE_ATTACH_REALCREDS);
		if (IS_ERR_OR_NULL(mm)) {
			pr_err("cannot access mm\n");
			put_task_struct(task);
			return ERR_PTR(-ESRCH);
		}

		mmput(mm);
	} else {
		get_task_struct(current);
		task = current;
	}

	return task;
}

static int svm_process_bind(struct task_struct *task,
		struct svm_device *sdev, u64 *ttbr, u64 *tcr, int *pasid)
{
	int err;
	unsigned long asid;
	struct pid *pid = NULL;
	struct svm_process *process = NULL;
	struct mm_struct *mm = NULL;

	if ((ttbr == NULL) || (tcr == NULL) || (pasid == NULL))
		return -EINVAL;

	pid = get_task_pid(task, PIDTYPE_PID);
	if (pid == NULL)
		return -EINVAL;

	mm = get_task_mm(task);
	if (!mm) {
		err = -EINVAL;
		goto err_put_pid;
	}

	asid = arm64_mm_context_get(mm);
	if (!asid) {
		err = -ENOSPC;
		goto err_put_mm;
	}

	/* If a svm_process already exists, use it */
	mutex_lock(&svm_process_mutex);
	process = find_svm_process(asid);
	if (process == NULL) {
		process = svm_process_alloc(sdev, pid, mm, asid);
		if (IS_ERR(process)) {
			err = PTR_ERR(process);
			mutex_unlock(&svm_process_mutex);
			goto err_put_mm_context;
		}
		err = mmu_notifier_register(&process->notifier, mm);
		if (err) {
			mutex_unlock(&svm_process_mutex);
			goto err_free_svm_process;
		}

		insert_svm_process(process);

		if (acpi_disabled)
			svm_dt_bind_cores(process);
		else
			svm_acpi_bind_cores(process);

		mutex_unlock(&svm_process_mutex);
	} else {
		mutex_unlock(&svm_process_mutex);
		arm64_mm_context_put(mm);
		put_pid(pid);
	}


	*ttbr = virt_to_phys(mm->pgd) | asid << ASID_SHIFT;
	*tcr  = read_sysreg(tcr_el1);
	*pasid = process->pasid;

	mmput(mm);
	return 0;

err_free_svm_process:
	kfree(process);
err_put_mm_context:
	arm64_mm_context_put(mm);
err_put_mm:
	mmput(mm);
err_put_pid:
	put_pid(pid);

	return err;
}

static pte_t *svm_get_pte(struct vm_area_struct *vma,
			  pud_t *pud,
			  unsigned long addr,
			  unsigned long *page_size,
			  unsigned long *offset)
{
	pte_t *pte = NULL;
	unsigned long size = 0;

	if (is_vm_hugetlb_page(vma)) {
		if (pud_present(*pud)) {
			if (pud_val(*pud) && !(pud_val(*pud) & PUD_TABLE_BIT)) {
				pte = (pte_t *)pud;
				*offset = addr & (PUD_SIZE - 1);
				size = PUD_SIZE;
			} else {
				pte = (pte_t *)pmd_offset(pud, addr);
				*offset = addr & (PMD_SIZE - 1);
				size = PMD_SIZE;
			}
		} else {
			pr_err("%s:hugetlb but pud not present\n", __func__);
		}
	} else {
		pmd_t *pmd = pmd_offset(pud, addr);

		if (pmd_none(*pmd))
			return NULL;

		if (pmd_trans_huge(*pmd)) {
			pte = (pte_t *)pmd;
			*offset = addr & (PMD_SIZE - 1);
			size = PMD_SIZE;
		} else {
			pte = pte_offset_map(pmd, addr);
			*offset = addr & (PAGE_SIZE - 1);
			size = PAGE_SIZE;
		}
	}

	if (page_size)
		*page_size = size;

	return pte;
}

/* Must be called with mmap_lock held */
static pte_t *svm_walk_pt(unsigned long addr, unsigned long *page_size,
			  unsigned long *offset)
{
	pgd_t *pgd = NULL;
	p4d_t *p4d = NULL;
	pud_t *pud = NULL;
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma = NULL;

	vma = find_vma(mm, addr);
	if (!vma)
		return NULL;

	pgd = pgd_offset(mm, addr);
	if (pgd_none(*pgd))
		return NULL;

	p4d = p4d_offset(pgd, addr);
	if (p4d_none(*p4d))
		return NULL;

	pud = pud_offset(p4d, addr);
	if (pud_none(*pud))
		return NULL;

	return svm_get_pte(vma, pud, addr, page_size, offset);
}

static int svm_get_phys(unsigned long __user *arg)
{
	int err;
	pte_t *ptep = NULL;
	pte_t pte;
	unsigned long index = 0;
	struct page *page;
	unsigned long addr, phys, offset;
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma = NULL;
	unsigned long len;

	if (!acpi_disabled)
		return -EPERM;

	if (get_user(addr, arg))
		return -EFAULT;

	down_read(&mm->mmap_lock);
	ptep = svm_walk_pt(addr, NULL, &offset);
	if (!ptep) {
		up_read(&mm->mmap_lock);
		return -EINVAL;
	}

	pte = READ_ONCE(*ptep);
	if (!pte_present(pte) || !(pfn_in_present_section(pte_pfn(pte)))) {
		up_read(&mm->mmap_lock);
		return -EINVAL;
	}

	page = pte_page(pte);
	get_page(page);

	phys = PFN_PHYS(pte_pfn(pte)) + offset;

	/* fix ts problem, which need the len to check out memory */
	len = 0;
	vma = find_vma(mm, addr);
	if (vma)
		len = vma->vm_end - addr;

	up_read(&mm->mmap_lock);

	mutex_lock(&va2pa_trunk.mutex);
	svm_clean_done_slots();
	if (va2pa_trunk.slot_used == va2pa_trunk.slot_total) {
		err = -ENOSPC;
		goto err_mutex_unlock;
	}

	err = svm_find_slot_init(&index);
	if (err)
		goto err_mutex_unlock;

	svm_set_slot_valid(index, phys, len);

	err = put_user(index * SVM_VA2PA_SLOT_SIZE, (unsigned long __user *)arg);
	if (err)
		goto err_slot_init;

	mutex_unlock(&va2pa_trunk.mutex);
	return 0;

err_slot_init:
	svm_set_slot_init(index);
err_mutex_unlock:
	mutex_unlock(&va2pa_trunk.mutex);
	put_page(page);
	return err;
}

static struct bus_type svm_bus_type = {
	.name		= "svm_bus",
};

static int svm_open(struct inode *inode, struct file *file)
{
	return 0;
}

static int svm_set_rc(unsigned long __user *arg)
{
	unsigned long addr, size, rc;
	unsigned long end, page_size, offset;
	pte_t *pte = NULL;
	struct mm_struct *mm = current->mm;

	if (acpi_disabled)
		return -EPERM;

	if (arg == NULL)
		return -EINVAL;

	if (get_user(addr, arg))
		return -EFAULT;

	if (get_user(size, arg + 1))
		return -EFAULT;

	if (get_user(rc, arg + 2))
		return -EFAULT;

	end = addr + size;
	if (addr >= end)
		return -EINVAL;

	down_read(&mm->mmap_lock);
	while (addr < end) {
		pte = svm_walk_pt(addr, &page_size, &offset);
		if (!pte) {
			up_read(&mm->mmap_lock);
			return -ESRCH;
		}
		pte->pte |= (rc & (u64)0x0f) << 59;
		addr += page_size - offset;
	}
	up_read(&mm->mmap_lock);

	return 0;
}

static long svm_remap_get_phys(struct mm_struct *mm, struct vm_area_struct *vma,
			       unsigned long addr, unsigned long *phys,
			       unsigned long *page_size, unsigned long *offset)
{
	long err = -EINVAL;
	pgd_t *pgd = NULL;
	p4d_t *p4d = NULL;
	pud_t *pud = NULL;
	pte_t *pte = NULL;

	if (mm == NULL || vma == NULL || phys == NULL ||
	    page_size == NULL || offset == NULL)
		return err;

	pgd = pgd_offset(mm, addr);
	if (pgd_none(*pgd))
		return err;

	p4d = p4d_offset(pgd, addr);
	if (p4d_none(*p4d))
		return err;

	pud = pud_offset(p4d, addr);
	if (pud_none(*pud))
		return err;

	pte = svm_get_pte(vma, pud, addr, page_size, offset);
	if (pte && pte_present(*pte)) {
		*phys = PFN_PHYS(pte_pfn(*pte));
		return 0;
	}

	return err;
}

static long svm_remap_proc(unsigned long __user *arg)
{
	long ret = -EINVAL;
	struct svm_proc_mem pmem;
	struct task_struct *ptask = NULL;
	struct mm_struct *pmm = NULL, *mm = current->mm;
	struct vm_area_struct *pvma = NULL, *vma = NULL;
	unsigned long end, vaddr, phys, buf, offset, pagesize;

	if (!acpi_disabled)
		return -EPERM;

	if (arg == NULL) {
		pr_err("arg is invalid.\n");
		return ret;
	}

	ret = copy_from_user(&pmem, (void __user *)arg, sizeof(pmem));
	if (ret) {
		pr_err("failed to copy args from user space.\n");
		return -EFAULT;
	}

	if (pmem.buf & (PAGE_SIZE - 1)) {
		pr_err("address is not aligned with page size, addr:%pK.\n",
		       (void *)pmem.buf);
		return -EINVAL;
	}

	if (pmem.pid) {
		ptask = find_get_task_by_vpid(pmem.pid);
		if (!ptask) {
			pr_err("No task for this pid\n");
			return -EINVAL;
		}
	} else {
		ptask = current;
	}

	pmm = ptask->mm;

	down_read(&mm->mmap_lock);
	down_read(&pmm->mmap_lock);

	pvma = find_vma(pmm, pmem.vaddr);
	if (pvma == NULL) {
		ret = -ESRCH;
		goto err;
	}

	vma = find_vma(mm, pmem.buf);
	if (vma == NULL) {
		ret = -ESRCH;
		goto err;
	}

	if (pmem.len > SVM_REMAP_MEM_LEN_MAX) {
		ret = -EINVAL;
		pr_err("too large length of memory.\n");
		goto err;
	}
	vaddr = pmem.vaddr;
	end = vaddr + pmem.len;
	buf = pmem.buf;
	vma->vm_flags |= VM_SHARED;
	if (end > pvma->vm_end || end < vaddr) {
		ret = -EINVAL;
		pr_err("memory length is out of range, vaddr:%pK, len:%u.\n",
		       (void *)vaddr, pmem.len);
		goto err;
	}

	do {
		ret = svm_remap_get_phys(pmm, pvma, vaddr,
					 &phys, &pagesize, &offset);
		if (ret) {
			ret = -EINVAL;
			goto err;
		}

		vaddr += pagesize - offset;

		do {
			if (remap_pfn_range(vma, buf, phys >> PAGE_SHIFT,
				PAGE_SIZE,
				__pgprot(vma->vm_page_prot.pgprot |
					 PTE_DIRTY))) {

				ret = -ESRCH;
				goto err;
			}

			offset += PAGE_SIZE;
			buf += PAGE_SIZE;
			phys += PAGE_SIZE;
		} while (offset < pagesize);

	} while (vaddr < end);

err:
	up_read(&pmm->mmap_lock);
	up_read(&mm->mmap_lock);
	put_task_struct(ptask);
	return ret;
}

static int svm_proc_load_flag(int __user *arg)
{
	static atomic_t l2buf_load_flag = ATOMIC_INIT(0);
	int flag;

	if (!acpi_disabled)
		return -EPERM;

	if (arg == NULL)
		return -EINVAL;

	if (0 == (atomic_cmpxchg(&l2buf_load_flag, 0, 1)))
		flag = 0;
	else
		flag = 1;

	return put_user(flag, arg);
}

static int svm_mmap(struct file *file, struct vm_area_struct *vma)
{
	int err;
	struct svm_device *sdev = file_to_sdev(file);

	if (!acpi_disabled)
		return -EPERM;

	if (vma->vm_flags & VM_PA32BIT) {
		unsigned long vm_size = vma->vm_end - vma->vm_start;
		struct page *page = NULL;

		if ((vma->vm_end < vma->vm_start) || (vm_size > MMAP_PHY32_MAX))
			return -EINVAL;

		/* vma->vm_pgoff transfer the nid */
		if (vma->vm_pgoff == 0)
			page = alloc_pages(GFP_KERNEL | GFP_DMA32,
					get_order(vm_size));
		else
			page = alloc_pages_node((int)vma->vm_pgoff,
					GFP_KERNEL | __GFP_THISNODE,
					get_order(vm_size));
		if (!page) {
			dev_err(sdev->dev, "fail to alloc page on node 0x%lx\n",
					vma->vm_pgoff);
			return -ENOMEM;
		}

		err = remap_pfn_range(vma,
				vma->vm_start,
				page_to_pfn(page),
				vm_size, vma->vm_page_prot);
		if (err)
			dev_err(sdev->dev,
				"fail to remap 0x%pK err=%d\n",
				(void *)vma->vm_start, err);
	} else {
		if ((vma->vm_end < vma->vm_start) ||
		    ((vma->vm_end - vma->vm_start) > sdev->l2size))
			return -EINVAL;

		vma->vm_page_prot = __pgprot((~PTE_SHARED) &
				    vma->vm_page_prot.pgprot);

		err = remap_pfn_range(vma,
				vma->vm_start,
				sdev->l2buff >> PAGE_SHIFT,
				vma->vm_end - vma->vm_start,
				__pgprot(vma->vm_page_prot.pgprot | PTE_DIRTY));
		if (err)
			dev_err(sdev->dev,
				"fail to remap 0x%pK err=%d\n",
				(void *)vma->vm_start, err);
	}

	return err;
}

static int svm_release_phys32(unsigned long __user *arg)
{
	struct mm_struct *mm = current->mm;
	struct vm_area_struct *vma = NULL;
	struct page *page = NULL;
	pte_t *pte = NULL;
	unsigned long phys, addr, offset;
	unsigned int len = 0;

	if (arg == NULL)
		return -EINVAL;

	if (get_user(addr, arg))
		return -EFAULT;

	down_read(&mm->mmap_lock);
	pte = svm_walk_pt(addr, NULL, &offset);
	if (pte && pte_present(*pte)) {
		phys = PFN_PHYS(pte_pfn(*pte)) + offset;
	} else {
		up_read(&mm->mmap_lock);
		return -EINVAL;
	}

	vma = find_vma(mm, addr);
	if (!vma) {
		up_read(&mm->mmap_lock);
		return -EFAULT;
	}

	page = phys_to_page(phys);
	len = vma->vm_end - vma->vm_start;

	__free_pages(page, get_order(len));

	up_read(&mm->mmap_lock);

	return 0;
}

static long svm_ioctl(struct file *file, unsigned int cmd,
		unsigned long arg)
{
	int err = -EINVAL;
	struct svm_bind_process params;
	struct svm_device *sdev = file_to_sdev(file);
	struct task_struct *task;

	if (!arg)
		return -EINVAL;

	if (cmd == SVM_IOCTL_PROCESS_BIND) {
		err = copy_from_user(&params, (void __user *)arg,
				sizeof(params));
		if (err) {
			dev_err(sdev->dev, "fail to copy params %d\n", err);
			return -EFAULT;
		}
	}

	switch (cmd) {
	case SVM_IOCTL_PROCESS_BIND:
		task = svm_get_task(params);
		if (IS_ERR(task)) {
			dev_err(sdev->dev, "failed to get task\n");
			return PTR_ERR(task);
		}

		err = svm_process_bind(task, sdev, &params.ttbr,
				&params.tcr, &params.pasid);
		if (err) {
			put_task_struct(task);
			dev_err(sdev->dev, "failed to bind task %d\n", err);
			return err;
		}

		put_task_struct(task);
		err = copy_to_user((void __user *)arg, &params,
				sizeof(params));
		if (err) {
			dev_err(sdev->dev, "failed to copy to user!\n");
			return -EFAULT;
		}
		break;
	case SVM_IOCTL_GET_PHYS:
		err = svm_get_phys((unsigned long __user *)arg);
		break;
	case SVM_IOCTL_SET_RC:
		err = svm_set_rc((unsigned long __user *)arg);
		break;
	case SVM_IOCTL_PIN_MEMORY:
		err = svm_pin_memory((unsigned long __user *)arg);
		break;
	case SVM_IOCTL_UNPIN_MEMORY:
		err = svm_unpin_memory((unsigned long __user *)arg);
		break;
	case SVM_IOCTL_REMAP_PROC:
		err = svm_remap_proc((unsigned long __user *)arg);
		break;
	case SVM_IOCTL_LOAD_FLAG:
		err = svm_proc_load_flag((int __user *)arg);
		break;
	case SVM_IOCTL_RELEASE_PHYS32:
		err = svm_release_phys32((unsigned long __user *)arg);
		break;
	default:
			err = -EINVAL;
		}

		if (err)
			dev_err(sdev->dev, "%s: %s failed err = %d\n", __func__,
					svm_cmd_to_string(cmd), err);

	return err;
}

static const struct file_operations svm_fops = {
	.owner			= THIS_MODULE,
	.open			= svm_open,
	.mmap			= svm_mmap,
	.unlocked_ioctl		= svm_ioctl,
};

static void cdev_device_release(struct device *dev)
{
	struct core_device *cdev = to_core_device(dev);

	if (!acpi_disabled)
		list_del(&cdev->entry);

	kfree(cdev);
}

static int svm_remove_core(struct device *dev, void *data)
{
	struct core_device *cdev = to_core_device(dev);

	if (!cdev->smmu_bypass) {
		iommu_dev_disable_feature(dev, IOMMU_DEV_FEAT_SVA);
		iommu_detach_group(cdev->domain, cdev->group);
		iommu_group_put(cdev->group);
		iommu_domain_free(cdev->domain);
	}

	device_unregister(&cdev->dev);

	return 0;
}

#ifdef CONFIG_ACPI
static int svm_acpi_add_core(struct svm_device *sdev,
		struct acpi_device *children, int id)
{
	int err;
	struct core_device *cdev = NULL;
	char *name = NULL;
	enum dev_dma_attr attr;
	const union acpi_object *obj;

	name = devm_kasprintf(sdev->dev, GFP_KERNEL, "svm_child_dev%d", id);
	if (name == NULL)
		return -ENOMEM;

	cdev = kzalloc(sizeof(*cdev), GFP_KERNEL);
	if (cdev == NULL)
		return -ENOMEM;
	cdev->dev.fwnode = &children->fwnode;
	cdev->dev.parent = sdev->dev;
	cdev->dev.bus = &svm_bus_type;
	cdev->dev.release = cdev_device_release;
	cdev->smmu_bypass = 0;
	list_add(&cdev->entry, &child_list);
	dev_set_name(&cdev->dev, "%s", name);

	err = device_register(&cdev->dev);
	if (err) {
		dev_info(&cdev->dev, "core_device register failed\n");
		list_del(&cdev->entry);
		kfree(cdev);
		return err;
	}

	attr = device_get_dma_attr(&children->dev);
	if (attr != DEV_DMA_NOT_SUPPORTED) {
		err = acpi_dma_configure(&cdev->dev, attr);
		if (err) {
			dev_dbg(&cdev->dev, "acpi_dma_configure failed\n");
			return err;
		}
	}

	err = acpi_dev_get_property(children, "hisi,smmu-bypass",
			DEV_PROP_U8, &obj);
	if (err)
		dev_info(&children->dev, "read smmu bypass failed\n");

	cdev->smmu_bypass = *(u8 *)obj->integer.value;

	cdev->group = iommu_group_get(&cdev->dev);
	if (IS_ERR_OR_NULL(cdev->group)) {
		dev_err(&cdev->dev, "smmu is not right configured\n");
		return -ENXIO;
	}

	cdev->domain = iommu_domain_alloc(sdev->dev->bus);
	if (cdev->domain == NULL) {
		dev_info(&cdev->dev, "failed to alloc domain\n");
		return -ENOMEM;
	}

	err = iommu_attach_group(cdev->domain, cdev->group);
	if (err) {
		dev_err(&cdev->dev, "failed group to domain\n");
		return err;
	}

	err = iommu_dev_enable_feature(&cdev->dev, IOMMU_DEV_FEAT_IOPF);
	if (err) {
		dev_err(&cdev->dev, "failed to enable iopf feature, %d\n", err);
		return err;
	}

	err = iommu_dev_enable_feature(&cdev->dev, IOMMU_DEV_FEAT_SVA);
	if (err) {
		dev_err(&cdev->dev, "failed to enable sva feature\n");
		return err;
	}

	return 0;
}

static int svm_acpi_init_core(struct svm_device *sdev)
{
	int err = 0;
	struct device *dev = sdev->dev;
	struct acpi_device *adev = ACPI_COMPANION(sdev->dev);
	struct acpi_device *cdev = NULL;
	int id = 0;

	down_write(&svm_sem);
	if (!svm_bus_type.iommu_ops) {
		err = bus_register(&svm_bus_type);
		if (err) {
			up_write(&svm_sem);
			dev_err(dev, "failed to register svm_bus_type\n");
			return err;
		}

		err = bus_set_iommu(&svm_bus_type, dev->bus->iommu_ops);
		if (err) {
			up_write(&svm_sem);
			dev_err(dev, "failed to set iommu for svm_bus_type\n");
			goto err_unregister_bus;
		}
	} else if (svm_bus_type.iommu_ops != dev->bus->iommu_ops) {
		err = -EBUSY;
		up_write(&svm_sem);
		dev_err(dev, "iommu_ops configured, but changed!\n");
		return err;
	}
	up_write(&svm_sem);

	list_for_each_entry(cdev, &adev->children, node) {
		err = svm_acpi_add_core(sdev, cdev, id++);
		if (err)
			device_for_each_child(dev, NULL, svm_remove_core);
	}

	return err;

err_unregister_bus:
	bus_unregister(&svm_bus_type);

	return err;
}
#else
static int svm_acpi_init_core(struct svm_device *sdev) { return 0; }
#endif

static int svm_of_add_core(struct svm_device *sdev, struct device_node *np)
{
	int err;
	struct resource res;
	struct core_device *cdev = NULL;
	char *name = NULL;

	name = devm_kasprintf(sdev->dev, GFP_KERNEL, "svm%llu_%s",
			sdev->id, np->name);
	if (name == NULL)
		return -ENOMEM;

	cdev = kzalloc(sizeof(*cdev), GFP_KERNEL);
	if (cdev == NULL)
		return -ENOMEM;

	cdev->dev.of_node = np;
	cdev->dev.parent = sdev->dev;
	cdev->dev.bus = &svm_bus_type;
	cdev->dev.release = cdev_device_release;
	cdev->smmu_bypass = of_property_read_bool(np, "hisi,smmu_bypass");
	dev_set_name(&cdev->dev, "%s", name);

	err = device_register(&cdev->dev);
	if (err) {
		dev_info(&cdev->dev, "core_device register failed\n");
		kfree(cdev);
		return err;
	}

	err = of_dma_configure(&cdev->dev, np, true);
	if (err) {
		dev_dbg(&cdev->dev, "of_dma_configure failed\n");
		return err;
	}

	err = of_address_to_resource(np, 0, &res);
	if (err) {
		dev_info(&cdev->dev, "no reg, FW should install the sid\n");
	} else {
		/* If the reg specified, install sid for the core */
		void __iomem *core_base = NULL;
		int sid = cdev->dev.iommu->fwspec->ids[0];

		core_base = ioremap(res.start, resource_size(&res));
		if (core_base == NULL) {
			dev_err(&cdev->dev, "ioremap failed\n");
			return -ENOMEM;
		}

		writel_relaxed(sid, core_base + CORE_SID);
		iounmap(core_base);
	}

	cdev->group = iommu_group_get(&cdev->dev);
	if (IS_ERR_OR_NULL(cdev->group)) {
		dev_err(&cdev->dev, "smmu is not right configured\n");
		return -ENXIO;
	}

	cdev->domain = iommu_domain_alloc(sdev->dev->bus);
	if (cdev->domain == NULL) {
		dev_info(&cdev->dev, "failed to alloc domain\n");
		return -ENOMEM;
	}

	err = iommu_attach_group(cdev->domain, cdev->group);
	if (err) {
		dev_err(&cdev->dev, "failed group to domain\n");
		return err;
	}

	err = iommu_dev_enable_feature(&cdev->dev, IOMMU_DEV_FEAT_IOPF);
	if (err) {
		dev_err(&cdev->dev, "failed to enable iopf feature, %d\n", err);
		return err;
	}

	err = iommu_dev_enable_feature(&cdev->dev, IOMMU_DEV_FEAT_SVA);
	if (err) {
		dev_err(&cdev->dev, "failed to enable sva feature, %d\n", err);
		return err;
	}

	return 0;
}

static int svm_dt_init_core(struct svm_device *sdev, struct device_node *np)
{
	int err = 0;
	struct device_node *child = NULL;
	struct device *dev = sdev->dev;

	down_write(&svm_sem);
	if (svm_bus_type.iommu_ops == NULL) {
		err = bus_register(&svm_bus_type);
		if (err) {
			up_write(&svm_sem);
			dev_err(dev, "failed to register svm_bus_type\n");
			return err;
		}

		err = bus_set_iommu(&svm_bus_type, dev->bus->iommu_ops);
		if (err) {
			up_write(&svm_sem);
			dev_err(dev, "failed to set iommu for svm_bus_type\n");
			goto err_unregister_bus;
		}
	} else if (svm_bus_type.iommu_ops != dev->bus->iommu_ops) {
		err = -EBUSY;
		up_write(&svm_sem);
		dev_err(dev, "iommu_ops configured, but changed!\n");
		return err;
	}
	up_write(&svm_sem);

	for_each_available_child_of_node(np, child) {
		err = svm_of_add_core(sdev, child);
		if (err)
			device_for_each_child(dev, NULL, svm_remove_core);
	}

	return err;

err_unregister_bus:
	bus_unregister(&svm_bus_type);

	return err;
}

int svm_get_pasid(pid_t vpid, int dev_id __maybe_unused)
{
	int pasid;
	unsigned long asid;
	struct task_struct *task = NULL;
	struct mm_struct *mm = NULL;
	struct svm_process *process = NULL;
	struct svm_bind_process params;

	params.flags = SVM_BIND_PID;
	params.vpid = vpid;
	params.pasid = -1;
	params.ttbr = 0;
	params.tcr = 0;
	task = svm_get_task(params);
	if (IS_ERR(task))
		return PTR_ERR(task);

	mm = get_task_mm(task);
	if (mm == NULL) {
		pasid = -EINVAL;
		goto put_task;
	}

	asid = arm64_mm_context_get(mm);
	if (!asid) {
		pasid = -ENOSPC;
		goto put_mm;
	}

	mutex_lock(&svm_process_mutex);
	process = find_svm_process(asid);
	mutex_unlock(&svm_process_mutex);
	if (process)
		pasid = process->pasid;
	else
		pasid = -ESRCH;

	arm64_mm_context_put(mm);
put_mm:
	mmput(mm);
put_task:
	put_task_struct(task);

	return pasid;
}
EXPORT_SYMBOL_GPL(svm_get_pasid);

static int svm_dt_setup_l2buff(struct svm_device *sdev, struct device_node *np)
{
	struct device_node *l2buff = of_parse_phandle(np, "memory-region", 0);

	if (l2buff) {
		struct resource r;
		int err = of_address_to_resource(l2buff, 0, &r);

		if (err) {
			of_node_put(l2buff);
			return err;
		}

		sdev->l2buff = r.start;
		sdev->l2size = resource_size(&r);
	}

	of_node_put(l2buff);
	return 0;
}

static int svm_device_probe(struct platform_device *pdev)
{
	int err = -1;
	struct device *dev = &pdev->dev;
	struct svm_device *sdev = NULL;
	struct device_node *np = dev->of_node;
	int alias_id;

	if (acpi_disabled && np == NULL)
		return -ENODEV;

	if (!dev->bus) {
		dev_dbg(dev, "this dev bus is NULL\n");
		return -EPROBE_DEFER;
	}

	if (!dev->bus->iommu_ops) {
		dev_dbg(dev, "defer probe svm device\n");
		return -EPROBE_DEFER;
	}

	sdev = devm_kzalloc(dev, sizeof(*sdev), GFP_KERNEL);
	if (sdev == NULL)
		return -ENOMEM;

	if (!acpi_disabled) {
		err = device_property_read_u64(dev, "svmid", &sdev->id);
		if (err) {
			dev_err(dev, "failed to get this svm device id\n");
			return err;
		}
	} else {
		alias_id = of_alias_get_id(np, "svm");
		if (alias_id < 0)
			sdev->id = probe_index;
		else
			sdev->id = alias_id;
	}

	sdev->dev = dev;
	sdev->miscdev.minor = MISC_DYNAMIC_MINOR;
	sdev->miscdev.fops = &svm_fops;
	sdev->miscdev.name = devm_kasprintf(dev, GFP_KERNEL,
			SVM_DEVICE_NAME"%llu", sdev->id);
	if (sdev->miscdev.name == NULL)
		return -ENOMEM;

	dev_set_drvdata(dev, sdev);
	err = misc_register(&sdev->miscdev);
	if (err) {
		dev_err(dev, "Unable to register misc device\n");
		return err;
	}

	if (!acpi_disabled) {
		err = svm_acpi_init_core(sdev);
		if (err) {
			dev_err(dev, "failed to init acpi cores\n");
			goto err_unregister_misc;
		}
	} else {
		/*
		 * Get the l2buff phys address and size, if it do not exist
		 * just warn and continue, and runtime can not use L2BUFF.
		 */
		err = svm_dt_setup_l2buff(sdev, np);
		if (err)
			dev_warn(dev, "Cannot get l2buff\n");

		if (svm_va2pa_trunk_init(dev)) {
			dev_err(dev, "failed to init va2pa trunk\n");
			goto err_unregister_misc;
		}

		err = svm_dt_init_core(sdev, np);
		if (err) {
			dev_err(dev, "failed to init dt cores\n");
			goto err_remove_trunk;
		}

		probe_index++;
	}

	mutex_init(&svm_process_mutex);

	return err;

err_remove_trunk:
	svm_remove_trunk(dev);

err_unregister_misc:
	misc_deregister(&sdev->miscdev);

	return err;
}

static int svm_device_remove(struct platform_device *pdev)
{
	struct device *dev = &pdev->dev;
	struct svm_device *sdev = dev_get_drvdata(dev);

	device_for_each_child(sdev->dev, NULL, svm_remove_core);
	misc_deregister(&sdev->miscdev);

	return 0;
}

static const struct acpi_device_id svm_acpi_match[] = {
	{ "HSVM1980", 0},
	{ }
};
MODULE_DEVICE_TABLE(acpi, svm_acpi_match);

static const struct of_device_id svm_of_match[] = {
	{ .compatible = "hisilicon,svm" },
	{ }
};
MODULE_DEVICE_TABLE(of, svm_of_match);

/*svm acpi probe and remove*/
static struct platform_driver svm_driver = {
	.probe	=	svm_device_probe,
	.remove	=	svm_device_remove,
	.driver	=	{
		.name = SVM_DEVICE_NAME,
		.acpi_match_table = ACPI_PTR(svm_acpi_match),
		.of_match_table = svm_of_match,
	},
};

module_platform_driver(svm_driver);

MODULE_DESCRIPTION("Hisilicon SVM driver");
MODULE_AUTHOR("Fang Lijun <fanglijun3@huawei.com>");
MODULE_LICENSE("GPL v2");
