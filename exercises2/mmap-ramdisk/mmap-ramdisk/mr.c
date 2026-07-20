#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/mm.h>
#include <linux/atomic.h>
#include <linux/writeback.h>
//#include <linux/backing-dev.h>    // noop_backing_dev_info is exported to GPL
#include <linux/pagemap.h>
#include <linux/rmap.h>

#define USE_INSERT_PFN
#define X86_ARCH  0

#ifndef PAGE_CACHE_SHIFT
#define PAGE_CACHE_SHIFT PAGE_SHIFT
#endif

static DEFINE_SPINLOCK(mr_vma_list_lock);
static LIST_HEAD(mr_vma_list);

//static inline pteval_t native_pte_val(pte_t pte)
//{
//        return pte.pte;
//}
//
//static inline pteval_t pte_flags(pte_t pte)
//{
//        return native_pte_val(pte) & PTE_FLAGS_MASK;
//}
/*
 * I modified mm/memory.c to print pte flags before and after the
 * do_wp_page() promotion in the handle_pte_fault() function..
 *
 * After the page is allocated write-protect when mrc.c does printf()
 * on a byte in the mmap'ed area, the pte flags are:
 *
 *    0x29  (Accessed, Write Through, Present)
 *
 * mrc.c then modifies the page by writing 1 to the first byte, which causes
 * a fault to the already-mapped page due to permissions being writeProtect.
 * The page_mkwrite() gets called and we see:
 *
 *    0x25  (Accessed, Userspace, Present)
 *
 * This is ok.  I see that in handle_pte_fault(), just after the call to do_wp_page()
 * that the page is marked Dirty and Writeable.
 *
 * I added a silly ioctl() to demote pages to WriteProtect to see the behaviour.
 * Before demotion, the page flags are:
 *
 *    0x67  (Dirty, Accessed, Userspace, Writeable, Present)
 *
 * and 0x67 after demotion (Clearing the Writeable bit).
 *
 * The next write from mrc.c does then cause another page fault, as evidenced by
 * a call into page_mkwrite().
 */
//extern unsigned long xx_mmap_fault_address;

/*
 * At what user virtual address is page expected in @vma?
 * Returns virtual address or -EFAULT if page's index/offset is not
 * within the range mapped the @vma.
 */
inline unsigned long
vma_address(struct page *page, struct vm_area_struct *vma)
{
	pgoff_t pgoff = page->index << (PAGE_CACHE_SHIFT - PAGE_SHIFT);
	unsigned long address;

	address = vma->vm_start + ((pgoff - vma->vm_pgoff) << PAGE_SHIFT);
	if (unlikely(address < vma->vm_start || address >= vma->vm_end)) {
		/* page should be within @vma mapping range */
		return -EFAULT;
	}
	return address;
}

#if 0
/*
 * At what user virtual address is page expected in vma?
 * Caller should check the page is actually part of the vma.
 */
static unsigned long mr_page_address_in_vma(struct page *page, struct vm_area_struct *vma)
{
	if (PageAnon(page)) {
		struct anon_vma *page__anon_vma = page_anon_vma(page);
		/*
		 * Note: swapoff's unuse_vma() is more efficient with this
		 * check, and needs it to match anon_vma when KSM is active.
		 */
		if (!vma->anon_vma || !page__anon_vma ||
		    vma->anon_vma->root != page__anon_vma->root)
			return -EFAULT;
	} else if (page->mapping && !(vma->vm_flags & VM_NONLINEAR)) {
		if (!vma->vm_file ||
		    vma->vm_file->f_mapping != page->mapping)
			return -EFAULT;
	} else
		return -EFAULT;
	return vma_address(page, vma);
}
#endif

/*
 * Check that @page is mapped at @address into @mm.
 *
 * If @sync is false, page_check_address may perform a racy check to avoid
 * the page table lock when the pte is not present (helpful when reclaiming
 * highly shared pages).
 *
 * On success returns with pte mapped and locked.
 */
static pte_t *mr__page_check_address(struct page *page, struct mm_struct *mm,
			  unsigned long address, spinlock_t **ptlp, int sync)
{
	pgd_t *pgd;
	p4d_t *p4d;
	pud_t *pud;
	pmd_t *pmd;
	pte_t *pte;
	spinlock_t *ptl;

	pgd = pgd_offset(mm, address);
	if (!pgd_present(*pgd))
		return NULL;

	p4d = p4d_offset(pgd, address);
	if (!p4d_present(*p4d))
		return NULL;

	pud = pud_offset(p4d, address);
	if (!pud_present(*pud))
		return NULL;

	pmd = pmd_offset(pud, address);
	if (!pmd_present(*pmd))
		return NULL;
	if (pmd_trans_huge(*pmd))
		return NULL;

	pte = pte_offset_map(pmd, address);
	/* Make a quick check before getting the lock */
	if (!sync && !pte_present(*pte)) {
		pte_unmap(pte);
		return NULL;
	}

	ptl = pte_lockptr(mm, pmd);
	spin_lock(ptl);
	if (pte_present(*pte) && page_to_pfn(page) == pte_pfn(*pte)) {
		*ptlp = ptl;
		return pte;
	}
	pte_unmap_unlock(pte, ptl);
	return NULL;
}

struct mr_vma_priv {
	struct vm_area_struct *vma;
	atomic_t refcnt;
	spinlock_t lock;
	int count, max;
	/*
	 * Probably need a pgoff as well, in case an existing VMA is expanded.
	 * In that case, one vma would have multiple backing mr_vma_privs,
	 * covering the entire range of the vma
	 */
	struct list_head list;
	struct page *pages[0];
};

static void mr_do_page_protect(struct mr_vma_priv *vp)
{
	struct vm_area_struct *vma = vp->vma;
	int i;

	pr_info("page_protect: vma=%p\n", vma);
	for (i = 0; i < vp->max; i++) {
		struct page *page = vp->pages[i];
		unsigned long address;
		spinlock_t *ptl;
		pte_t *ptep;

		if (!page)
			continue;
		pr_info("page_protect: page=%p\n", page);

		// Gives me EFAULT
		//address = mr_page_address_in_vma(page, vma);
		address = vma_address(page, vma);
		if (address == -EFAULT) {
			pr_info("%s: Got EFAULT\n", __FUNCTION__);
			return;
		} else if (address == 0) {
			pr_info("%s: Got NULL\n", __FUNCTION__);
			return;
		}
		pr_info("page_protect: address=%lx\n", address);
		ptep = mr__page_check_address(page, vma->vm_mm, address, &ptl, 0);
		if (!ptep) {
			pr_info("%s: No pte\n", __FUNCTION__);
			return;
		}

#if X86_ARCH
		pr_info("page_protect: Before protect pte_flags=%lx\n", pte_flags(*ptep));
#endif
		//if (pte_dirty(*ptep))
		//	pr_info("PTE is dirty\n");
		//else
		//	pr_info("PTE is clean\n");
		ptep_set_wrprotect(vma->vm_mm, address, ptep);
		pte_unmap_unlock(ptep, ptl);
		/*
		 * Not exported in SLES11.2. try flush_cache_page()?
		 * I'm not sure that anything is required on x86...
		 */
		//mmu_notifier_invalidate_page(vma->vm_mm, address);
		flush_cache_page(vma, address, pte_pfn(*ptep));
#if X86_ARCH
		pr_info("page_protect: After protect pte_flags=%lx\n", pte_flags(*ptep));
#endif
	}
}

static void mr_do_page_protect_all(void)
{
	struct mr_vma_priv *vp;

	if (list_empty(&mr_vma_list)) {
		pr_info("List entry was NULL\n");
		return;
	}

	list_for_each_entry(vp, &mr_vma_list, list) {
		mr_do_page_protect(vp);
	}

	return;
}

static void mr_vm_open(struct vm_area_struct * vma)
{
	struct mr_vma_priv *vp;
	vp = vma->vm_private_data;
	atomic_inc(&vp->refcnt);
}

static void mr_vm_close(struct vm_area_struct * vma)
{
	int i;
	struct mr_vma_priv *vp;
	vp = vma->vm_private_data;

	pr_info("close: refcnt=%u\n", atomic_read(&vp->refcnt));

	if (!atomic_dec_and_test(&vp->refcnt))
		return;

	for(i=0; i<vp->max; i++) {
		struct page *page;
		page = vp->pages[i];

		if (!page)
			continue;

		vp->pages[i] = NULL;

#if 0
		ClearPageReserved(page);
#endif
		pr_info("mr: freeing page=%p "
			"Ref=%d Act=%d Drt=%u Rsr=%d U2d=%d Err=%d\n",
			page,
			PageReferenced(page),
			PageActive(page),
			PageDirty(page),
			PageReserved(page),
			PageUptodate(page),
			PageError(page)
			);
		__free_page(page);
		vp->count --;
	}
	//xx_mmap_fault_address = 0;

	spin_lock(&mr_vma_list_lock);
	list_del(&vp->list);
	spin_unlock(&mr_vma_list_lock);

	pr_info("close: free\n");

	vfree(vp);
}

static int mr_vm_fault(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
	struct mr_vma_priv *vp;
	struct page *page;
	unsigned long index;
#ifndef USE_INSERT_PFN
	unsigned long pfn;
#endif

	vp = vma->vm_private_data;

	pr_info("fault: vma=%p file=%p as=%p va=%p\n", vma,
			vma->vm_file,
			vma->vm_file ? vma->vm_file->f_mapping : NULL,
			(void*)vmf->address);

	pr_info("fault: fault pfoff=%lu\n", vmf->pgoff);

	index = vmf->pgoff;

	pr_info("fault: fault index=%lu/%u\n", index, vp->max);
	if (index >= vp->max)
		return VM_FAULT_SIGBUS;

	page = vp->pages[index];
	if (page) {
		pr_info("fault: old page=%08lx\n", (uintptr_t)page);
	} else {
		page = alloc_page(GFP_KERNEL);
		if (!page)
			return VM_FAULT_OOM;

		spin_lock(&vp->lock);
		if (!vp->pages[index]) {
			vp->pages[index] = page;
			page->index = index;
			vp->count ++;
		} else {
			__free_page(page);
			page = vp->pages[index];
		}
		spin_unlock(&vp->lock);
		pr_info("fault: new page=%08lx\n", (uintptr_t)page);
	}

#if 0
	SetPageReserved(page);
#endif

	get_page(page);
	//xx_mmap_fault_address = (unsigned long)vmf->virtual_address;

#ifndef USE_INSERT_PFN
	pfn = page_to_pfn(page);
	vm_insert_pfn(vma, (unsigned long)vmf->virtual_address, pfn);
	return VM_FAULT_NOPAGE;
#else
	vmf->page = page;
	return 0;
#endif
}

#if 0
static int mr_vm_access(struct vm_area_struct *vma, unsigned long addr,
		void *buf, int len, int write)
{
	return 0;
}
#endif


/* Notify when pages are promoted from WriteProtect to Read/Write */
static int mr_vm_mkwrite(struct vm_fault *vmf)
{
	struct vm_area_struct *vma = vmf->vma;
	pr_info("mkwrite: vma=%p, file=%p, as=%p, va=%p\n", vma, vma->vm_file,
			vma->vm_file ? vma->vm_file->f_mapping : NULL,
			(void*)vmf->address);
	return VM_FAULT_LOCKED;
}

static struct vm_operations_struct mr_vmops = {
	.open   = mr_vm_open,
	.close  = mr_vm_close,
	.fault  = mr_vm_fault,
	.page_mkwrite = mr_vm_mkwrite,
#if 0
	.access = mr_vm_access,
#endif
};


static int mr_mmap(struct file *filp, struct vm_area_struct *vma)
{
	struct mr_vma_priv *vp;
	unsigned long size = vma->vm_end - vma->vm_start;
	unsigned pages, vp_size;

	pr_info("mmap: vma: start=%lx end=%lx size=%lu pgoff=%lx\n",
			vma->vm_start,
			vma->vm_end,
			size,
			vma->vm_pgoff);

	if (vma->vm_pgoff) {
		pr_err("memory region assigned\n");
		return -EINVAL;
	}

	if (size & ~PAGE_MASK) {
		pr_err("size not aligned: %ld\n", size);
		return -ENXIO;
	}

	/* we only support shared mappings. Copy on write mappings are
	   rejected here. A shared mapping that is writeable must have the
	   shared flag set.
	   */
	if ((vma->vm_flags & VM_WRITE) && !(vma->vm_flags & VM_SHARED)) {
		pr_err("writeable mappings must be shared, rejecting\n");
		return -EINVAL;
	}

	pages = size >> PAGE_SHIFT;
	vp_size = sizeof(*vp) + pages * sizeof(vp->pages[0]);

	pr_info("mmap: new context for pages=%u\n", pages);

	vp = vzalloc(vp_size);
	if (!vp) {
		pr_err("failed to allocate %u bytes\n", vp_size);
		return -ENOMEM;
	}

	vp->vma = vma;
	vp->max = pages;

	spin_lock_init(&vp->lock);
	atomic_set(&vp->refcnt, 1);

	vma->vm_private_data = vp;
#if 0
	vma->vm_flags |= VM_IO;           // uses vmops->access()
	vma->vm_flags |= VM_LOCKED;       // pages are mlock()'ed
	vma->vm_flags |= VM_PFNMAP;       // linear mapping ptr -> PFN
	vma->vm_flags |= VM_RESERVED;     // unevictable, kernel allocated
	vma->vm_flags |= VM_DONTEXPAND;   // no mremap(), kernel allocated
#endif

	vma->vm_ops = &mr_vmops;

	spin_lock(&mr_vma_list_lock);
	list_add(&vp->list, &mr_vma_list);
	spin_unlock(&mr_vma_list_lock);

	pr_info("mmap: vm_file=%p\n", vma->vm_file);

	return 0;
}

static int mr_writepages(struct address_space *mapping,
		struct writeback_control *wbc)
{
	pr_info("writepages: mapping=%p nr=%lu skip=%lu range=%llx:%llx sync=%d\n",
			mapping, wbc->nr_to_write, wbc->pages_skipped,
			wbc->range_start, wbc->range_end, wbc->sync_mode);
	pr_info("writepages: host=%p\n", mapping->host);
	return 0;
}

static int mr_writepage(struct page *page, struct writeback_control *wbc)
{
	pr_info("writepage: page=%p nr=%lu skip=%lu range=%llx:%llx sync=%d\n",
			page, wbc->nr_to_write, wbc->pages_skipped,
			wbc->range_start, wbc->range_end, wbc->sync_mode);
	return 0;
}

static int mr_readpage(struct file *filp, struct page *page)
{
	pr_info("readpage: page=%p\n", page);
	return 0;
}

static int mr_write_begin(struct file *file, struct address_space *mapping,
		loff_t pos, unsigned len, unsigned flags,
		struct page **pagep, void **fsdata)
{
	pr_info("write_begin: mapping=%p pos=%llx len=%u fl=%u "
		"page=%p fsdata=%p\n",
		mapping, pos, len, flags, pagep ? *pagep : NULL,
		fsdata ? *fsdata : NULL);
	return 0;
}

static int mr_write_end(struct file *file, struct address_space *mapping,
		loff_t pos, unsigned len, unsigned copied,
		struct page *page, void *fsdata)
{
	pr_info("write_end: mapping=%p pos=%llx len=%u copied=%u "
		"page=%p fsdata=%p\n",
		mapping, pos, len, copied, page, fsdata);
	return 0;
}

static int mr_set_page_dirty(struct page *page)
{
	int rc;

	rc = __set_page_dirty_nobuffers(page);

	return rc;
}

#ifndef CONFIG_MMU
static unsigned mr_mmap_capabilities(struct file *file)
{
	return NOMMU_MAP_COPY
		| NOMMU_MAP_DIRECT
		| NOMMU_MAP_READ
		| NOMMU_MAP_WRITE
		| NOMMU_MAP_EXEC;
}
#endif

static struct address_space_operations mr_asops = {
	.readpage = mr_readpage,
	.writepage = mr_writepage,
	.writepages = mr_writepages,
	.write_begin = mr_write_begin,
	.write_end = mr_write_end,
	.set_page_dirty = mr_set_page_dirty,
};

static int mr_open(struct inode *inode, struct file *filp)
{
	int rc;

	rc = nonseekable_open(inode, filp);
	if (rc)
		return rc;

	pr_info("open: inode=%p i_mapping=%p i_data=%p\n",
			inode, inode->i_mapping, &inode->i_data);

	pr_info("open: mapping->a_ops=%p data->a_ops=%p\n",
			inode->i_mapping
			? inode->i_mapping->a_ops : NULL,
			inode->i_data.a_ops);

	if (inode->i_mapping && inode->i_mapping->a_ops) {
		pr_info("open: mapping->a_ops writepage=%p\n",
				inode->i_mapping->a_ops->writepage);
	}

	if (inode->i_data.a_ops) {
		pr_info("open: data->a_ops writepage=%p\n",
				inode->i_data.a_ops->writepage);
	}

	inode->i_mapping->a_ops =
		inode->i_data.a_ops = &mr_asops;

	return rc;
}

static int mr_fsync(struct file *filp, loff_t start, loff_t end, int datasync)
{
	int rc;
	struct inode *inode = filp->f_mapping->host;
#if 0
	struct writeback_control wbc = {
		.sync_mode = WB_SYNC_ALL,
		.nr_to_write = 0,       /* sys_fsync did this */
	};
	if (!inode)
		return -EINVAL;
	rc = sync_inode(inode, &wbc);
#elif 0
	if (!inode)
		return -EINVAL;

	pr_info("fsync: file=%p f->f_m=%p i->i_m=%p range=%llx:%llx\n",
			filp, filp->f_mapping, inode->i_mapping,
			start, end);

	pr_info("fsync: i_mapping->nrpages=%lu\n", inode->i_mapping->nrpages);
	pr_info("fsync: i_data.a_ops->nrpages=%lu\n", inode->i_data.nrpages);

	rc = filemap_write_and_wait_range(inode->i_mapping, start, end);
	pr_info("fsync: rc=%d\n", rc);
	if (!rc) {
		mutex_lock(&inode->i_mutex);
		rc = sync_inode_metadata(inode, 1);
		mutex_unlock(&inode->i_mutex);
	}
#else
	struct address_space *im = inode->i_mapping;
	if (!inode)
		return -EINVAL;

	pr_info("fsync: file=%p f->f_m=%p i->i_m=%p range=%llx:%llx\n",
			filp, filp->f_mapping, im,
			start, end);

	pr_info("fsync: i_mapping->nrpages=%lu\n", im->nrpages);
	pr_info("fsync: i_data.a_ops->nrpages=%lu\n", inode->i_data.nrpages);

#ifdef _LINUX_BACKING_DEV_H
	pr_info("fsync: cap=%d\n",
			mapping_cap_writeback_dirty(im));
#endif

	pr_info("fsync: writepages=%p, writepage=%p\n",
			im->a_ops->writepages,
			im->a_ops->writepage);

	rc = filemap_fdatawrite_range(im, start, end);
	pr_info("fsync: rc=%d\n", rc);
	if (rc != -EIO) {
		int rc2 = filemap_fdatawait_range(im, start, end);
		pr_info("fsync: rc2=%d\n", rc2);
		if (!rc)
			rc = rc2;
	}

	if (!rc) {
		inode = igrab(inode);
		rc = sync_inode_metadata(inode, 1);
		iput(inode);
	}
#endif
	return rc;
}

static ssize_t mr_sendpage(struct file *filp, struct page *page, int offset,
		size_t size, loff_t *ppos, int more)
{
	pr_info("sendpage: page=%p ofs=%u size=%lu pos=%llu more=%u\n",
			page, offset, size, *ppos, more);

	return size;
}

ssize_t mr_write(struct file *filp, const char __user *buf, size_t size,
		loff_t *ppos)
{
	pr_info("write: buf=%p size=%lu pos=%llu\n",
			buf, size, *ppos);

	return size;
}


static long mr_ioctl(struct file *filp, unsigned int command, unsigned long arg)
{

	pr_info("ioctl: cmd=%d\n", command);
	switch (command) {
	case 55:
		mr_do_page_protect_all();
		return 0;
	default:
		return -EINVAL;
	}
}


static const struct file_operations mr_fops = {
	.owner = THIS_MODULE,
	.open  = mr_open,
	.fsync = mr_fsync,
	.mmap  = mr_mmap,
	.sendpage = mr_sendpage,
	.write = mr_write,
	.unlocked_ioctl = mr_ioctl,
#ifndef CONFIG_MMU
	.mmap_capabilities = mr_mmap_capabilities,
#endif
};

static struct miscdevice mr_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = "mr",
	.fops  = &mr_fops,
};

static int __init mr_init(void)
{
	pr_info("mr: init\n");

	if (misc_register(&mr_misc)) {
		pr_err("unable to get major for mr module\n");
		return -EBUSY;
	}

	return 0;
}

static void __exit mr_exit(void)
{
	pr_info("mr: exit\n");

	misc_deregister(&mr_misc);
}

module_init(mr_init);
module_exit(mr_exit);
