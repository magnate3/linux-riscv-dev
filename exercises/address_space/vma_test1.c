/*
 *  mtest_dump_vma_list（）：打印出当前进程的各个VMA，这个功能我们简称"listvma"
 *  mtest_find_vma()： 找出某个虚地址所在的VMA，这个功能我们简称"findvma"
 *  my_follow_page( )：根据页表，求出某个虚地址所在的物理页面，这个功能我们简称"findpage"
 *  mtest_write_val(), 在某个地址写上具体数据，这个功能我们简称"writeval".
 */
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/string.h>
#include <linux/vmalloc.h>
#include <asm/uaccess.h>
#include <linux/init.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/vmalloc.h>
#include <linux/rmap.h>
#include <asm/current.h>
#include <linux/sched.h>
#include <linux/highmem.h>
#include <linux/pagemap.h>
#include <linux/rbtree.h>
#include <linux/interval_tree.h> // interval_tree_iter_first
#include <linux/interval_tree_generic.h> // INTERVAL_TREE_DEFINE
#include<linux/page-flags.h>
MODULE_LICENSE("GPL");
struct task_struct * task =NULL;
void show_is_high(struct page *p)
{
     if(PageHighMem(p)) 
     {
         pr_info("page is high \n");
     }
     else 
     {
         pr_info("page is not  high \n");
     }
}
static int find_task(pid_t nr)
{
	struct pid *vpid = find_vpid(nr);
        pr_info("The process pid %d \n", nr);
	if (vpid != NULL) {
		printk("the find_vpid result's count is: %d\n",
		       vpid->count.counter);
		printk("the find_vpid result's level is: %d\n", vpid->level);
	} else {
		printk("failed to find_vpid");
	}
        task=pid_task(vpid,PIDTYPE_PID);
        if (task) {
            pr_info("The process is \"%s\" (pid %i)\n", task->comm, task->pid);
        }
        else
        {
            pr_info("The process is not exist \n");
        }
	return 0;
}
int va_to_phy(unsigned long va)
{
	unsigned long long pageFN;
	unsigned long long pa;

	pgd_t *pgd;
	pmd_t *pmd;
	pud_t *pud;
	pte_t *pte;
	
	struct mm_struct *mm;

	int found = 0;
	mm = task->mm;
        if (!task) {
            pr_info("The process is not exist \n");
            return 0;
        }
	pgd  = pgd_offset(mm,va);
	printk(KERN_ALERT "virt Address: 0x%16lx\n", va);
	if(!pgd_none(*pgd) && !pgd_bad(*pgd))
	{
		pud = pud_offset(pgd,va);
		if(!pud_none(*pud) && !pud_bad(*pud))
		{
			pmd = pmd_offset(pud,va);
			if(!pmd_none(*pmd) && !pmd_bad(*pmd))
			{
				pte = pte_offset_kernel(pmd,va);
				if(!pte_none(*pte))
				{
					pageFN = pte_pfn(*pte);
					pa = ((pageFN<<12)|(va&0x00000FFF));
					found = 1;
					printk(KERN_ALERT "Physical Address: 0x%08llx\npfn: 0x%04llx\n", pa, pageFN);
				}
			}
		}
	}
	if(pgd_none(*pgd) || pud_none(*pud) || pmd_none(*pmd) || pte_none(*pte))
	{
		unsigned long long swapID = (pte_val(*pte) >> 32);
		found = 1;
		printk(KERN_ALERT "swap ID: 0x%08llx\n", swapID);
	}
	if(found == 0)
	{
		printk(KERN_ALERT "not available\n");
	}
return 0;	
}
/*
 *  @如何编写代码查看自己的进程到底有哪些虚拟区？
 */
static void mtest_dump_vma_list(void)
{
         
	struct mm_struct *mm = task->mm;
	struct vm_area_struct *vma;
        unsigned long count = 0;
        if (!task) {
            pr_info("The process is not exist \n");
            return ;
        }
	printk("The current process is %s\n",task->comm);
	printk("mtest_dump_vma_list\n");
	down_read(&mm->mmap_sem);
	for (vma = mm->mmap;vma; vma = vma->vm_next) {
		printk("VMA 0x%lx-0x%lx ",
				vma->vm_start, vma->vm_end);
		if (vma->vm_flags & VM_WRITE)
			printk("WRITE ");
		if (vma->vm_flags & VM_READ)
			printk("READ ");
		if (vma->vm_flags & VM_EXEC)
			printk("EXEC ");
		printk("\n");
                ++count;
	}
	up_read(&mm->mmap_sem);
        pr_info(" vma count : %lu \n", count);
}

/*
 *  @如果知道某个虚地址，比如，0×8049000,
 *  又如何找到这个地址所在VMA是哪个？
 */
static void  mtest_find_vma(unsigned long addr)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm ;
        if (!task) {
            pr_info("The process is not exist \n");
            return ;
        }
	mm = task->mm;
	printk("mtest_find_vma\n");
	down_read(&mm->mmap_sem);
	vma = find_vma(mm, addr);
	if (vma && addr >= vma->vm_start) {
		printk("found vma 0x%lx-0x%lx flag %lx for addr 0x%lx\n",
				vma->vm_start, vma->vm_end, vma->vm_flags, addr);
	} else {
		printk("no vma found for %lx\n", addr);
	}
	up_read(&mm->mmap_sem);
}
static void  test_cow(void)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm ;
        struct list_head* pos;
        size_t cow_bytes= 0,  vm_bytes = 0;
        struct anon_vma_chain* chain;
        if (!task) {
            pr_info("The process is not exist \n");
            return ;
        }
	mm = task->mm;
	printk("mtest_cow\n");
        
	down_read(&mm->mmap_sem);
          for (vma = mm->mmap; NULL != vma; vma = vma->vm_next)
        {
            list_for_each(pos, &vma->anon_vma_chain)
            {
                chain = list_entry(pos, struct anon_vma_chain, same_vma);
                printk("vma 0x%lx-0x%lx \n", chain->vma->vm_start, chain->vma->vm_end);
                cow_bytes += (chain->vma->vm_end - chain->vma->vm_start);
                         // if there are multiple entries on a chain for a single VM
                         //                 // area, it means the memory area has been passed down though
                         //                                 // multiple processes (more than just the parent). We only
                         //                                                 // want to count such regions once, to avoid over-calculating
                         //                                                                 // the size of the COW regions.
                break;
            }
                 
             vm_bytes += vma->vm_end - vma->vm_start;
        }

	up_read(&mm->mmap_sem);
        pr_info("%d total COW memory bytes %ld, VM bytes %ld\n", task->pid, cow_bytes, vm_bytes);
}
#if 0
static inline void anon_vma_free(struct anon_vma *anon_vma)
{
	VM_BUG_ON(atomic_read(&anon_vma->refcount));

	/*
 *                                                       
 *                                                                                                                   
 *                                                                                                                             
 *                                                                                                                                
 *                                                                                                                                                                                                   
 *                                                                                                                                                                                                                                                            
 *                                                                                                                                                                                                                                                                                                                       
 *                                                                                                                                                                                                                                                                                                                          
 *                                                                                                                                                                                                                                                                                                                                                                     
 *                                                                                                                                                                                                                                                                                                                                                                                                                    
 *                                                                                                                                                                                                                                                                                                                                                                                                                                      
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
 *                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     */
	if (mutex_is_locked(&anon_vma->root->mutex)) {
		anon_vma_lock(anon_vma);
		anon_vma_unlock(anon_vma);
	}

	kmem_cache_free(anon_vma_cachep, anon_vma);
}
void __put_anon_vma(struct anon_vma *anon_vma)
{
	struct anon_vma *root = anon_vma->root;

	if (root != anon_vma && atomic_dec_and_test(&root->refcount))
		anon_vma_free(root);

	anon_vma_free(anon_vma);
}
struct anon_vma *page_lock_anon_vma_read(struct page *page)
{
        struct anon_vma *anon_vma = NULL;
        struct anon_vma *root_anon_vma;
        unsigned long anon_mapping;

        rcu_read_lock();
        anon_mapping = (unsigned long)READ_ONCE(page->mapping);
        if ((anon_mapping & PAGE_MAPPING_FLAGS) != PAGE_MAPPING_ANON)
                goto out;
        if (!page_mapped(page))
                goto out;

        anon_vma = (struct anon_vma *) (anon_mapping - PAGE_MAPPING_ANON);
        root_anon_vma = READ_ONCE(anon_vma->root);
        if (down_read_trylock(&root_anon_vma->rwsem)) {
                /*
 *                  * If the page is still mapped, then this anon_vma is still
 *                                   * its anon_vma, and holding the mutex ensures that it will
 *                                                    * not go away, see anon_vma_free().
 *                                                                     */
                if (!page_mapped(page)) {
                        up_read(&root_anon_vma->rwsem);
                        anon_vma = NULL;
                }
                goto out;
        }

        /* trylock failed, we got to sleep */
        if (!atomic_inc_not_zero(&anon_vma->refcount)) {
                anon_vma = NULL;
                goto out;
        }

        if (!page_mapped(page)) {
                rcu_read_unlock();
                put_anon_vma(anon_vma);
                return NULL;
        }

        /* we pinned the anon_vma, its safe to sleep */
        rcu_read_unlock();
        anon_vma_lock_read(anon_vma);

        if (atomic_dec_and_test(&anon_vma->refcount)) {
                /*
 *                  * Oops, we held the last refcount, release the lock
 *                                   * and bail -- can't simply use put_anon_vma() because
 *                                                    * we'll deadlock on the anon_vma_lock_write() recursion.
 *                                                                     */
                anon_vma_unlock_read(anon_vma);
                __put_anon_vma(anon_vma);
                anon_vma = NULL;
        }

        return anon_vma;

out:
        rcu_read_unlock();
        return anon_vma;
}
#endif
#if 1

static inline unsigned long vma_start_pgoff(struct vm_area_struct *v)
{
        return v->vm_pgoff;
}

static inline unsigned long vma_last_pgoff(struct vm_area_struct *v)
{
        return v->vm_pgoff + ((v->vm_end - v->vm_start) >> PAGE_SHIFT) - 1;
}
static inline unsigned long avc_start_pgoff(struct anon_vma_chain *avc)
{
        return vma_start_pgoff(avc->vma);
}

static inline unsigned long avc_last_pgoff(struct anon_vma_chain *avc)
{
        return vma_last_pgoff(avc->vma);
}
INTERVAL_TREE_DEFINE(struct anon_vma_chain, rb, unsigned long, rb_subtree_last,
                     avc_start_pgoff, avc_last_pgoff,
                     static inline, __anon_vma_interval_tree)
#define __anon_vma_interval_tree_foreach(avc, root, start, last)           \
        for (avc = __anon_vma_interval_tree_iter_first(root, start, last); \
             avc; avc = __anon_vma_interval_tree_iter_next(avc, start, last))
INTERVAL_TREE_DEFINE(struct vm_area_struct, shared.rb,
                     unsigned long, shared.rb_subtree_last,
                     vma_start_pgoff, vma_last_pgoff,, vma_interval_tree)

#define vma_interval_tree_foreach(vma, root, start, last)               \
        for (vma = vma_interval_tree_iter_first(root, start, last);     \
             vma; vma = vma_interval_tree_iter_next(vma, start, last))
#endif
#if 0
void rmap_walk(struct page *page, struct rmap_walk_control *rwc)
{
        if (unlikely(PageKsm(page)))
                rmap_walk_ksm(page, rwc);
        else if (PageAnon(page))
                rmap_walk_anon(page, rwc, false);
        else
                rmap_walk_file(page, rwc, false);
}
#endif
static void test_anon_page(struct page *page)
{
    //struct rmap_walk_control rwc = {
    //    .rmap_one = try_to_unmap_one, //用于解除一页的映射
    //    .arg = (void *)flags,
    //    .done = page_mapcount_is_zero,//获取页的引用计数
    //    .anon_lock = page_lock_anon_vma_read,// 返回匿名页的anon_vma
    //};
     struct anon_vma *anon_vma = NULL;
     //struct anon_vma *anon_vma = vma->anon_vma;
     struct anon_vma_chain *avc;
     struct vm_area_struct *vma;
     unsigned long anon_mapping;
     pgoff_t pgoff_start, pgoff_endm, pgoff;
     if (NULL == page)
     {
         return ;
     }
     if (PageAnon(page)) {
          pr_info("page is anonoyous and compare  rmap_walk_anon \n");
#if 0
          if (page_lock_anon_vma_read(page))
          {
                pr_err("page_lock_anon_vma return not null \n");
           }
          else 
          {
                pr_err("page_lock_anon_vma return  null \n");
           }
#endif
     
          anon_mapping = (unsigned long) page->mapping;
          if (!(anon_mapping & PAGE_MAPPING_ANON))
          {
               pr_info(" not anon_mapping \n");
               return ;
          }
          if (!page_mapped(page))
          {
               pr_info(" not page mapped \n");
               return ;
          }
    
          pgoff= page_to_index(page);
          anon_vma = (struct anon_vma *) (anon_mapping - PAGE_MAPPING_ANON);
          __anon_vma_interval_tree_foreach(avc, &anon_vma->rb_root,  pgoff, pgoff) {
                vma = avc->vma;
                printk("vma 0x%lx-0x%lx flag %lx , vma task comm: %s and pid %d , d_iname : %s \n", vma->vm_start, vma->vm_end, vma->vm_flags, vma->vm_mm->owner->comm, vma->vm_mm->owner->pid, vma->vm_file ? (char *) (vma->vm_file->f_path.dentry->d_iname) : "no vm file");
                if (vma->vm_ops){
                    pr_info("vma->vm_ops: %p \n", vma->vm_ops);
                }
                else {
                    pr_info("vma->vm_ops is null \n");
                }
          }
#if 0
          interval_tree_iter_first(&anon_vma->rb_root,0,0);
          //anon_vma_interval_tree_foreach(avc, &anon_vma->rb_root,
          //              pgoff_start, pgoff_end) {
          //      vma = avc->vma;
          //}
          //if (anon_vma)
          //list_for_each_entry(avc, &anon_vma->head, same_anon_vma) {
          //list_for_each_entry(avc, &vma->anon_vma_chain, same_vma) {
          //    vma = avc->vma;
          //    printk("vma 0x%lx-0x%lx flag %lx , vma task comm: %s \n", vma->vm_start, vma->vm_end, vma->vm_flags, vma->vm_mm->owner->comm);
          //}
          //list_for_each_entry(vma, &anon_vma->head, anon_vma_node) { 
          //}
#endif
          pr_info(" page_mapcount(page) went negative! (%d)\n", page_mapcount(page));
     } 
     else if (PageKsm(page)) {
          pr_info("page is ksm\n");
     }
     //else if (PageMappingFlags(page)) {
     //     pr_info("page is address_space \n");
     //}
     else {
          // rmap_walk_file
        pr_info("page is file and compare rmap_walk_file\n");
        struct address_space *mapping = page_mapping(page);
#if 1
       //struct node mapping->host
       struct address_space_operations *ops=mapping->a_ops;
       if (ops){
           pr_info("mapping->a_ops: %p \n", ops);
       }
#endif
        pgoff= page_to_index(page);
        vma_interval_tree_foreach(vma, &mapping->i_mmap,
                        pgoff, pgoff) {
                printk("vma 0x%lx-0x%lx flag %lx , vma task comm: %s and pid %d , d_iname : %s \n", vma->vm_start, vma->vm_end, vma->vm_flags, vma->vm_mm->owner->comm, vma->vm_mm->owner->pid, vma->vm_file ? (char *) (vma->vm_file->f_path.dentry->d_iname) : "no vm file");
       if (vma->vm_ops){
           pr_info("vma->vm_ops: %p \n", vma->vm_ops);
       }
       }
     }
     pr_info(" page_mapcount(page) went negative! (%d)\n", page_mapcount(page));
     pr_info(" page->flags = %lx\n", page->flags);
     pr_info(" page->count = %x\n", page_count(page));
     pr_info(" page->mapping = %p\n", page->mapping);
     return ;
}
/*
 *  @一个物理页在内核中用struct page来描述。
 *  给定一个虚存区VMA和一个虚地址addr，
 *  找出这个地址所在的物理页面page.
 */
static struct page *
my_follow_page(struct vm_area_struct *vma, unsigned long addr)
{
	pud_t *pud;
	pmd_t *pmd;
	pgd_t *pgd;
	pte_t *pte;
	spinlock_t *ptl;
	struct page *page = NULL;
	struct mm_struct *mm = vma->vm_mm;
        if (!task) {
            pr_info("The process is not exist \n");
            return NULL;
        }
	pgd = pgd_offset(mm, addr);
	if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd))) {
		goto out;
	}
	pud = pud_offset(pgd, addr);
	if (pud_none(*pud) || unlikely(pud_bad(*pud)))
		goto out;
	pmd = pmd_offset(pud, addr);
	if (pmd_none(*pmd) || unlikely(pmd_bad(*pmd))) {
		goto out;
	}
	pte = pte_offset_map_lock(mm, pmd, addr, &ptl);
	if (!pte)
		goto out;
	if (!pte_present(*pte))
		goto unlock;
	page = pfn_to_page(pte_pfn(*pte));
	if (!page)
		goto unlock;
	get_page(page);
unlock:
	pte_unmap_unlock(pte, ptl);
out:
	return page;
}

/*
 *  @ 根据页表，求出某个虚地址所在的物理页面，
 *  这个功能我们简称"findpage"
 */
static void   mtest_find_page(unsigned long addr)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm ;
	unsigned long kernel_addr;
	struct page *page;
        if (!task) {
            pr_info("The process is not exist \n");
            return ;
        }
	mm = task->mm;
	printk("mtest_write_val\n");
	down_read(&mm->mmap_sem);
	vma = find_vma(mm, addr);
	page = my_follow_page(vma, addr);
	if (!page)
	{
		printk("page not found  for 0x%lx\n", addr);
		goto out;
	}
	printk("page  found  for 0x%lx\n", addr);
	kernel_addr = (unsigned long)page_address(page);
	kernel_addr += (addr&~PAGE_MASK);
	printk("find  0x%lx to kernel address 0x%lx\n", addr, kernel_addr);
        show_is_high(page);
        test_anon_page(page);
out:
	up_read(&mm->mmap_sem);
}

/*
 *  @你是否有这样的想法，
 *  给某个地址写入自己所想写的数据？
 */
static void
mtest_write_val(unsigned long addr, unsigned long val)
{
	struct vm_area_struct *vma;
	struct mm_struct *mm = task->mm;
	struct page *page;
	unsigned long kernel_addr;
        if (!task) {
            pr_info("The process is not exist \n");
            return ;
        }
	printk("mtest_write_val\n");
	down_read(&mm->mmap_sem);
	vma = find_vma(mm, addr);
	if (vma && addr >= vma->vm_start && (addr + sizeof(val)) < vma->vm_end) {
		if (!(vma->vm_flags & VM_WRITE)) {
			printk("vma is not writable for 0x%lx\n", addr);
			goto out;
		}
		page = my_follow_page(vma, addr);
		if (!page) {
			printk("page not found  for 0x%lx\n", addr);
			goto out;
		}
		kernel_addr = (unsigned long)page_address(page);
		kernel_addr += (addr&~PAGE_MASK);
		printk("write 0x%lx to address 0x%lx\n", val, kernel_addr);
		*(unsigned long *)kernel_addr = val;
		put_page(page);
	} else {
		printk("no vma found for %lx\n", addr);
	}
out:
	up_read(&mm->mmap_sem);
}

static ssize_t
mtest_write(struct file *file, const char __user * buffer,
		size_t count, loff_t * data)
{
	char buf[128];
	unsigned long val, val2;
        int pid;
	printk("mtest_write  ………..  \n");
	if (count > sizeof(buf))
		return -EINVAL;
	if (copy_from_user(buf, buffer, count))
		return -EINVAL;
	if (memcmp(buf, "findtask", 8) == 0) {
		if (sscanf(buf + 8, "%d", &pid) == 1) {
			find_task(pid);
		}
        }
	else if (memcmp(buf, "listvma", 7) == 0)
		mtest_dump_vma_list();
	else if (memcmp(buf, "countcow", 8) == 0)
		test_cow();
	else if (memcmp(buf, "va2phy", 6) == 0) {
		if (sscanf(buf + 6, "%lx", &val) == 1) {
			va_to_phy(val);
		}
	}
	else if (memcmp(buf, "findvma", 7) == 0) {
		if (sscanf(buf + 7, "%lx", &val) == 1) {
			mtest_find_vma(val);
		}
	}
	else if (memcmp(buf, "findpage", 8) == 0) {
		if (sscanf(buf + 8, "%lx", &val) == 1) {
			mtest_find_page(val);
			//my_follow_page(vma, addr);
		}
	}
	else  if (memcmp(buf, "writeval", 8) == 0) {
		if (sscanf(buf + 8, "%lx %lx", &val, &val2) == 2) {
			mtest_write_val(val, val2);
		}
	}
	return count;
}

static struct
file_operations proc_mtest_operations = {
	.write        = mtest_write
};


static struct proc_dir_entry *mtest_proc_entry;
//整个操作我们以模块的形式实现，因此，模块的初始化和退出函数如下：
static int __init
mtest_init(void)
{
        
        mtest_proc_entry = proc_create("mtest", 0777, NULL, &proc_mtest_operations);
	printk("create the filename mtest mtest_init sucess  \n");
	return 0;
}

static void
__exit mtest_exit(void)
{
	printk("exit the module……mtest_exit \n");
	remove_proc_entry("mtest", NULL);
}

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("mtest");
MODULE_AUTHOR("Zou Nan hai");
module_init(mtest_init);
module_exit(mtest_exit);
