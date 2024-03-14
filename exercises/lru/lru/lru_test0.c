//#include <linux/kernel.h>
//#include <linux/syscalls.h>
//#include <linux/mm.h>
//#include <linux/rmap.h>
//#include <linux/list.h>
//#include <linux/cpuset.h>
//#include <linux/percpu-defs.h>
//#include <linux/vm_event_item.h>
//#include <linux/memcontrol.h>
//#include <linux/swap.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/list.h>
#include <linux/cpuset.h>
#include <linux/memcontrol.h>
#include <linux/vm_event_item.h>
#include <linux/mmzone.h>
#define prev_page(p)	list_entry((p)->lru.prev, struct page, lru)
#define lru_to_page(head) (list_entry((head)->prev, struct page, lru))
//extern struct mem_cgroup *root_mem_cgroup;
static struct mem_cgroup *get_mem_cgroup_from_mm(struct mm_struct *mm)
{
        struct mem_cgroup *memcg = NULL;

        rcu_read_lock();
        do {
                /*
 *                  * Page cache insertions can happen withou an
 *                                   * actual mm context, e.g. during disk probing
 *                                                    * on boot, loopback IO, acct() writes etc.
 *                                                                     */
                if (unlikely(!mm))
                {
                        //memcg = root_mem_cgroup;
                }
                else {
                        memcg = mem_cgroup_from_task(rcu_dereference(mm->owner));
                        //if (unlikely(!memcg))
                        //        memcg = root_mem_cgroup;
                }
        } while (!css_tryget_online(&memcg->css));
        rcu_read_unlock();
        return memcg;
}
void show_pfn(struct list_head *src)
{
	int page_count = 0;
	struct page *p = NULL;

	while (!list_empty(src))
	{
		if (++page_count >= 20)
			break;

		p = lru_to_page(src);
		printk(KERN_CONT "(%lx) ", page_to_pfn(p));
		p = prev_page(p);
	}
}

void show_list(void)
{
	struct pglist_data *current_pglist = NULL;
	struct lruvec *lruvec = NULL;
	struct mem_cgroup *memcg = NULL;
	struct mem_cgroup_per_node *mz;
	int i;

	for (i = 0; i < MAX_NUMNODES; i++) {
		if (NODE_DATA(i) == NULL)
			continue;

		current_pglist = NODE_DATA(i);

		if (current_pglist->node_present_pages == 0) {
			printk(KERN_ALERT "Node-%d does not have any pages.\n", i);
			continue;
		}

		spin_lock_irq(&current_pglist->lru_lock);

#if 1
		memcg = get_mem_cgroup_from_mm(current->mm);
		//memcg = get_mem_cgroup_from_mm(NULL);
		mz = mem_cgroup_nodeinfo(memcg, current_pglist->node_id);
		lruvec = &mz->lruvec;
                if (!lruvec)
                {
		     printk("========== lruvec is null ============\n");
                }
#else
		lruvec = mem_cgroup_lruvec(memcg, current_pglist);
#endif

#if 1
		printk("========== LRU_ACTIVE_FILE ============\n");
		show_pfn(&lruvec->lists[LRU_ACTIVE_FILE]);
		printk("========== LRU_INACTIVE_FILE ============\n");
		show_pfn(&lruvec->lists[LRU_INACTIVE_FILE]);
		printk("========== LRU_ACTIVE_ANON ============\n");
		show_pfn(&lruvec->lists[LRU_ACTIVE_ANON]);
		printk("========== LRU_INACTIVE_ANON ============\n");
		show_pfn(&lruvec->lists[LRU_INACTIVE_ANON]);
		printk("========== LRU_UNEVICTABLE ============\n");
		show_pfn(&lruvec->lists[LRU_UNEVICTABLE]);
#endif		
		spin_unlock_irq(&current_pglist->lru_lock);
	}
}
static int __init show_lru_init(void)
{
	show_list();
	return 0;
}

static void __exit show_lru_exit(void)
{
}

module_init(show_lru_init);
module_exit(show_lru_exit);
MODULE_LICENSE("GPL");
