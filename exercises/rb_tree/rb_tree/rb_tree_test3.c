#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/rbtree.h>
#include <linux/slab.h>
#include <linux/random.h>
#include <asm/timex.h>
#include <linux/ktime.h>
#include <linux/sched.h>
#include <linux/types.h>
#include <linux/unistd.h>

struct task_struct *result;

struct my_node
{
	int value;
	struct rb_node rb;
};

static struct rb_root_cached rbtree_10_root = RB_ROOT_CACHED;

static struct my_node *rbtree_10 = NULL;

static struct rnd_state rnd;

void insert(struct my_node *node, struct rb_root_cached *root)
{
	struct rb_node **new = &root->rb_root.rb_node, *parent = NULL;
	int value = node->value;

	while (*new) {
		parent = *new;
		if (value < rb_entry(parent, struct my_node, rb)-> value) {
			new = &parent->rb_left;
		}
		else {
			new = &parent->rb_right;
		}
	}

	rb_link_node(&node->rb, parent, new);
	rb_insert_color(&node->rb, &root->rb_root);
}

static inline void erase(struct my_node *node, struct rb_root_cached *root)
{
	rb_erase(&node->rb, &root->rb_root);
}

static void init(void)
{
	int i;
	for (i = 0; i < 10; i++) {
			rbtree_10[i].value = prandom_u32_state(&rnd);
	}	
}

void RB_example(void)
{	
	int i;
	struct rb_node *node;

	rbtree_10 = kmalloc_array(10, sizeof(*rbtree_10), GFP_KERNEL);

	prandom_seed_state(&rnd, 3141592653589793238ULL);
	init();

	for (i = 0; i < 10; i++) {
		insert(rbtree_10 + i, &rbtree_10_root);
	}
	
	for (node = rb_first(&rbtree_10_root.rb_root); node; node = rb_next(node))
        {
            pr_info("value:  %d  \n ",  rb_entry(node, struct my_node, rb)-> value);
        }
	for (i = 0; i < 10; i++) {
		erase(rbtree_10 + i, &rbtree_10_root);
	}

        kfree(rbtree_10);
}

int __init rbtree_module_init(void)
{
	//nice(-20);
	result = pid_task(find_vpid((int) task_pid_nr(current)), PIDTYPE_PID);

	printk("\n********** rbtree_fifo testing!! **********\n");

	printk("rt_priority: %d\n", result->rt_priority);
	printk("scheduling policy: %d\n", result->policy);
	printk("first vruntime: %lld\n", result->se.vruntime);

	RB_example();

	printk("second vruntime: %lld\n\n", result->se.vruntime);

	return 0;
}

void __exit rbtree_module_cleanup(void)
{
	printk("\nBye module\n");
}

module_init(rbtree_module_init);
module_exit(rbtree_module_cleanup);
MODULE_LICENSE("GPL");
