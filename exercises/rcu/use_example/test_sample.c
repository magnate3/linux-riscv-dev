/* Reference:
 * https://wiki.bit-hive.com/north/pg/rcu%28READ%20COPY%20UPDATE%29
 */
#include <linux/module.h>
#include <linux/swap.h>

struct hoge {
	struct rcu_head hoge_rcu;
	struct hoge *down;
	char name[6];
	struct hoge *rcu_tmp;
};

static struct hoge a;
static struct hoge b;
static struct hoge c;

static void print_hoge(char *msg, struct hoge *a)
{
	struct hoge *tmp;

	tmp = a->down;
	pr_info("%s:%s\n", msg, tmp->name);
}

static void hoge_rcu_callback(struct rcu_head *head)
{
	struct hoge *hoge = container_of(head, struct hoge, hoge_rcu);

	hoge->down = hoge->rcu_tmp;
}

static int __init init_sample_(void)
{
	strcpy(a.name, "foo1");
	strcpy(b.name, "foo2");
	a.down = &b;

	rcu_read_lock();
	memcpy((char *)&c, (char *)a.down, sizeof(struct hoge));
	strcpy(c.name, "hoge3");
	a.rcu_tmp = &c;
	rcu_read_unlock();

	print_hoge("before call rcu", &a);
	call_rcu(&a.hoge_rcu, hoge_rcu_callback);
	schedule_timeout_interruptible(100);
	print_hoge("after call rcu", &a);

	return 0;
}

static void __exit exit_sample_(void)
{
}

MODULE_AUTHOR("Fumiya Shigemitsu");
MODULE_DESCRIPTION("sample: -");
MODULE_LICENSE("GPL");

module_init(init_sample_)
module_exit(exit_sample_)
