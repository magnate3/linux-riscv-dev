#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/kthread.h>
#include <linux/wait.h>
#include <linux/slab.h>

struct foo {
    int a;
    char b;
    long c;
};
DEFINE_SPINLOCK(foo_mutex);

static struct foo *gbl_foo = NULL;

/*
 * Create a new struct foo that is the same as the one currently
 * pointed to by gbl_foo, except that field "a" is replaced
 * with "new_a".  Points gbl_foo to the new structure, and
 * frees up the old structure after a grace period.
 *
 * Uses rcu_assign_pointer() to ensure that concurrent readers
 * see the initialized version of the new structure.
 *
 * Uses synchronize_rcu() to ensure that any readers that might
 * have references to the old structure complete before freeing
 * the old structure.
 */
static void foo_update_a(int new_a, int new_b, int new_c)
{

    struct foo *new_fp;
    struct foo *old_fp;


    spin_lock(&foo_mutex);
    new_fp = (struct foo*)kmalloc(sizeof(struct foo), GFP_KERNEL);
    old_fp = gbl_foo;
    new_fp->a = new_a;
    new_fp->b = new_b;
    new_fp->c = new_c;
    rcu_assign_pointer(gbl_foo, new_fp);
    spin_unlock(&foo_mutex);

    synchronize_rcu();

    if(likely(old_fp)){
        kfree(old_fp);
        old_fp = NULL;
        printk("kfree(old_fp) \n");
    }
}

/*
 * Return the value of field "a" of the current gbl_foo
 * structure.  Use rcu_read_lock() and rcu_read_unlock()
 * to ensure that the structure does not get deleted out
 * from under us, and use rcu_dereference() to ensure that
 * we see the initialized version of the structure (important
 * for DEC Alpha and for people reading the code).
 */
static int foo_get_a(void)
{
    int retval;

    rcu_read_lock();
    rcu_read_lock();
    retval = rcu_dereference(gbl_foo)->a;
    rcu_read_unlock();
    rcu_read_unlock();
    return retval;
}



static int __init kernel_init(void)
{
    printk("Module starting ... ... ...\n");

    foo_update_a(10, 11, 12);
    printk("gbl_foo->a: %d \n", foo_get_a());

    foo_update_a(11, 12, 13);
    printk("gbl_foo->a: %d \n", foo_get_a());

    return 0;
}

static void __exit kernel_fini(void)
{
    if(likely(gbl_foo)){
        kfree(gbl_foo);
        gbl_foo = NULL;
        printk("kfree(gbl_foo) \n");
    }

    printk("Module terminating ... ... ...\n");
}

module_init(kernel_init);
module_exit(kernel_fini);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("NINHLD");
MODULE_VERSION("1.0.0");
