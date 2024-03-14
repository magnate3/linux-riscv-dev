#include <linux/kobject.h>
#include <linux/string.h>
#include <linux/sysfs.h>
#include <linux/module.h>
#include <linux/init.h>
/* /sys/devices directory */
extern struct kset *devices_kset;

static int demo_init(void)
{
    struct kobject *test_kobject;
    struct kobject *kobj;

    pr_info("Demo Procedure Entence...\n");
#if 0
    /* Create a kobject and parent is "devices" */
    test_kobject = kobject_create_and_add("",
                                                  &devices_kset->kobj);
    if (!test_kobject) {
        pr_info("Unable to create and add kobject.\n");
        return -EINVAL;
    }

    /* kset's parent */
    if (test_kobject->parent)
        pr_info("%s parent: %s\n", test_kobject->name,
                                  test_kobject->parent->name);

    pr_info("kobject: state_in_sysfs:    %d\n",
                          test_kobject->state_in_sysfs);
    pr_info("kobject: state_initialized: %d\n",
                          test_kobject->state_initialized);
#endif
    /* Traverse all kobject on /sys/devices/ */
    list_for_each_entry(kobj, &devices_kset->list, entry) {
        if (kobj->name)
            pr_info("%s\n", kobj->name);
    }

    return 0;
}
static void demo_exit(void)
{
}
module_init(demo_init);
module_exit(demo_exit);
MODULE_LICENSE("GPL");
