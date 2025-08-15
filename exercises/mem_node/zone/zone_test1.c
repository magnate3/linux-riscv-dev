#include <linux/init.h>
#include <linux/module.h>
#include <linux/mmzone.h>

static int __init test_init(void)
{
    struct zoneref *z;
    int idx;
    enum zone_type highidx = __MAX_NR_ZONES - 1;
    int nid = numa_node_id();
    gfp_t gfp_mask = 0;
    struct zonelist *zlist = node_zonelist(nid, gfp_mask);

    for (idx = 0; idx < highidx; idx++) {
        z = &zlist->_zonerefs[idx];
        printk(KERN_ALERT "z->zone->name: %s\n", z->zone->name);
    }

    return 0;
}

static void __exit test_exit(void)
{
}

module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");