#include <linux/init.h>
#include <linux/module.h>
#include <linux/mmzone.h>

static int __init test_init(void)
{
    int         nid = 0;
    struct zone *zone;
    struct pglist_data *pgdata = NODE_DATA(nid);
    int zidx, fidx;
    struct free_area *farea;
    printk("NODES_SHIFT = %d, MAX_NUMNODES = %d\n", NODES_SHIFT, MAX_NUMNODES);
    for (zidx = 0; zidx < pgdata->nr_zones; zidx++) {
        zone = &pgdata->node_zones[zidx];
        printk(KERN_ALERT "zone->name: %s\n", zone->name);
        for (fidx = 0; fidx < MAX_ORDER; fidx++) {
            farea = &zone->free_area[fidx];
            printk(KERN_ALERT "\t[%d] nr_free: %ld\n", fidx, farea->nr_free);
        }
    }

    return 0;
}

static void __exit test_exit(void)
{
}

module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");
