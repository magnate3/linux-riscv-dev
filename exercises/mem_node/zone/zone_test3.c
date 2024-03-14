#include <linux/init.h>
#include <linux/module.h>
#include <linux/mmzone.h>
#include <linux/mm.h> //  list_for_each_entry(page
#define DEBUG_PRINT(...) printk(KERN_INFO __VA_ARGS__)
#define DEVICE_NAME ""
static int GetElementNumberOfList(struct list_head *head)
{
	int count;
	struct list_head *pos;

	count = 0;
	list_for_each(pos, head)
	{
		count++;
	}

	return count;
}
void mark_free_pages(struct zone *zone)
{
        unsigned long pfn, max_zone_pfn, page_count ;
        unsigned long flags;
        unsigned int order, t;
        struct page *page;

        if (zone_is_empty(zone))
                return;

        spin_lock_irqsave(&zone->lock, flags);

#if 0
        max_zone_pfn = zone_end_pfn(zone);
        for (pfn = zone->zone_start_pfn; pfn < max_zone_pfn; pfn++)
                if (pfn_valid(pfn)) {
                        page = pfn_to_page(pfn);

                }

#endif
        for_each_migratetype_order(order, t) {
                list_for_each_entry(page,
                                &zone->free_area[order].free_list[t], lru) {
                       //判读页是否在伙伴系统中，page->_mapcount = PAGE_BUDDY_MAPCOUNT_VALUE
                       if(!PageBuddy(page))
                       {
                                pr_info("zone name %s, page %p is not in buddy \n",zone->name, page);
                       }
                }
        }
        spin_unlock_irqrestore(&zone->lock, flags);
}
static int __init test_init(void)
{
	struct pglist_data *pNode;
	struct zone *pZone;
	int i, j, k;
#if 0
        unsigned int order, t, flc;
       	unsigned long start_pfn, end_pfn;
	struct page *start_page, *end_page;
        struct page *page;
#endif
	for(i = 0; i < MAX_NUMNODES; i++)
	{
		pNode = node_data[i];
		if(pNode == 0)
			continue;

		for(j = 0; j < MAX_NR_ZONES; j++)
		{
			pZone = &(pNode->node_zones[j]);
			if(pZone->managed_pages == 0)
				continue;

			DEBUG_PRINT(DEVICE_NAME " ************************ ZONE NAME %s\n", pZone->name);
                        mark_free_pages(pZone);
#if 0
                        for (flc = 0; flc < FREE_AREA_COUNTS; flc++)
                            pr_info("[%d]: label = %lu, segment = %lu\n", flc, pZone->zone_label[flc].label, pZone->zone_label[flc].segment);
#endif
			for(k = 0; k < MAX_ORDER; k++)
			{
				DEBUG_PRINT(DEVICE_NAME " ZONE NAME %s, ZONE %d free_area[%d].nr_free = 0x%lx\n", pZone->name, j, k, pZone->free_area[k].nr_free);

				DEBUG_PRINT(DEVICE_NAME " ZONE %d free_area[%d].free_list[Unmovable] = 0x%d\n", j, k, GetElementNumberOfList(&(pZone->free_area[k].free_list[MIGRATE_UNMOVABLE])));
				DEBUG_PRINT(DEVICE_NAME " ZONE %d free_area[%d].free_list[Reclaimable] = 0x%d\n", j, k, GetElementNumberOfList(&(pZone->free_area[k].free_list[MIGRATE_RECLAIMABLE])));
				DEBUG_PRINT(DEVICE_NAME " ZONE %d free_area[%d].free_list[Movable] = 0x%d\n", j, k, GetElementNumberOfList(&(pZone->free_area[k].free_list[MIGRATE_MOVABLE])));
				//DEBUG_PRINT(DEVICE_NAME " ZONE %d free_area[%d].free_list[Reserve] = 0x%d\n", j, k, GetElementNumberOfList(&(pZone->free_area[k].free_list[MIGRATE_RESERVE])));
				//DEBUG_PRINT(DEVICE_NAME " ZONE %d free_area[%d].free_list[CMA] = 0x%d\n", j, k, GetElementNumberOfList(&(pZone->free_area[k].free_list[MIGRATE_CMA])));
				DEBUG_PRINT(DEVICE_NAME " ZONE %d free_area[%d].free_list[Isolate] = 0x%d\n", j, k, GetElementNumberOfList(&(pZone->free_area[k].free_list[MIGRATE_ISOLATE])));
#if 0
                                pr_info("*************** MIGRATE_MOVABLE page_to_pfn %\n ");
                                //page = list_entry(pZone->free_area[k].free_list[MIGRATE_MOVABLE].next, struct page, lru);
                                list_for_each_entry(page, &pZone->free_area[k].free_list[MIGRATE_MOVABLE], lru) {
                                }
                                //start_pfn = page_to_pfn(page);
	                        //start_pfn = start_pfn & ~(pageblock_nr_pages-1);
	                        //start_page = pfn_to_page(start_pfn);
	                        //end_page = start_page + pageblock_nr_pages - 1;
	                        //end_pfn = start_pfn + pageblock_nr_pages - 1;
                                pr_info("\n");
#endif
			}
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
