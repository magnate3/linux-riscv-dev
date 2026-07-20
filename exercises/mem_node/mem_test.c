#define pr_fmt(fmt) KBUILD_MODNAME ": %s :"fmt,__func__

#include <linux/stddef.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/swap.h>
#include <linux/interrupt.h>
#include <linux/pagemap.h>
#include <linux/jiffies.h>
#include <linux/memblock.h>
#include <linux/compiler.h>
#include <linux/kernel.h>
#include <linux/kasan.h>
#include <linux/module.h>
#include <linux/suspend.h>
#include <linux/pagevec.h>
#include <linux/blkdev.h>
#include <linux/slab.h>
#include <linux/ratelimit.h>
#include <linux/oom.h>
#include <linux/topology.h>
#include <linux/sysctl.h>
#include <linux/cpu.h>
#include <linux/cpuset.h>
#include <linux/memory_hotplug.h>
#include <linux/nodemask.h>
#include <linux/vmalloc.h>
#include <linux/vmstat.h>
#include <linux/mempolicy.h>
#include <linux/memremap.h>
#include <linux/stop_machine.h>
#include <linux/sort.h>
#include <linux/pfn.h>
#include <linux/backing-dev.h>
#include <linux/fault-inject.h>
#include <linux/page-isolation.h>
#include <linux/page_ext.h>
#include <linux/debugobjects.h>
#include <linux/kmemleak.h>
#include <linux/compaction.h>
#include <trace/events/kmem.h>
#include <trace/events/oom.h>
#include <linux/prefetch.h>
#include <linux/mm_inline.h>
#include <linux/migrate.h>
#include <linux/hugetlb.h>
#include <linux/sched/rt.h>
#include <linux/sched/mm.h>
#include <linux/page_owner.h>
#include <linux/kthread.h>
#include <linux/memcontrol.h>
#include <linux/ftrace.h>
#include <linux/lockdep.h>
#include <linux/nmi.h>
//#include <linux/psi.h>
#include <linux/timer.h>
#include <linux/time.h>
#include <asm/sections.h>
#include <asm/tlbflush.h>
#include <asm/div64.h>
#include <linux/init.h>         /* For init/exit macros */
#include <linux/module.h>       /* For MODULE_ marcros  */
#include <linux/delay.h>

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/init.h>
#include <linux/fs.h>
#include <linux/string.h>
#include <linux/mm.h>
#include <linux/syscalls.h>
#include <asm/unistd.h>
#include <asm/uaccess.h>
#include <linux/init.h>  
#include <linux/kernel.h>  
#include <linux/module.h>  
#include <linux/fs.h>  
#include <linux/types.h>  
#include <linux/errno.h>  
#include <linux/fcntl.h>  
#include <linux/vmalloc.h>  
//#include <asm/uaccess.h>  
#include <asm/io.h>  
#include <asm/page.h>  
#include <linux/mm.h>  
#include <linux/platform_device.h>  
#include <linux/device.h>  
#include <linux/moduleparam.h>  
#include <linux/cdev.h>  
#include <linux/slab.h> 
#include <linux/pci.h>
#include <linux/dma-mapping.h>
#include <linux/dmapool.h>
#include <linux/device.h>

extern void print_node(void);
//extern void set_flag_main(int a);

void print_order(int order)
{
	int i = 0,j=0;
	struct page * page= NULL;
   unsigned long flags;
   unsigned long pfn = 0;
	struct zone* pzone = NULL;
	struct pglist_data * pgdat = NODE_DATA(0);
	unsigned int counter=0;
	
	for(i=0;i<pgdat->nr_zones;i++)
	{
		pzone = &pgdat->node_zones[i];
		spin_lock_irqsave(&pzone->lock, flags);
		
      for(j=0;j<MIGRATE_TYPES;j++)
      {
         counter = 0;
         if(!list_empty(&pzone->free_area[order].free_list[j]))
         {
            list_for_each_entry(page,&pzone->free_area[order].free_list[j], lru) 
            {
               pfn = page_to_pfn(page);
               counter ++ ;
               //pr_info("order %d %15s[%d] counter %d, pfn %lx\n",order,migratetype_names[j],j,counter,pfn);
               if(10 == order)
                  break;
            }
         }
         else
         {
            pfn = 0;
            //pr_info("order %d %15s[%d] counter %d, pfn %lx\n",order,migratetype_names[j],j,counter,pfn);
         }

      }
		spin_unlock_irqrestore(&pzone->lock, flags);
	}
}
void print_orders(int order1,int order2)
{
   int i = 0;
   for(i=order1;i<=order2;i++)
   {
      print_order(i);
      pr_info("\n");
   }
}

extern struct lruvec *get_lruvec(void);
static void print_lruvec(struct lruvec * lruvec)
{
	int i = 0;
	struct pglist_data * pgdat = NODE_DATA(0);
   struct page * page= NULL;
	unsigned int counter=0;
   unsigned long pfn = 0;
	struct lruvec* plru = lruvec;//get_lruvec();//&(NODE_DATA(0)->lruvec);
   unsigned long flags = 0;
   const static char* lru_str[NR_LRU_LISTS] = {
   "INACTIVE_ANON" ,
	"ACTIVE_ANON" ,
	"INACTIVE_FILE" ,
	"ACTIVE_FILE" ,
	"UNEVICTABLE",
};

	spin_lock_irqsave(&pgdat->lru_lock, flags);	
	for(i=0;i<NR_LRU_LISTS;i++)
	{
      counter = 0;
      if(!list_empty(&plru->lists[i]))
      {
         list_for_each_entry(page,&plru->lists[i],lru) 
         {
            pfn = page_to_pfn(page);
            counter ++ ;
            pr_info("%15s[%d] counter %d, pfn %lX\n",lru_str[i],i,counter,pfn);
         }
      }
      else
         pr_info("%15s[%d] counter %d\n",lru_str[i],i,counter);
	}
   spin_unlock_irqrestore(&pgdat->lru_lock, flags);
}

void alloc_lruvec_pages(void)
{
   int i = 0;
   struct page *page = NULL;
   page = alloc_pages(GFP_KERNEL, 6);
   
   for(i=0;i<64;i++)
	{
      get_page(page);
      //__SetPageSwapBacked(page);
      SetPageActive(page);
		//lru_cache_add(page);
		page++;
	}
   // char* p = NULL;
   // p = kmalloc(GFP_KERNEL,PAGE_SIZE * 64);
}
void set_watermark(void)
{
	int i = 0;
   unsigned long flags;
	struct zone* pzone = NULL;
	struct pglist_data * pgdat = NODE_DATA(0);
	
	for(i=0;i<pgdat->nr_zones;i++)
	{
		pzone = &pgdat->node_zones[i];
		spin_lock_irqsave(&pzone->lock, flags);
#if 0
      pr_info("min %ld, low %ld, high %ld\n",pzone->_watermark[WMARK_MIN], pzone->_watermark[WMARK_LOW],pzone->_watermark[WMARK_HIGH]);
      pzone->_watermark[WMARK_MIN] = 120000;
      pzone->_watermark[WMARK_LOW] = 150000;
      pzone->_watermark[WMARK_HIGH]= 180000;
      pr_info("min %ld, low %ld, high %ld\n",pzone->_watermark[WMARK_MIN], pzone->_watermark[WMARK_LOW],pzone->_watermark[WMARK_HIGH]);
		spin_unlock_irqrestore(&pzone->lock, flags);
#endif
	}
}
static unsigned long power(int x, int n)
{
	int i;
	int s=1;
	for(i=1;i<=n;i++ )
		s*=x;
	return s;
}
void alloc_prepare(void)
{
	int i = 0,j=0,k=0;
	struct page * page= NULL;
	unsigned long pages = 0;
	unsigned long total_pages = 0;
	 unsigned long flags;
	struct zone* pzone = NULL;
	struct pglist_data * pgdat = NODE_DATA(0);
	unsigned int counter[MAX_ORDER][MIGRATE_TYPES];

	for(k=0;k<pgdat->nr_zones;k++)
	{
		pzone = &pgdat->node_zones[k];
		spin_lock_irqsave(&pzone->lock, flags);
	
      for(i=0;i<MAX_ORDER;i++)//MAX_ORDER
      {
         pages = power(2,i) * pzone->free_area[i].nr_free;
         total_pages += pages;
         for(j=0;j<MIGRATE_TYPES;j++)
         {
            counter[i][j] = 0;
            if(!list_empty(&pzone->free_area[i].free_list[j]))
            {
               list_for_each_entry(page,&pzone->free_area[i].free_list[j], lru) 
                  counter[i][j] ++ ;
            }
         }
#if 0
         pr_info("order:%02d:---- nr_free:%3ld, pages %8ld, total pages %8ld --- order:%02d: %s %3d, %s %3d, %s %3d, %s %3d, %s %3d,%s %3d\n",
            i,pzone->free_area[i].nr_free,pages,total_pages,i,
            migratetype_names[0],counter[i][0],
            migratetype_names[1],counter[i][1],
            migratetype_names[2],counter[i][2],
            migratetype_names[3],counter[i][3],
            migratetype_names[4],counter[i][4],
            migratetype_names[5],counter[i][5]);
#endif
      }
		spin_unlock_irqrestore(&pzone->lock, flags);
	}
   for(i=0;i<152;i++)
   {
      page =  alloc_pages(GFP_KERNEL, 10);
     // memset(page_address(page),0x12,1024*4*1024);
   }
   page = alloc_pages(GFP_KERNEL, 9);
  // memset(page_address(page),0x12,512*4*1024);
   page = alloc_pages(GFP_KERNEL, 7);
 //  memset(page_address(page),0x12,128*4*1024);
   page = alloc_pages(GFP_KERNEL, 5);
  // memset(page_address(page),0x12,32*4*1024);
   page = alloc_pages(GFP_KERNEL, 5);
  // memset(page_address(page),0x12,32*4*1024);
   page = alloc_pages(GFP_KERNEL, 4);
 //  memset(page_address(page),0x12,16*4*1024);
}
extern void setup_per_zone_wmarks(void);

int __init mem_test(void)
{
   // int i=0;
   // struct page *page1 = NULL;
   // struct page *page2 = NULL;
   // struct page *page3 = NULL;
   int* p = NULL;
   pr_info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> test +++++++++++++++++++++++++++++++\n");
   // alloc_prepare();
   // print_node();
   alloc_lruvec_pages();
   print_lruvec(get_lruvec());
   // print_node();
    //set_flag_main(1);

   // print_orders(7,10);
   // page1 = alloc_pages(GFP_KERNEL, 7);
   // pr_info("I get pfn pfn1 %llx\n",(u64)page_to_pfn(page1));

   // print_orders(7,10);
   // page2 = alloc_pages(GFP_KERNEL, 7);
   // pr_info("I get pfn pfn2 %llx\n",(u64)page_to_pfn(page2));

   // print_orders(7,10);
   // page3 = alloc_pages(GFP_KERNEL, 7);
   // pr_info("I get pfn pfn3 %llx\n",(u64)page_to_pfn(page3));
   // pr_info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> free pages +++++++++++++++++++++++++++++++\n");
   // print_orders(7,10);
   // __free_pages(page1,7);

   // print_orders(7,10);
   // __free_pages(page2,7);

   // print_orders(7,10);
   // __free_pages(page3,7);

   // print_orders(7,10);

   // print_node();

   p = (int*)kmalloc(100,GFP_KERNEL);

   //set_flag_main(0);
	return 0;
}

module_init(mem_test);
MODULE_LICENSE("Dual BSD/GPL");
