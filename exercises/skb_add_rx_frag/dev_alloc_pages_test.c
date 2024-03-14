/********************************************************************************
* File Name:	page_zone.c
* Description:	第6章实例训练          
* Reference book:《Linux内核API完全参考手册》邱铁，周玉，邓莹莹 ，机械工业出版社.2010.9  
* E_mail:openlinux2100@gmail.com			                
*
********************************************************************************/

#include <linux/init.h> 
#include <linux/module.h> 
#include <linux/mm.h> 
#include <linux/mmzone.h> 
#include <linux/skbuff.h> 

MODULE_LICENSE("GPL"); 
static int __init page_zone_init(void); 
static void __exit page_zone_exit(void); 


int  test_page_zone(void) 
{	 
        struct page * page = NULL;
	page = dev_alloc_pages(0); 	 //分配一个物理页
	if(!page) 
	{ 
		printk("<0>alloc_pages failed!\n"); 
		return -ENOMEM; 
	} 
	else 
	{ 
		printk("<0>alloc_pages Successfully!\n"); 
		struct zone * zone = NULL; 
		zone = page_zone( page );           //获取物理页page所属的区描述符
		if(!zone) 
		{ 
			printk("<0>page_zone wrong!\n");			 
		} 
		else 
		{ 
			if( ZONE_NORMAL==  zone_idx(zone) )         //判断是否属于ZONE_NORMAL
				printk("<0>the zone is NORMAL.\n"); 
			if(  ZONE_DMA ==  zone_idx(zone))           //判断是否属于ZONE_DMA
				printk("<0>the zone is DMA.\n"); 
			//if( ZONE_MOVABLE == zone_idx(zone) || ZONE_HIGHMEM ==  zone_idx(zone))       //判断是否属于ZONE_HIGHMEM
			if( ZONE_MOVABLE == zone_idx(zone))       //判断是否属于ZONE_HIGHMEM
				printk("<0>the zone is HIGHMEM.\n"); 
		} 		 
                 __free_pages(page,0);
	}	 
	return 0; 
}
int __init page_zone_init(void) 
{
	test_page_zone();
        return 0;
}

void __exit page_zone_exit(void) 
{ 
	//if(page) 
	//	__free_pages(page,0);  //释放所分配的物理页
	printk("<0>exit!\n"); 
}

module_init(page_zone_init); 
module_exit(page_zone_exit);
