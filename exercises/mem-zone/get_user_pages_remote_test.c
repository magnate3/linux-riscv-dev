/*
linux/mm.h
mm/gup.c
long get_user_pages_remote(struct task_struct *tsk, struct mm_struct *mm,
		unsigned long start, unsigned long nr_pages,
		unsigned int gup_flags, struct page **pages,
		struct vm_area_struct **vmas, int *locked)
 sfw**  get_user_pages_remote从内核远程持有用户空间的页，get操作必须对应put操作。
 文件名gup--get_user_page。
 get_user_pages_remote() - 把用户空间页固定在内存中（阻止swap到硬盘交换区）。
 @tsk：用作page的默认引用计数的task_struct，如果默认不用引用记录用NULL。
 @mm： 目标mm的mm_struct(有一串struct vm_area_struct组成)。
 @start：用户空间起始地址（目标）。
 @nr_pages: 从start起要固定的页数。
 @gup_flags: 修饰查找方式的标记。
 @pages：指向被固定的页的指针的数组。最少应该 nr_pgaes 长。如果调用者只是想确认页
 	已经被固定可以是NULL。（输出参数）
 @vmas: 关联到页的vmas的指针数组。如果调用者不需要可以是NULL。（输出参数）
 @locked：指向锁标记，表示锁是否被持有以及VM_FAULT_RETRY功能是否能用。锁必须被初始化为持有。
*/

#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/mm_types.h>

static int __init get_user_pages_remote_init(void){
	
	struct page *pages_arry[100];
	struct page **pages = pages_arry;

	struct page *pages1 = NULL;
	
	struct mm_struct *mm = NULL;
	unsigned long addr = 0;
	int len = 0;
	unsigned int gup_flags = 0;
	int locked = 1;
	int ret = 0;
        //struct zone *zone;
        //long pfn;	
	mm = current->mm;
	addr = mm->mmap->vm_start;
	len = vma_pages(mm->mmap);
	gup_flags = 0;
	
	printk("vma_pages(mm->pages) = %d\n",len);
	printk("address of pages is: 0x%lx\n",(unsigned long)pages);
	printk("address of pages1 is: 0x%lx\n",(unsigned long)pages1);
	
	down_read(&current->mm->mmap_sem);
	ret = get_user_pages_remote(current,mm,addr,len,gup_flags,pages,NULL,&locked);//ok
	up_read(&current->mm->mmap_sem);

	printk("ret = %d\n",ret);
	if(ret > 0){
		printk("page_count(pages)) = %d\n",page_count(*pages));
		printk("address of pages is: 0x%lx\n",(unsigned long)pages);
	}

	//-----
	down_read(&current->mm->mmap_sem);
	//ret = get_user_pages_remote(current,mm,addr,len,gup_flags,NULL,NULL,NULL);//ok
	//ret = get_user_pages_remote(current,mm,addr,len,gup_flags,&pages1,NULL,NULL);//ok
	ret = get_user_pages_remote(current,mm,addr,len,gup_flags,&pages1,NULL,&locked);//ok
	up_read(&current->mm->mmap_sem);

	printk("ret = %d\n",ret);
	if(ret > 0){
		printk("page_count(pages1)) = %d\n",page_count(pages1));
		printk("address of pages1 is: 0x%lx\n",(unsigned long)pages1);
		printk("pfn: 0x%lx, zone name :%s \n", page_to_pfn(&pages1[0]),  page_zone(&pages1[0])->name);
	}
	return 0;
}

static void __exit get_user_pages_remote_exit(void){
	printk("exit ok!\n");
	return;
}

module_init(get_user_pages_remote_init);
module_exit(get_user_pages_remote_exit);

MODULE_LICENSE("GPL");
