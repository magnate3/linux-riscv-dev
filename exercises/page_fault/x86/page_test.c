#include <linux/init.h>
#include <linux/module.h>
#include <linux/init_task.h>
#include <asm/highmem.h>

MODULE_DESCRIPTION("Hello_world");
MODULE_LICENSE("GPL");

static unsigned long vaddr2paddr(unsigned long vaddr, struct task_struct *task)
{
    pgd_t *pgd;
    pud_t *pud;
    pmd_t *pmd;
    pte_t *pte;
    p4d_t *p4d;
    unsigned long paddr = 0;
    unsigned long page_addr = 0;
    unsigned long page_offset = 0;
    
    unsigned long pg,pm,pt;

    pgd = pgd_offset(task->mm, vaddr);
    pg = pgd_val(*pgd);
    //unsigned long index = pgd_index(vaddr);
    //printk("index = %lx", index);
    if (pgd_none(*pgd)) {
        //printk("not mapped in pgd\n");
        return -1;
    }

    p4d = p4d_offset(pgd, vaddr);
    if(p4d_none(*p4d)){
        //printk("not mapped in p4d\n");
        return -1;

    }

    pud = pud_offset(p4d, vaddr);
    if (pud_none(*pud)) {
        //printk("not mapped in pud\n");
        return -1;
    }

    pmd = pmd_offset(pud, vaddr);
    pm = pmd_val(*pmd);
    if (pmd_none(*pmd)) {
        //printk("not mapped in pmd\n");
        return -1;
    }

    pte = pte_offset_kernel(pmd, vaddr);
    pt = pte_val(*pte);
    if (pte_none(*pte)) {
        //printk("not mapped in pte\n");
        return -1;
    }

    /* Page frame physical address mechanism | offset */
    page_addr = pte_val(*pte) & PAGE_MASK;
    page_offset = vaddr & ~PAGE_MASK;
    paddr = page_addr | page_offset;

    printk("   va   ->   pgd  ->   pmd  ->   pte  ->   pa \n");
    printk("%8lx  %8lx  %8lx  %8lx  %8lx \n", vaddr, pg, pm, pt, paddr);
    return paddr;
}

/*
virtual kernel memory layout:
                   fixmap  : 0xfff14000 - 0xfffff000   ( 940 kB)
                 cpu_entry : 0xffa00000 - 0xffb39000   (1252 kB)
                   pkmap   : 0xff600000 - 0xff800000   (2048 kB)
                   vmalloc : 0xf7dfe000 - 0xff5fe000   ( 120 MB)
                   lowmem  : 0xc0000000 - 0xf75fe000   ( 885 MB)
*/
static void print_memory_layout(void){

    printk(KERN_INFO "virtual kernel memory layout:\n"
		"    fixmap  : 0x%08lx - 0x%08lx   (%4ld kB)\n"
#if 	CPU_ENTRY 
		"  cpu_entry : 0x%08lx - 0x%08lx   (%4ld kB)\n"
#endif
#ifdef CONFIG_HIGHMEM
		"    pkmap   : 0x%08lx - 0x%08lx   (%4ld kB)\n"
#endif
		"    vmalloc : 0x%08lx - 0x%08lx   (%4ld MB)\n"
		"    lowmem  : 0x%08lx - 0x%08lx   (%4ld MB)\n"
,
		FIXADDR_START, FIXADDR_TOP,
		(FIXADDR_TOP - FIXADDR_START) >> 10,
#if CPU_ENTRY
		CPU_ENTRY_AREA_BASE,
		CPU_ENTRY_AREA_BASE + CPU_ENTRY_AREA_MAP_SIZE,
		CPU_ENTRY_AREA_MAP_SIZE >> 10,
#endif
#ifdef CONFIG_HIGHMEM
		PKMAP_BASE, PKMAP_BASE+LAST_PKMAP*PAGE_SIZE,
		(LAST_PKMAP*PAGE_SIZE) >> 10,
#endif

		VMALLOC_START, VMALLOC_END,
		(VMALLOC_END - VMALLOC_START) >> 20,

		(unsigned long)__va(0), (unsigned long)high_memory,
		((unsigned long)high_memory - (unsigned long)__va(0)) >> 20);

}

static int test_init(void)
{	
    print_memory_layout();
    struct task_struct *tmp; 
    tmp = &init_task;
    struct task_struct *cur = current;
    int flag = 0;  
    unsigned long i;

    for(i =0xc0000000; i <=  0xf75fe000; i += 0x1000)
    {

        //printk("%lx\n",i);
        unsigned long old;
        unsigned long new;

        old = vaddr2paddr(i, cur);

        for_each_process(tmp){

            if(tmp -> mm == NULL){
                continue;
            }
            printk("========== pid: %d ===========\n", tmp->pid);
            new = vaddr2paddr(i, tmp);

            if(new != old){
                flag = 1;
                goto f;
            }
            
        }

    }

    f:
    if(flag){
        printk("False!!\n");
    }
    else{
        printk("True!!\n");
    }

    return 0;
}

static void __exit test_exit(void)
{

    printk(KERN_INFO "Bye !\n");
}

module_init(test_init);
module_exit(test_exit);
