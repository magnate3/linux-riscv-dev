#include<linux/module.h>
#include<linux/kernel.h>
#include<linux/mm.h>
#include<linux/mm_types.h>
#include <linux/seq_file.h>
#include <linux/sched/mm.h>
#include <linux/mm.h>
#include <linux/slab.h>
#include<linux/file.h>
#include<linux/fs.h>
#include<linux/path.h>
#include<linux/slab.h>
#include<linux/dcache.h>
#include<linux/sched.h>
#include<linux/uaccess.h>
#include<linux/fs_struct.h>
#include <asm/tlbflush.h>
#include<linux/uaccess.h>
#include<linux/device.h>
#include <linux/kthread.h> 
#include <linux/delay.h>
#include <linux/vmalloc.h>
#include <linux/rmap.h>
#include <linux/huge_mm.h>
#include <linux/kprobes.h>
#include <asm/pgtable.h>
#include <uapi/asm-generic/mman-common.h>
#include <linux/mman.h>

#include "btplus.h"

#define PAGES 512

#define HUGE_PAGE (1 << 21)
#define DEFAULT_PAGE 4096

static int major;
atomic_t  device_opened;
static struct class *demo_class;
struct device *demo_device;

bool list_init= false;
struct task_struct *kth_task = NULL;

struct kobject *cs614_kobject;
unsigned promote = 0;

static ssize_t  sysfs_show(struct kobject *kobj,
                        struct kobj_attribute *attr, char *buf);
static ssize_t  sysfs_store(struct kobject *kobj,
                        struct kobj_attribute *attr,const char *buf, size_t count);

struct kobj_attribute sysfs_attr; 

struct address{
    unsigned long from_addr;
    unsigned long to_addr;
};

struct input{
    unsigned long addr;
    unsigned length;
    struct address * buff;
};

int move_vma_to(unsigned long src, unsigned long dst) {
        struct vm_area_struct *src_vma, *after_dst_vma;
        struct mm_struct *mm;
        size_t copy_len;
        unsigned long dst_addr;
        unsigned long src_prot, prot_flags = 0, map_flags = 0;
        char *src_buffer;
        mm = get_task_mm(current);
        src_vma = vma_lookup(mm, src);
        if (!src_vma) {
                printk(KERN_INFO "illegal source address: vma not found\n");
                return -1;
        }
        after_dst_vma = find_vma(mm, dst);
        copy_len =  (size_t)(src_vma->vm_end - src_vma->vm_start);
        src_buffer = (char*)kzalloc(copy_len*sizeof(char), GFP_KERNEL);
        if(after_dst_vma != NULL) {
                // check if dst address lies inside this vma
                if (dst >= after_dst_vma->vm_start) {
                        printk(KERN_INFO "destination address already mapped\n");
                        goto err;
                }
                // check if space between dst and after_dst_vma->vm_start is large enough
                else if (copy_len >= after_dst_vma->vm_start - dst) {
                        printk(KERN_INFO "not enough space present at the destination\n");
                        goto err;
                }
        }
        
        src_prot = src_vma->vm_page_prot.pgprot;

        if (src_prot & PROT_READ) {
                prot_flags |= PROT_READ;
        }
        if (src_prot & PROT_WRITE) {
                prot_flags |= PROT_WRITE;
        }
        if (src_prot & PROT_EXEC) {
                prot_flags |= PROT_EXEC;
        }
        if (src_vma->vm_flags & MAP_SHARED) {
                map_flags |= MAP_SHARED;
        }
        if (src_vma->vm_flags & MAP_FIXED) {
                map_flags |= MAP_FIXED;
        }
        if(src_vma->vm_flags & MAP_PRIVATE) {
                map_flags |= MAP_PRIVATE;
        }
	if(src_vma->vm_flags & MAP_ANONYMOUS) {
		map_flags |= MAP_ANONYMOUS;
	}

        dst_addr = vm_mmap(NULL, dst, copy_len, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, 0);
        
        if (dst_addr != dst) {
                printk(KERN_INFO "vm_mmap failed\n");
                // vm_munmap(dst_addr, copy_len);
                goto err;
        }
        if(copy_from_user(src_buffer, (void*)(src_vma->vm_start), copy_len)){
                printk(KERN_INFO "copy_from_user failed\n");
                goto err;
        }

        if(copy_to_user((void*)dst_addr, src_buffer, copy_len)){
                printk(KERN_INFO "copy_to_user failed\n");
                goto err;
        }
	
	vm_munmap(src_vma->vm_start, copy_len);
        kfree(src_buffer);
        return 0;
err:
        kfree(src_buffer);
	return -1;
}

int move_vma(unsigned long src, unsigned long *dst) {
        struct mm_struct* mm;
        struct vma_iterator vm_iter;
        struct vm_area_struct* src_vma, *vma;
        unsigned long copy_len, dst_addr, prev_addr, new_addr;
        char* src_buffer;

        mm = get_task_mm(current);
        src_vma = vma_lookup(mm, src);
        
        if(src_vma == NULL){
                printk(KERN_INFO "Illegal source address. No vma found\n");
                return -1;
        }

        copy_len = src_vma->vm_end - src_vma->vm_start;
        src_buffer =  (char*)kzalloc(sizeof(char)*copy_len, GFP_KERNEL);

        prev_addr = src_vma->vm_end;
        vma_iter_init(&vm_iter, mm, prev_addr);
        for_each_vma(vm_iter, vma){
                // We have found a hole
                if(prev_addr + copy_len < vma->vm_start)
                        break;
                
                prev_addr = vma->vm_end;
        }

        dst_addr = prev_addr;
        new_addr = vm_mmap(NULL, dst_addr, copy_len, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, 0);

        if(new_addr != dst_addr){
                printk(KERN_INFO "vm_mmap failed\n");
                goto err;
        }
        
        if(copy_from_user(src_buffer, (void*)src_vma->vm_start, copy_len)){
        printk(KERN_INFO "copy_from_user failed\n");
                goto err;
        }
        
        if(copy_to_user((void*)new_addr, src_buffer, copy_len)){
                printk(KERN_INFO "copy_to_user failed\n");
                goto err;
        }

        memcpy(dst, &new_addr, sizeof(unsigned long));
        printk(KERN_INFO "value of dst:%lx, new_addr:%lx\n", *dst, new_addr);
        kfree(src_buffer);
        vm_munmap(src_vma->vm_start, copy_len);
        return 0;
err:
        kfree(src_buffer);
	return -1;

}

struct k_args {
	struct task_struct *proc;
};


void clear_ptes(struct mm_struct *mm, unsigned long address, pmd_t *pmd) {

        pte_t *ptep;

        for(int i =0; i < PAGES; i++) {
                ptep = pte_offset_map(pmd, address);
                if(ptep == NULL){
                        printk(KERN_INFO "pte_p is null\n\n");
                        goto nul_ret;
                }
                if(!pte_present(*ptep)) {
                        printk(KERN_INFO "pte not present\n");
                        goto nul_ret;    
                }
                pte_clear(mm, address, ptep);
                address += DEFAULT_PAGE;
        }


nul_ret:
        return;
}




pmd_t *get_pmd(unsigned long address, struct mm_struct *mm) {
        pgd_t *pgd;
        p4d_t *p4d;
        pud_t *pud;
        pmd_t *pmd;
       

        pgd = pgd_offset(mm, address);
        if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
                goto nul_ret;
        p4d = p4d_offset(pgd, address);
        if (p4d_none(*p4d))
                goto nul_ret;
        if (unlikely(p4d_bad(*p4d)))
                goto nul_ret;
        pud = pud_offset(p4d, address);
        if (pud_none(*pud))
                goto nul_ret;
        if (unlikely(pud_bad(*pud)))
                goto nul_ret;
        pmd = pmd_offset(pud, address);
        if (pmd_none(*pmd))
                goto nul_ret;
		
        if (unlikely(pmd_trans_huge(*pmd))){
                printk(KERN_INFO "I am huge\n");
                goto nul_ret;
        }

        return pmd;

nul_ret:
        return NULL;
}


pte_t *check_physically_allocated(unsigned long addr, struct mm_struct *mm) {
        pmd_t *pmd = get_pmd(addr, mm);
        pte_t *ptep;
        if (pmd == NULL){
                return NULL;
        }
        
        ptep = pte_offset_map(pmd, addr);
        if(!ptep) {
                return NULL;
        }
        if(!pte_present(*ptep))
                return NULL;
        
        return ptep;
}


pmd_t *check_ptes(unsigned long address, struct mm_struct *mm)
{
        pgd_t *pgd;
        p4d_t *p4d;
        pud_t *pud;
        pmd_t *pmd;
        pte_t *ptep;
        struct vm_area_struct *vma = find_vma(mm, address);
        if(!vma){
                 goto nul_ret;
        }
       

        pgd = pgd_offset(mm, address);
        if (pgd_none(*pgd) || unlikely(pgd_bad(*pgd)))
                goto nul_ret;
        p4d = p4d_offset(pgd, address);
        if (p4d_none(*p4d))
                goto nul_ret;
        if (unlikely(p4d_bad(*p4d)))
                goto nul_ret;
        pud = pud_offset(p4d, address);
        if (pud_none(*pud))
                goto nul_ret;
        if (unlikely(pud_bad(*pud)))
                goto nul_ret;
        pmd = pmd_offset(pud, address);
        if (pmd_none(*pmd))
                goto nul_ret;
		
        if (unlikely(pmd_trans_huge(*pmd))){
                printk(KERN_INFO "I am huge\n");
                goto nul_ret;
        }

        for(int i =0; i < PAGES; i++) {
                ptep = pte_offset_map(pmd, address);
                if(ptep == NULL){
                        printk(KERN_INFO "pte_p is null\n\n");
                        goto nul_ret;
                }
                if(!pte_present(*ptep)) {
                        printk(KERN_INFO "pte not present\n");
                        goto nul_ret;    
                }

                address += DEFAULT_PAGE;
        }

        return pmd;

nul_ret:
        return NULL;

}


struct pmd_entry {
        struct list_head list;
        pmd_t default_pmd;
        unsigned long addr;
};

struct list_head pmd_entry_list;

void demote_pages(unsigned long start_addr, size_t len) {
        struct vm_area_struct *curr_vma;
        unsigned long addr = start_addr;
        struct mm_struct *mm  = get_task_mm(current);
        pmd_t *pmd;
        struct pmd_entry *iter;
        struct page *huge_page;
        curr_vma = find_vma(mm, addr);
        while(curr_vma->vm_end - addr >= HUGE_PAGE) {
                unsigned long rem = addr % HUGE_PAGE;
                if (rem != 0) {
                        printk(KERN_INFO "%s: %ld is not 2MB aligned\n", __func__,addr);
                        addr  -= rem;
                        if (addr < curr_vma->vm_start)
                                addr += HUGE_PAGE;
                        
                        continue; 
                }
                pmd = get_pmd(addr, mm);
                if (!pmd) {
                        // printk(KERN_INFO "%s: region starting with %ld is not mapped fully\n",__func__, addr);
                        addr += HUGE_PAGE;
                        if (addr > start_addr + len) {
                                break;
                        }
                        continue;
                }

                list_for_each_entry(iter, &pmd_entry_list, list) {
                        if (iter->addr == addr) {
                                break;
                        }
                }
                // list_del(&iter->list);
                // kfree(iter);
                huge_page = pmd_page(*pmd);
                __free_pages(huge_page, 9);
                set_pmd(pmd, iter->default_pmd);
                addr += HUGE_PAGE;
                if(addr > start_addr + len)
                        break;
        }

}

static int __kprobes handle_munmap(struct kprobe *p, struct pt_regs *regs) {
        unsigned long addr;
        size_t len;
        addr = (unsigned long)regs->di;
        len = (size_t)regs->si;
        demote_pages(addr, len);
        // struct pmd_entry *tmp, *iter;
        // if (list_init) {
        //         list_for_each_entry_safe(iter, tmp, &pmd_entry_list, list) {
        //                 list_del(&iter->list);
        //                 kfree(iter);
        //         }
        //         list_init = false;
        // }
        return 0;
}

int promote_pages(void *args) {
        struct k_args *_kargs = (struct k_args *)(args);
        struct task_struct *proc = _kargs->proc;
        struct vma_iterator vmi;
        struct vm_area_struct *vma;
        struct mm_struct *mm = get_task_mm(proc);
        struct page *huge_page;
        unsigned long haddr;
        struct pmd_entry *_list_entry;
        pmd_t *pmd;
        pmd_t entry;
        char *kbuf;

        while (promote == 0) {
                schedule_timeout_interruptible(5);
        }
        while(!kthread_should_stop()) {
                if (promote == 0) {
                        msleep(5000);
                }
        
                INIT_LIST_HEAD(&pmd_entry_list);
                list_init = true;
                vma = find_vma(mm, 0);
                if (!vma) {
                        printk(KERN_INFO "%s: vma not found\n", __func__);
                        goto exit_while;
                }
                kbuf = kzalloc(HUGE_PAGE*sizeof(char), GFP_KERNEL);

                vma_iter_init(&vmi, mm, vma->vm_start);
                
                for_each_vma(vmi, vma) {
                        unsigned long addr = vma->vm_start;
                        // check if size of vma is more than 2 MB
                        while (vma->vm_end - addr >= HUGE_PAGE) {
                                unsigned long rem = addr % HUGE_PAGE;
                                if (rem != 0) {
                                        printk(KERN_INFO "%s: %ld is not 2MB aligned\n", __func__,addr);

                                        addr  += HUGE_PAGE - rem;
                                        continue; 
                                }

                                //proceed to check physical allocation of 2MB region starting with addr
                                pmd = check_ptes(addr, mm);
                                if (!pmd) {
                                        printk(KERN_INFO "%s: region starting with %ld is not mapped fully\n", 
                                        __func__, addr);
                                        addr += HUGE_PAGE;
                                        continue;
                                }

                                 // add this pmd to list
                                _list_entry = (struct pmd_entry*)kzalloc(sizeof(struct pmd_entry), GFP_KERNEL);
                                _list_entry->default_pmd = *pmd;
                                _list_entry->addr = addr;
                                list_add(&_list_entry->list, &pmd_entry_list);

                                printk(KERN_INFO "region starting with %ld is fully allocated\n", addr);
                                // copy the data from 512 pages 
                                access_process_vm(proc, addr, kbuf, HUGE_PAGE, FOLL_REMOTE);                        
                                // clear the ptes
                                // clear_ptes(mm, addr, pmd);
                                // allocate the new region
                                huge_page = alloc_pages(GFP_KERNEL, 9);
                                printk(KERN_INFO "%s: Huge page allocated\n", __func__);
                                //copy the data to new huge page
                                memcpy((void*)page_address(huge_page), kbuf, HUGE_PAGE);
                                printk(KERN_INFO "memcpy success\n");
                                // set the new pmd
                                entry = pmd_mkhuge(mk_pmd(huge_page, vma->vm_page_prot));

                                if (vma->vm_flags && VM_WRITE)
                                        entry = pmd_mkwrite(pmd_mkdirty(entry));

                                haddr = addr;
                                set_pmd_at(mm, haddr, pmd, entry);
                                printk(KERN_INFO "set_pmd_at done\n");
                                
                               
                                _list_entry = NULL;
                                addr += HUGE_PAGE;
                        }
                        printk(KERN_INFO "vma processed\n");


                }
        exit_while:        
                promote = 0;
                break;
        }

        kfree(kbuf);
        printk(KERN_INFO "Promotion done for %d\n", proc->pid);
        kth_task = NULL;
        return 0;
}

void initialize_thread(void) {
	char k_name[20];
	struct k_args *args = (struct k_args*)kzalloc(sizeof(struct k_args), GFP_KERNEL);
	sprintf(k_name, "promote");
	args->proc = current;
	kth_task = kthread_create(promote_pages, args, (const char *)(k_name));
	if (kth_task != NULL) {
		wake_up_process(kth_task);
                printk(KERN_INFO "kernel thread is running\n");
	}
	else {
		printk(KERN_INFO "kernel thread promote could not be created\n");
	}
}	

size_t get_pgnr(unsigned long addr, unsigned long start) {
        return ((addr -  start) >> 12);
}

int promote_compacted_vma(unsigned long start, unsigned long end, struct mm_struct *mm, struct vm_area_struct *vma) {
        char *kbuf;
        unsigned long addr = start;
        struct pmd_entry *_list_entry;
        pmd_t *pmd;
        struct page *huge_page;
        pmd_t entry;
        INIT_LIST_HEAD(&pmd_entry_list);
        list_init = true;
        if (!vma) {
                printk(KERN_INFO "%s: vma not found\n", __func__);
                return -1;
        }
        kbuf = kzalloc(HUGE_PAGE*sizeof(char), GFP_KERNEL);

        while (end - addr >= HUGE_PAGE) {
                unsigned long rem = addr % HUGE_PAGE;
                if (rem != 0) {
                        printk(KERN_INFO "%s: %ld is not 2MB aligned\n", __func__,addr);

                        addr  += HUGE_PAGE - rem;
                        continue; 
                }

                //proceed to check physical allocation of 2MB region starting with addr
                pmd = check_ptes(addr, mm);
                if (!pmd) {
                        printk(KERN_INFO "%s: region starting with %ld is not mapped fully\n", 
                        __func__, addr);
                        addr += HUGE_PAGE;
                        continue;
                }

                // add this pmd to list
                _list_entry = (struct pmd_entry*)kzalloc(sizeof(struct pmd_entry), GFP_KERNEL);
                _list_entry->default_pmd = *pmd;
                _list_entry->addr = addr;
                list_add(&_list_entry->list, &pmd_entry_list);

                printk(KERN_INFO "region starting with %ld is fully allocated\n", addr);
                // copy the data from 512 pages 
                access_process_vm(current, addr, kbuf, HUGE_PAGE, FOLL_REMOTE);                        
   
                huge_page = alloc_pages(GFP_KERNEL, 9);
                printk(KERN_INFO "%s: Huge page allocated\n", __func__);
                //copy the data to new huge page
                memcpy((void*)page_address(huge_page), kbuf, HUGE_PAGE);
                printk(KERN_INFO "memcpy success\n");
                // set the new pmd
                entry = pmd_mkhuge(mk_pmd(huge_page, vma->vm_page_prot));

                if (vma->vm_flags && VM_WRITE)
                        entry = pmd_mkwrite(pmd_mkdirty(entry));

                set_pmd_at(mm, addr, pmd, entry);
                printk(KERN_INFO "set_pmd_at done\n");
                
                
                _list_entry = NULL;
                addr += HUGE_PAGE;
        }

        kfree(kbuf);
        return 0;

}

int compact_vma(unsigned long start, int num_pages, struct address *mapping_buf) {
        struct mm_struct *mm = get_task_mm(current);
        struct vm_area_struct *vma = find_vma(mm, start);
        size_t len = num_pages*DEFAULT_PAGE;
        char *buf = (char*)kzalloc(DEFAULT_PAGE, GFP_KERNEL);
        unsigned long end_addr = start + len;
        unsigned long addr = start;
        // unsigned long prev_unalloc = 0;
        pte_t *ptep;
        pte_t **pte_arr = (pte_t**)kzalloc(num_pages*sizeof(pte_t*), GFP_KERNEL);
        
        unsigned long *empty_pages_addr = (unsigned long*)kzalloc(num_pages*sizeof(unsigned long), GFP_KERNEL);
        bool *page_to_alloc_map = (bool*)kzalloc(num_pages*sizeof(bool), GFP_KERNEL);
        size_t unalloc_pages= 0, alloc_pages = 0;
        size_t i = 0, j = 0;
        if (start < vma->vm_start) {
                printk(KERN_INFO "start addr out of vm area\n");
                return -1;
        }
        if (end_addr > vma->vm_end) {
                printk(KERN_INFO "addr range out of vma\n");
                return -1;
        }

        addr = start;
        while(addr < end_addr) {
                mapping_buf[i].from_addr = addr;
                mapping_buf[i].to_addr = addr;
                i++;
                addr += DEFAULT_PAGE;
        }

        i = 0;
        addr =  start;
        while(addr < end_addr) {
                ptep = check_physically_allocated(addr, mm);
                if(ptep == NULL) {
                        empty_pages_addr[unalloc_pages++] = addr;
                        page_to_alloc_map[i] = false;
                }
                else {
                        page_to_alloc_map[i] = true;
                        alloc_pages ++;
                }
                addr += DEFAULT_PAGE;
                pte_arr[i] = ptep;
                i++;
        }
        addr = start;
        i = 0;
        j = 0;
        while(addr < end_addr) {
                if(page_to_alloc_map[i]) {
                        if (i >= alloc_pages) {
                                if (copy_from_user(buf, (void*)addr, DEFAULT_PAGE)) {
                                        printk(KERN_INFO "copy_from_user failed\n");
                                        goto err;
                                }
                                if (copy_to_user((void*)empty_pages_addr[j], buf, DEFAULT_PAGE)) {
                                        printk(KERN_INFO "copy_to_user failed\n");
                                        goto err;
                                }

                                if(pte_arr[i] == NULL) {
                                        printk(KERN_INFO "page unallocted unexpected\n");
                                        goto err;
                                }
                                pte_clear(mm, addr, pte_arr[i]);
                                mapping_buf[i].to_addr = empty_pages_addr[j];
                                mapping_buf[get_pgnr(empty_pages_addr[j], start)].to_addr = addr;
                                j++;
                        }
                }
                i++;
                addr += DEFAULT_PAGE;
        }


        kfree(buf);
        kfree(empty_pages_addr);
        kfree(page_to_alloc_map);
        kfree(pte_arr);

        return promote_compacted_vma(start, end_addr, mm, vma);

err:
        kfree(buf);
        kfree(empty_pages_addr);
        kfree(page_to_alloc_map);
        kfree(pte_arr);
        return -1;
}

static int device_open(struct inode *inode, struct file *file)
{
        atomic_inc(&device_opened);
        try_module_get(THIS_MODULE);
        printk(KERN_INFO "Device opened successfully\n");
        return 0;
}

static int device_release(struct inode *inode, struct file *file)
{
        

        atomic_dec(&device_opened);
        module_put(THIS_MODULE);
        printk(KERN_INFO "Device closed successfully\n");

        return 0;
}


static ssize_t device_read(struct file *filp,
                           char *buffer,
                           size_t length,
                           loff_t * offset){

    printk("read called\n");

    return 0;
}

static ssize_t
device_write(struct file *filp, const char *buff, size_t len, loff_t * off){
    
        printk("write called\n");
        

        return 8;
}


long device_ioctl(struct file *file,	
		 unsigned int ioctl_num,
		 unsigned long ioctl_param)
{
	//unsigned long addr = 1234;
	int ret = 0; // on failure return -1
	struct address * buff = NULL;
	unsigned long vma_addr = 0;
	unsigned long to_addr = 0;
	unsigned length = 0;
	struct input* ip;
	// unsigned index = 0;
	struct address temp;
        struct address *mapping;
	/*
	 * Switch according to the ioctl called
	 */
	switch (ioctl_num) {
	case IOCTL_MVE_VMA_TO:
	    buff = (struct address*)vmalloc(sizeof(struct address)) ;
	    printk("move VMA at a given address");
	    if(copy_from_user(buff,(char*)ioctl_param,sizeof(struct address))){
	        pr_err("MVE_VMA address write error\n");
		return ret;
	    }
	    vma_addr = buff->from_addr;
	    to_addr = buff->to_addr;
	    printk("address from :%lx, to:%lx \n",vma_addr,to_addr);
	    vfree(buff);
            ret = move_vma_to(vma_addr, to_addr);

	    return ret;
	case IOCTL_MVE_VMA:
	    buff = (struct address*)vmalloc(sizeof(struct address)) ;
	    printk("move VMA to available hole address");
	    if(copy_from_user(buff,(char*)ioctl_param,sizeof(struct address))){
	        pr_err("MVE_VMA address write error\n");
		return ret;
	    }
	    vma_addr = buff->from_addr;
	    printk("VMA address :%lx \n",vma_addr);
            to_addr = 0;
            ret = move_vma(vma_addr, &to_addr);
            buff->to_addr = to_addr;
            if (copy_to_user((char*)ioctl_param, buff, sizeof(struct address))){
                printk(KERN_INFO "copy_to_user failed in ioctl\n");
                ret = -1;
            }
	    vfree(buff);
            return ret;
        case IOCTL_PROMOTE_VMA:
            printk("promote 4KB pages to 2\n");
	    initialize_thread();
	    return ret;
	case IOCTL_COMPACT_VMA:
	    printk("compact VMA\n");
	    ip = (struct input*)vmalloc(sizeof(struct input)) ;
	    if(copy_from_user(ip,(char*)ioctl_param,sizeof(struct input))){
                pr_err("MVE_MERG_VMA address write error\n");
                return ret;
            }
	    vma_addr = ip->addr;
	    length = ip->length;
	    buff = ip->buff;
	    temp.from_addr = vma_addr;
	    temp.to_addr = vma_addr;
	    printk("vma address:%lx, length:%u, buff:%lx\n",vma_addr,length,(unsigned long)buff);
	    //populate old to new address mapping in user buffer.
	    //number of entries in this buffer is equal to the number of 
	    //virtual pages in vma address range
	    //index of moved addr in mapping table is , index = (addr-vma_address)>>12
            mapping = (struct address *)kzalloc(sizeof(struct address)*length, GFP_KERNEL);
            ret = compact_vma(vma_addr, length, mapping);
        
	    if(copy_to_user((struct address *)buff, mapping, length*sizeof(struct address))){
                kfree(mapping);
                vfree(ip);
	        pr_err("COMPACT VMA read error\n");
		return -1;
	    }
	    vfree(ip);
            kfree(mapping);
            return ret;
	}
	return ret;
}


static struct kprobe kp = {
        .symbol_name   = "__vm_munmap",
	.pre_handler = handle_munmap,
	.post_handler = NULL,
};

static struct file_operations fops = {
        .read = device_read,
        .write = device_write,
	.unlocked_ioctl = device_ioctl,
        .open = device_open,
        .release = device_release,
};

static char *demo_devnode(struct device *dev, umode_t *mode)
{
        if (mode && dev->devt == MKDEV(major, 0))
                *mode = 0666;
        return NULL;
}

//Implement required logic
static ssize_t sysfs_show(struct kobject *kobj, struct kobj_attribute *attr,
                      char *buf)
{

        pr_info("sysfs read\n");
        sprintf(buf, "%d",promote);
        return 0;
}

//Implement required logic
static ssize_t sysfs_store(struct kobject *kobj, struct kobj_attribute *attr,
                     const char *buf, size_t count)
{
        int err, val;
        printk("sysfs write\n");
        err = kstrtoint(buf, 10, &val);
        if (err || val != 1) {
                printk(KERN_INFO "%d: wrong value passed in buffer\n", val);
                return -EINVAL;
        }
        promote = val;
        return count;
}



int init_module(void)
{
        int err;
	printk(KERN_INFO "Hello kernel\n");
        major = register_chrdev(0, DEVNAME, &fops);
        err = major;
        if (err < 0) {      
             printk(KERN_ALERT "Registering char device failed with %d\n", major);   
             goto error_regdev;
        }                 
        
        demo_class = class_create(THIS_MODULE, DEVNAME);
        err = PTR_ERR(demo_class);
        if (IS_ERR(demo_class))
                goto error_class;

        demo_class->devnode = demo_devnode;

        demo_device = device_create(demo_class, NULL,
                                        MKDEV(major, 0),
                                        NULL, DEVNAME);
        err = PTR_ERR(demo_device);
        if (IS_ERR(demo_device))
                goto error_device;
 
        printk(KERN_INFO "I was assigned major number %d. To talk to\n", major);                                                              
        atomic_set(&device_opened, 0);
        
	cs614_kobject = kobject_create_and_add("kobject_cs614", kernel_kobj);
        
	if(!cs614_kobject)
            return -ENOMEM;
	
	sysfs_attr.attr.name = "promote";
	sysfs_attr.attr.mode = 0666;
	sysfs_attr.show = sysfs_show;
	sysfs_attr.store = sysfs_store;

	err = sysfs_create_file(cs614_kobject, &(sysfs_attr.attr));
	if (err){
	    pr_info("sysfs exists:");
	    goto r_sysfs;
	}
        err = register_kprobe(&kp);
        if (err < 0) {
                printk(KERN_INFO "register_kprobe failed, returned %d\n", err);
                return err;
        }
        printk(KERN_INFO "Planted kprobe at %lx\n", (unsigned long)kp.addr);
	return 0;
r_sysfs:
	kobject_put(cs614_kobject);
        sysfs_remove_file(kernel_kobj, &sysfs_attr.attr);
error_device:
         class_destroy(demo_class);
error_class:
        unregister_chrdev(major, DEVNAME);
error_regdev:
        return  err;
}

void cleanup_module(void)
{
        device_destroy(demo_class, MKDEV(major, 0));
        class_destroy(demo_class);
        unregister_chrdev(major, DEVNAME);
	kobject_put(cs614_kobject);
	sysfs_remove_file(kernel_kobj, &sysfs_attr.attr);
        unregister_kprobe(&kp);
        if (kth_task != NULL) {
                printk(KERN_INFO "calling kthread_stop\n");
                BUG_ON(!kth_task);
                kthread_stop(kth_task);
                kth_task = NULL;
        }
	printk(KERN_INFO "Goodbye kernel\n");
}

MODULE_AUTHOR("cs614");
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("assignment2");
