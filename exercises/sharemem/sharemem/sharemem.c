#include <linux/module.h>
#include <asm/uaccess.h>
#include <linux/version.h>
#include <linux/io.h>
#include <linux/kobject.h>
#include <linux/string.h>

#include <asm/cacheflush.h>
#include <linux/fdtable.h>
#include <linux/file.h>
#include <linux/freezer.h>
#include <linux/fs.h>
#include <linux/list.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/nsproxy.h>
#include <linux/poll.h>
#include <linux/debugfs.h>
#include <linux/rbtree.h>
#include <linux/sched.h>
#include <linux/seq_file.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>
#include <linux/slab.h>
#include <linux/pid_namespace.h>
#include <linux/security.h>

#define get_task_struct(tsk) do { atomic_inc(&(tsk)->usage); } while(0)
#define FORBIDDEN_MMAP_FLAGS                (VM_WRITE)
static struct dentry *sharemem_debugfs_dentry_root;
static void* buffer_addr=NULL;
struct page **pages_ptr_array=NULL;
struct list_head sharemem_procs;
struct sharemem_proc{
    struct vm_area_struct *vma;
	struct mm_struct *vma_vm_mm;
    void* kernel_addr;
    unsigned long buffer_size;
    unsigned long vma_size;
   	struct page **pages;
	struct task_struct *tsk;
	struct files_struct *files;
    struct list_head list;
    unsigned int namelen;
    struct mutex mmap_mutex;
};

static int sharemem_open(struct inode *inodep, struct file *filp)
{
    struct sharemem_proc *proc;
    proc = kzalloc(sizeof(*proc), GFP_KERNEL);
    if (proc == NULL)
		return -ENOMEM;
    list_add(&proc->list,&sharemem_procs);
    get_task_struct(current);
    proc->tsk = current;
    proc->namelen=strlen(proc->tsk->comm);
    proc->kernel_addr=NULL;
    mutex_init(&proc->mmap_mutex);
    filp->private_data=proc;
    return 0;
}

static int sharemem_mmap(struct file *file,struct vm_area_struct *vma)
{
    int ret=0;
    struct sharemem_proc *proc=file->private_data;
   	struct vm_struct vm_area; 
   	struct vm_struct *area;
    struct page** pages;
#ifdef DEBUG_USER_MAP_SHARED
    static int once=0;
#endif
    if(proc->tsk != current){
        printk("%s:tsk=%s current=%s --jiazi\n",__func__,proc->tsk->comm,current->comm);
        ret=-EINVAL;
        goto out;
    } 
 //   printk("%s: mmap_size=%ld  --jiazi\n",__func__,vma->vm_end-vma->vm_start); 


    vma->vm_flags = (vma->vm_flags | VM_DONTCOPY) ;
    mutex_lock(&proc->mmap_mutex);
    if(buffer_addr) {
   	//	ret = -EBUSY;
		printk("%s:already mapped buffer=0x%x--jiazi\n",__func__,((int*)buffer_addr)[0]);
        mutex_unlock(&proc->mmap_mutex);
        goto MMAP_USER_SPACE;
    }
   	area = __get_vm_area(vma->vm_end - vma->vm_start, VM_IOREMAP,VMALLOC_START,VMALLOC_END);
	if (area == NULL) {
		ret = -ENOMEM;
		printk("%s:get_vm_area failed --jiazi\n",__func__);
		goto out;
	}
    buffer_addr=area->addr;
    //printk("%s:%d  --jiazi\n",__FILE__,__LINE__);
    pages_ptr_array = kzalloc(sizeof(pages_ptr_array[0]), GFP_KERNEL);
	pages_ptr_array[0] = alloc_page(GFP_KERNEL | __GFP_HIGHMEM | __GFP_ZERO);

    vm_area.addr=buffer_addr;
    vm_area.size=PAGE_SIZE<<1;
    pages=pages_ptr_array;
  //  printk("%s[1]: %p  %p--jiazi\n",__func__,pages,pages_ptr_array[0]);
  //  ret=map_vm_area(&vm_area, PAGE_KERNEL, &pages); //pages will be modified
    // The protype of map_vm_area is changed in Kernel 3.18
    ret=map_vm_area(&vm_area, PAGE_KERNEL, pages);
    if(ret){
        printk("%s: ret=%d --jiazi\n",__func__,ret);
        goto out;
    }
  //  printk("%s[2]: %p  %p--jiazi\n",__func__,pages,pages_ptr_array[0]);
    mutex_unlock(&proc->mmap_mutex);

MMAP_USER_SPACE:
	proc->kernel_addr = buffer_addr;
    proc->pages = pages_ptr_array;
    //printk("%s[3]: %p  --jiazi\n",__func__,proc->pages);
    //BUG_ON(proc->pages);

    if(proc->pages[0] == NULL) {
        printk("%s: binder_alloc_buf failed for page at %p\n",
				__func__, buffer_addr);
        ret=-ENOMEM;
        goto out;
    }
    printk("%s:vm_start=0x%p  page=0x%p --jiazi\n",__func__,(int*)vma->vm_start,proc->pages[0]);
    ret = vm_insert_page(vma, vma->vm_start, proc->pages[0]);
	if (ret) {
	    printk(" binder_alloc_buf failed to map page at %lx-%lx in userspace\n",  vma->vm_start,vma->vm_end);
		goto out;
	}
#ifdef DEBUG_USER_MAP_SHARED
    if(!once) {
        printk("-----------------once time----------------------\n");
        ((int*)buffer_addr)[0]=0x11223344;
        once=1;
    }
#endif
    printk("%s Success vm_start[0]=0x%x--jiazi\n",__func__,((int*)vma->vm_start)[0]);


    out:
        return ret;
}
static ssize_t sharemem_read(struct file *filp, char __user *buf, size_t count,
                loff_t *f_pos)
{
    struct sharemem_proc *proc=filp->private_data;
    int retval;
    int i=0;
    int* buffer=(int*)buffer_addr;
    if(count > proc->namelen){
        retval=proc->namelen;
        proc->namelen=0;
    }else{
        retval=count;
        proc->namelen -= count;
    }
    if (copy_to_user(buf, proc->tsk->comm,retval)) {
		retval = -EFAULT;
        printk("%s:%p %d\n",__func__,buf,(int)count);
	}
    //printk("%s:%s %d\n",__func__,buf,count);
    if(buffer){
        for(i=0; i<1;i++)
        printk("%s:%p->%x  --jiazi\n",__func__,buffer,buffer[i]);
    }
    return retval;
}
ssize_t sharemem_write(struct file *filp, const char __user *buf, size_t count,
                loff_t *f_pos)
{
    struct sharemem_proc *proc=filp->private_data;
    int ret=count;
    if(buffer_addr == NULL){
        ret=-EINVAL;
        printk("No Mapping now  --jiazi\n");
        goto out;
    }
    if(count > PAGE_SIZE)
        count=PAGE_SIZE;
    mutex_lock(&proc->mmap_mutex);
    if(copy_from_user(buffer_addr,buf,count)){
        ret=-EFAULT;
        mutex_unlock(&proc->mmap_mutex);
        goto out;
    }
    mutex_unlock(&proc->mmap_mutex);
    out:
        return ret;
}
static const struct file_operations sharemem_fops = {
	.owner = THIS_MODULE,
	.mmap = sharemem_mmap,
	.open = sharemem_open,
    .read = sharemem_read,
    .write = sharemem_write,
};
static struct miscdevice sharemem_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "sharemem",
	.fops = &sharemem_fops
};
static int __init sharemem_init(void)
{
    int ret=0;
    INIT_LIST_HEAD(&sharemem_procs);
    sharemem_debugfs_dentry_root=debugfs_create_dir("sharemem", NULL);
    ret = misc_register(&sharemem_miscdev);    
    return ret;
}
static void __exit sharemem_exit(void)
{
    int ret;
    struct sharemem_proc *proc;
    __free_page(pages_ptr_array[0]);
    kfree(pages_ptr_array);
    list_for_each_entry(proc,&sharemem_procs,list)
        kfree(proc);
    misc_deregister(&sharemem_miscdev);
    //ret=misc_deregister(&sharemem_miscdev);
    //if(ret)
    //    printk("failed to unregister %s \n",__func__);
}
module_init(sharemem_init);
module_exit(sharemem_exit)
MODULE_LICENSE("GPL");
