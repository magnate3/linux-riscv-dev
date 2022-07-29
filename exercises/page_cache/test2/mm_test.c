#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/sched.h> 
#include <linux/init.h>
#include <linux/sched/signal.h>   
#include <linux/fdtable.h>
#include <linux/fs_struct.h>
#include <linux/file.h>
#include <linux/fs.h>
#include <linux/path.h>
#include <linux/dcache.h>
#include <linux/backing-dev-defs.h>
#include <linux/moduleparam.h>
#include <linux/slab.h>
#include <linux/pagemap.h>
#include <linux/buffer_head.h>
#include <linux/genhd.h>

MODULE_LICENSE("Dual BSD/GPL");



static int processid;


module_param(processid, int, 0660);

void PCB_Task_Structure(int processid){
	
	struct task_struct*  task_list;
	size_t pc = 0;
	struct file* cur;
	char *path_name = kmalloc(1280,GFP_KERNEL);
	char *path_res = kmalloc(1280,GFP_KERNEL);
	struct page* page_cache;
	int totalpage = 0;
	for_each_process (task_list){
		
		if(task_list->pid == processid){
			
			printk(KERN_INFO "PCB of the process with given pid = %u  \n",processid);
			
			struct fs_struct* fs = task_list->fs; // file system information
			
			struct buffer_head *bh; 
			
			struct files_struct* os = task_list->files; // open filesystem
			int count = 0;
			int index = 0;
			int i = 0;
			
			while(os->fdt->fd[index] != NULL){
				cur = os->fdt->fd[index];
				
				printk(KERN_INFO "1 - descriptor number= %u  \n",index);
				printk(KERN_INFO "2 - current file position number = %lli  \n",cur->f_pos);
				printk("3 - user id = %u  \n",cur->f_path.dentry->d_inode->i_uid);
				printk(KERN_INFO "4 - process access mode = %i  \n",cur->f_mode);
				printk(KERN_INFO "5 - ***** name of the file = %s  \n",cur->f_path.dentry->d_iname);
				printk(KERN_INFO "6 - inode number of the file = %li  \n",cur->f_path.dentry->d_inode->i_ino);
				printk(KERN_INFO "7 - file length = %lld bytes   \n",cur->f_path.dentry->d_inode->i_size);
				printk(KERN_INFO "8 - number of blocks allocated to file = %lu    \n",cur->f_path.dentry->d_inode->i_blocks);
				i=  cur->f_path.dentry->d_inode->i_mapping->nrpages;
				if(i>0){
					count = 0;
					while(count != i){
						if(fs->pwd.dentry->d_inode->i_mapping != NULL){
							//printk("1");
							page_cache = find_get_page(cur->f_path.dentry->d_inode->i_mapping,cur->f_path.dentry->d_inode->i_mapping->writeback_index);
							if(page_cache!= NULL){
								//printk("2%i\n",page_cache->flags);
			                                         printk(KERN_INFO "Use Count: %ld", page_cache->_refcount);
								bh = page_cache->private;
                                                                 
								if(bh != NULL){
									printk("3");
									printk("11 - the storage device the block is in = %lu \n",bh->b_count);
                                                                        printk("12 Storage Device (Search Key): %d", bh->b_bdev->bd_dev);
			                                                printk("13 Block Number: %ld", bh->b_blocknr);
								}
							}
						}
						count++;
					}
				}
				totalpage += cur->f_path.dentry->d_inode->i_mapping->nrpages; // blocks that are cached for the processes are counted here
				index++;
			}
			
			path_res = d_path(&fs->pwd, path_name, 1280);
			printk(KERN_INFO "9 - name of the current directory of the process = %s  \n",path_res);
			
			
			printk("10 - blocks that are cached for the process in the page cache = %i \n",totalpage);
			
			
			index = 0;
			count = 0;
			/*if(totalpage > 100){
				while(index < 100){
					
					while(os->fdt->fd[index] != NULL){
						cur = os->fdt->fd[index];
						i = 0;
						while(i< cur->f_path.dentry->d_inode->i_mapping->nrpages ){
							
							page_cache = find_get_page(cur->f_path.dentry->d_inode->i_mapping,index);
							
							bh = (struct buffer_head*)page_cache->private;
						
						if(bh->b_bdev->bd_disk->disk_name!= NULL){
							printk("11 - the storage device the block is in = %s \n",bh->b_bdev->bd_disk->disk_name);
						}
							printk("12 - the block number = %i \n",count);
						
							//printk("13 - use count = %li \n",page_cache->flags);
							count++;
							if(count >= 100)
								break;
							i++;
						}
						if(count >= 100)
								break;
					}
					index++;
				}
			}*/
			
		}
		pc ++;
	}
	
	
}



int init_module(void){
	printk(KERN_INFO "Hello World \n");
	
	PCB_Task_Structure(processid);
  
	return 0;
}

void cleanup_module(void){
	printk(KERN_INFO "Goodbye world \n");
}
