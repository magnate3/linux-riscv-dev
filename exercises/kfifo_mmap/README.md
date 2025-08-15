# mmap loldata

```
struct lophilo_data {
	lophilo_update_t updates[LOPHILO_SOURCE_MAX+1];
};
```

```
	// set datastore
	loldata = (struct lophilo_data*) kmalloc(PAGE_SIZE, GFP_KERNEL);
	if (remap_pfn_range(vma, vma->vm_start, __pa(loldata) >> PAGE_SHIFT,
	                    size, vma->vm_page_prot)) {
		printk(KERN_INFO "Allocation failed!");
                return -EAGAIN;
	}
```

## fifo

```
static DEFINE_KFIFO(updates, lophilo_update_t, LOPHILO_FIFO_SIZE);
```

#  lophilo_user2 
```
[root@centos7 cdev]# ./lophilo_user2 
***********bytes 32,  count 4 ****************
index 0 , data ->value 1655364167 
index 1 , data ->value 793576 
index 2 , data ->value 1655364167 
***********bytes 32,  count 4 ****************
index 0 , data ->value 803576 
index 1 , data ->value 2 
index 2 , data ->value 813576 
***********bytes 32,  count 4 ****************
index 0 , data ->value 3 
index 1 , data ->value 1655364167 
index 2 , data ->value 4 
***********bytes 32,  count 4 ****************
index 0 , data ->value 1655364167 
index 1 , data ->value 833575 
index 2 , data ->value 1655364167 
***********bytes 32,  count 4 ****************
index 0 , data ->value 843575 
index 1 , data ->value 6 
index 2 , data ->value 853575 
```