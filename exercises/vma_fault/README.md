
#  bash simple_load 
```
[root@centos7 simple]# bash simple_load 
[root@centos7 simple]# cat /proc/devices | grep simple
241 simple
[root@centos7 simple]# ls /dev/simple
ls: cannot access /dev/simple: No such file or directory
```

```
[root@centos7 simple]# ls /dev/simple*
/dev/simplen  /dev/simpler
[root@centos7 simple]# cat 
```
#   simple_remap_mmap and  simple_remap_vm_ops

```
static int simple_remap_mmap(struct file *filp, struct vm_area_struct *vma)
{
    // no physical address
	if (remap_pfn_range(vma, vma->vm_start, vma->vm_pgoff,
			    vma->vm_end - vma->vm_start,
			    vma->vm_page_prot))
		return -EAGAIN;

	vma->vm_ops = &simple_remap_vm_ops;
	simple_vma_open(vma);
	return 0;
}
```
# simple_nopage_mmap and simple_nopage_vm_ops
```
static int simple_nopage_mmap(struct file *filp, struct vm_area_struct *vma)
{
	unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;

	if (offset >= __pa(high_memory) || (filp->f_flags & O_SYNC))
		vma->vm_flags |= VM_IO;
	vma->vm_flags |= VM_RESERVED;

	vma->vm_ops = &simple_nopage_vm_ops;
	simple_vma_open(vma);
	return 0;
}

```


# fault.c 
```
 open("/dev/simplen", O_RDWR);
 simple_setup_cdev(SimpleDevs + 1, 1, &simple_fault_ops)
 /map_addr1 = mmap(NULL, 4096 * 6, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 0);
 ```
 
 ```
 [root@centos7 simple]# ./fault 
Open done!
mmap done: 0xffff80f30000
Bus error
[root@centos7 simple]# 
 ```
 
 
```
root@centos7 simple]# dmesg | tail -n 40
[165354.625431] Simple VMA open, virt ffffbe460000, phys 0
[165354.630654] ---- fault, off 0 phys 0
[165354.634301] VA is ffff800000000000
[165354.637773] Page at ffff7fe000000000
[165354.641425] ----invalid page frame  //////////////////////////////
[165354.641429] fault[36756]: unhandled level 3 translation fault (7) at 0xffffbe460000, esr 0x92000007, in fault[400000+10000]
[165354.656165] CPU: 97 PID: 36756 Comm: fault Tainted: G        W  OE  ------------   4.14.0-115.el7a.0.1.aarch64 #1
[165354.666470] Hardware name: Huawei TaiShan 200 (Model 2280)/BC82AMDD, BIOS 1.08 12/14/2019
[165354.674701] task: ffffa05fc9ee3100 task.stack: ffff00002e960000
[165354.680689] PC is at 0x4008b8
[165354.683730] LR is at 0x400898
[165354.686771] pc : [<00000000004008b8>] lr : [<0000000000400898>] pstate: 80000000
[165354.694226] sp : 0000fffffd8fa450
[165354.697615] x29: 0000fffffd8fa450 x28: 0000000000000000 
[165354.702995] x27: 0000000000000000 x26: 0000000000000000 
[165354.708370] x25: 0000000000000000 x24: 0000000000000000 
[165354.713750] x23: 0000000000000000 x22: 0000000000000000 
[165354.719129] x21: 0000000000400680 x20: 0000000000000000 
[165354.724505] x19: 0000000000000000 x18: 0000fffffd8fa220 
[165354.729885] x17: 0000ffffbe4f10a8 x16: 0000ffffbe5f0030 
[165354.735261] x15: 00000000001915d7 x14: 0000ffffbe65ffa8 
[165354.740640] x13: 000000000000000f x12: 0000000000000090 
[165354.746014] x11: 0000000090000000 x10: 00000000ffffffff 
[165354.751394] x9 : 0000ffffbe5f15b0 x8 : 0000000000000040 
[165354.756769] x7 : 0000000000000000 x6 : 0000ffffbe4baf24 
[165354.762148] x5 : 0000000000000bd0 x4 : 00000000004009dd 
[165354.767523] x3 : 0000000000000000 x2 : 0000000000000001 
[165354.772903] x1 : 0000ffffbe460000 x0 : 0000ffffbe460000 
[165354.778324] Simple VMA close.
```

#  fault.c  demo2 

```
map_addr1 = mmap(NULL, 4096 * 6, PROT_READ, MAP_PRIVATE | MAP_FILE, fd, 4096);
```
```
[root@centos7 simple]# ./fault 
Open done!
map1 fail: Invalid argument
```

# remap
```
[root@centos7 simple]# gcc remap.c  -o remap
[root@centos7 simple]# ./remap 
open fail: No such file or directory
[root@centos7 simple]# bash simple_load 
[root@centos7 simple]# ./remap 
Open done!
mmap done: 0xffff96dc0000
----------  0
----------  0
----------  0
----------  0
----------  0
----------  0
----------  0
----------  0
----------  0
----------  0
[root@centos7 simple]# dmesg | tail -n 10
[163733.989966] x9 : 0000ffff87f115b0 x8 : 0000000000000040 
[163733.995341] x7 : 0000000000000000 x6 : 0000ffff87ddaf24 
[163734.000720] x5 : 0000000000000bd0 x4 : 00000000004009dd 
[163734.006096] x3 : 0000000000000000 x2 : 0000000000000001 
[163734.011476] x1 : 0000ffff87d80000 x0 : 0000ffff87d80000 
[163734.016902] Simple VMA close.
[164669.945844] ---- simple_remap_mmap 
[164669.949411] ---- simple_remap_mmap and call  simple_vma_open
[164669.955144] Simple VMA open, virt ffff96dc0000, phys 0  //////////// phys 0
[164669.960470] Simple VMA close.
[root@centos7 simple]# 
```

# 正确的test

```
 ret = posix_memalign((void **)&ptr, pagesize, pagesize);
 memcpy(ptr, str, strlen(str) +1);
 phyaddr = mem_virt2phy(ptr);
 addr = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, phyaddr);
```

通过 mmap传递物理地址,也就是 vma->vm_pgoff    
```
[root@centos7 simple]# gcc mmap_test.c  -o mmap_test
[root@centos7 simple]# ./mmap_test 
virt addr 0xffffa5be0000, phy addr of ptr  0x5fea170000 
addr: 0xffffa5bc0000 
Zero page frame number
mmap virt addr 0xffffa5bc0000, phy addr of ptr  0x0 
buf is: hello krishna 

Write/Read test ...
0x66616365
```
+  buf is: hello krishna 实现了共享内存      
+ 可以对mmap virt addr 0xffffa5bc0000进行test_write_read，但是物理地址是0    

# reference
https://github.com/blue119/kernel_user_space_interfaces_example    
https://github.com/ljrcore/linuxmooc/blob/bdaf02620e55bf06e9c84b72afb8ff47e7384447/%E7%B2%BE%E5%BD%A9%E6%96%87%E7%AB%A0/%E6%96%B0%E6%89%8B%E4%B8%8A%E8%B7%AF%EF%BC%9ALinux%E5%86%85%E6%A0%B8%E4%B9%8B%E6%B5%85%E8%B0%88%E5%86%85%E5%AD%98%E5%AF%BB%E5%9D%80.md    