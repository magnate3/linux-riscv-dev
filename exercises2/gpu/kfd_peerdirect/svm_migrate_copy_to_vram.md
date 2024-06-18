Memory Manager
+ 存储区域   

SYSTEM域：位于系统内存，CPU可以直接访问，GPU不可访问   
GTT域：位于系统内存，CPU可以直接访问，GPU通过GART表访问（ring buffer就在这里）   
VRAM域：位于显存，CPU通过I/O映射访问，GPU可以直接访问，也称local,(顶点一般放这，纹理GTT和VRAM皆可)   
+  GPUVM:   
​ 是GPU提供的MMU功能，类似GART， 每一个GPU 进程有一份GPU页表，一般最多16个，在提交Command的时候GPU驱动会把进程对应的GPU页表设置到硬件寄存器中，在细节上GPU的地址分配和映射和CPU稍有不同。     和GART区别是，GART只支持一个地址空间，而GPUVM支持多个地址空间，用VMID来标识。   

​ GPUVM不仅可以映射系统内存还可以映射显存，也就是说将系统内存和显存放在统一的地址空间中管理。还可以映射为snooped/nonsnoop(cache/uncache system page)   

​ 在执行cmd buffer的时候，内核应该告诉engine使用使用的cmd buffer的VMID，VMID是在Submit时候动态分配的。   

​ GPUVM由1-2或1-5级页表表示，具体取决于chip,它也支持RWX属性或其他属性，比如加密和caching 属性  

​ 在AMD显卡中，VMID0是给内核驱动预留的，除了page table管理的apertures,vmid0还有其他apertures,有一个apertures用来直接访问VRAM，还有一个旧的AGP aperature仅仅将（memory）访问直接转发到system physical address(或当IOMMU存在时的IOVA)，（这个应该说的是旧的PA模式）这些aperature提供了对这些内存的直接访问，而没有页表的开销，VIMD0由KMD用于内存管理等任务   

​ GPU clients(engines),即每个APP都有自己唯一的地址空间，KMD管理每个进程的GPU VM page Table，当访问的无效页面时，会触发Page Fault    

+ 共享显存

一般在核显中常见，当作VRAM去处理，而不是GART，CPU需要IO映射才能访问，这种也叫UMA
 

# svm_migrate_copy_to_vram （migrate to gpu vram ）      

```
svm_migrate_copy_to_vram  --> 
svm_migrate_copy_memory_gart  --> amdgpu_copy_buffer
```


vram物理页    

+  amdgpu_res_first 获取cursor

+   migrate->dst[i] = svm_migrate_addr_to_pfn(adev, dst[i])   
```
	amdgpu_res_first(prange->ttm_res, ttm_res_offset,
			 npages << PAGE_SHIFT, &cursor);
	for (i = j = 0; i < npages; i++) {
		struct page *spage;

		dst[i] = cursor.start + (j << PAGE_SHIFT);
		migrate->dst[i] = svm_migrate_addr_to_pfn(adev, dst[i]);
```

#  gart table  





AGP(Accelerated Graphics Port)   

一种高速总线，允许图形卡从系统内存读数据，基于GART使不连续的内存在图形卡眼中作为连续处理，并使用DMA传输(还记着scatter-gather模式)。   

GART(Graphics Address Re-Mapping Table)  
我的理解就是IOMMU/SMMU之类的东西，外设的页表，然后外设可以访问不连续内存。   

GTT   
Global Graphics Translation Table，负责GPU虚拟地址空间到物理地址空间的映射，好像是intel来的   
 
 
# references

[openEuler Kernel 技术解读 | UADK框架介绍](https://ost.51cto.com/posts/15214)   