
# references   

[参考链接](http://htmlpreview.github.io/?https://github.com/joyxu/joyxu.github.com/blob/d731e24f896353b17979030a924bdc3928445b98/2023/12/11/pcie-p2p/index.html#L389)

[参考链接2](https://joyxu.github.io/2023/12/11/pcie-p2p/)


[nvidia-GPU-uvm-驱动分析](https://joyxu.github.io/2022/05/24/nvidia-GPU-uvm-%E9%A9%B1%E5%8A%A8%E5%88%86%E6%9E%90/)   

# 原理
+ mmap怎么从这个bar分配内存呢？     
由于bar是作为zone_device注册的,zone_device仍然是基于sparsemem_vmemmap的，它提供了基于struct page的服务，包括pfn_to_page,page_to_pfn和get_user_pages等。
但它的分配和释放并不是当做普通内存来管理的，而是通过generic purpose allocator来管理分配的，并会针对该设备创建一个内存池。
当map的时候，会调用p2pmem_alloc_mmap，并通过vm_insert_page把这个地址映射到用户态的va。    

+ 对端设备怎么用这个地址呢？   
另外一个设备拿到这个地址之后，怎么找到这个地址对应的设备的bar offset呢？
这就有回到类似dma的作用，把dma地址映射到内存的物理地址，而在p2p场景中，则是把dma地址，转换成bar offset。
要理解这个细节，可以看pci_p2pmem_virt_to_bus，它会调用gen_pool_virt_to_phys，返回pcie bar的offset，于是对端设备实际上配置的是pcie的地址空间。
于是当对端设备针对这个地址发起访问的时候，就没有上pcie root port，而是直接到该设备了。    

注意：    
当前内核一旦开启了P2P，默认会关掉ACS，也可能会影响到SVA。   
原则上ATS/ACS和P2P是有些冲突的，因为当ATS打开之后，设备发出的PCIe TLP报文会声称该报文的地址是否是翻译过的。     
如果没有翻译，则先路由到RC的TA处进行地址翻译;如果翻译过，则直接使用，绕过了IOMMU的隔离，直接访问这个物理地址了，导致安全风险。
比如说，开启了P2P和ATS以后，同一个PCIe Switch后的所有EP设备，必须都分给同一个虚拟机，不然分给不同虚拟机的话，可以从这个PCIe设备的另外一个Function攻击到其它的虚拟机。
于是呢，就引入了ACS（访问控制）来决定一个TLP是否能正常路由，还是被阻塞或者重定向。
可以参考这个patch的评论PCI/P2PDMA: Clear ACS P2P flags for all devices behind switches。
也可以参考SBSA的测试用例SBSA PCIe ATS test。


#  ZONE_DEVICE
The ZONE_DEVICE facility builds upon SPARSEMEM_VMEMMAP to offer struct page mem_map services for device driver identified physical address ranges. The “device” aspect of ZONE_DEVICE relates to the fact that the page objects for these address ranges are never marked online, and that a reference must be taken against the device, not just the page to keep the memory pinned for active use. ZONE_DEVICE, via *devm_memremap_pages()*, performs just enough memory hotplug to turn on *pfn_to_page(), page_to_pfn()*, and *get_user_pages()* service for the given range of pfns. Since the page reference count never drops below 1 the page is never tracked as free memory and the page’s struct list_head lru space is repurposed for back referencing to the host device / driver that mapped the memory.    

While SPARSEMEM presents memory as a collection of sections, optionally collected into memory blocks, ZONE_DEVICE users have a need for smaller granularity of populating the mem_map. Given that ZONE_DEVICE memory is never marked online it is subsequently never subject to its memory ranges being exposed through the sysfs memory hotplug api on memory block boundaries. The implementation relies on this lack of user-api constraint to allow sub-section sized memory ranges to be specified to *arch_add_memory()*, the top-half of memory hotplug. Sub-section support allows for 2MB as the cross-arch common alignment granularity for *devm_memremap_pages()*.   

The users of ZONE_DEVICE are:   

pmem: Map platform persistent memory to be used as a direct-I/O target via DAX mappings.  
hmm: Extend ZONE_DEVICE with ->page_fault() and ->page_free() event callbacks to allow a device-driver to coordinate memory management events related to device-memory, typically GPU memory. See Documentation/vm/hmm.rst.   
p2pdma: Create struct page objects to allow peer devices in a PCI/-E topology to coordinate direct-DMA operations between themselves, i.e. bypass host memory.   