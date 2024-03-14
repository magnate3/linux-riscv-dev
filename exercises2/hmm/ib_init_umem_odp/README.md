# os

```
root@ubuntux86:# uname -a
Linux ubuntux86 5.13.0-39-generic #44~20.04.1-Ubuntu SMP Thu Mar 24 16:43:35 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
root@ubuntux86:# 
```

# range length
+ hmm_range_fault以PAGE_SHIFT（pagesize）为单位   
```
        for (addr = start; addr < end; addr = range.end) {
                range.start = addr;
                range.end = min(addr + (ARRAY_SIZE(pfns) << PAGE_SHIFT), end);

                ret = dmirror_range_fault(mm, &range);
                if (ret)
                        break;
        }
```
+ 以PAGE_SHIFT为单位
```
  for (pfn = (range->start >> PAGE_SHIFT);
             pfn < (range->end >> PAGE_SHIFT);
             pfn++, pfns++) {
                struct page *page;
                void *entry;

                /*
                 * Since we asked for hmm_range_fault() to populate pages,
                 * it shouldn't return an error entry on success.
                 */
                WARN_ON(*pfns & HMM_PFN_ERROR);
                WARN_ON(!(*pfns & HMM_PFN_VALID));

                page = hmm_pfn_to_page(*pfns);
                WARN_ON(!page);
                if(pfn == range->start >> PAGE_SHIFT)
                {
                   page_offset = range->start & ~PAGE_MASK;
                   addr =  page_address(page);
                   addr += page_offset;
                   memcpy(buf,addr,32);
                   pr_info("buf is %s \n", buf);
                }
```

用户态：   
```
 char * addr2 = (char *)malloc(getpagesize());
  memcpy(addr2,str,strlen(str)+1)
 read(fd, addr2, getpagesize());
```

在内核态获取addr2（range->hmm_pfns中第一个page）对应的内核地址
```
if(pfn == range->start >> PAGE_SHIFT)
                {
                   page_offset = range->start & ~PAGE_MASK;
                   addr =  page_address(page);
                   addr += page_offset;
                   memcpy(buf,addr,32);
                   pr_info("buf is %s \n", buf);
                }
```

+ read操作以一个pagesize页为单位   
read(fd, addr2, getpagesize());   

# 运行结果

```
root@ubuntux86:# ./mmap_test 
addr: 0x7fda2a7b7000 

Write/Read test ...
0x66616365
0x66616365
0x66616365
root@ubuntux86:# dmesg | tail -n 10
[24818.518753] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[24818.518759] nvme 0000:02:00.0:    [ 0] RxErr                 
[26106.516239] pcieport 0000:00:1d.0: AER: Corrected error received: 0000:02:00.0
[26106.517474] nvme 0000:02:00.0: PCIe Bus Error: severity=Corrected, type=Physical Layer, (Receiver ID)
[26106.517479] nvme 0000:02:00.0:   device [15b7:5006] error status/mask=00000001/0000e000
[26106.517484] nvme 0000:02:00.0:    [ 0] RxErr                 
[26343.509677] remap ret 0 
[26343.509774] do in  dmirror_do_fault  func
[26343.509778] buf is hello world 
[26343.509783] fault ret 0 
root@ubuntux86:#
```
内核输出：buf is hello world 日志和用户态的一致   

# ib_umem_odp_map_dma_and_lock

```
		ret = ib_umem_odp_map_dma_single_page(
				umem_odp, dma_index, hmm_pfn_to_page(range.hmm_pfns[pfn_index]),
				access_mask);
```