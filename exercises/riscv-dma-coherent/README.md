

# os
```
root@Ubuntu-riscv64:/boot# uname -a
Linux Ubuntu-riscv64 5.15.24-rt31 #10 SMP PREEMPT_RT Mon Jul 11 15:49:38 HKT 2022 riscv64 riscv64 riscv64 GNU/Linux
root@Ubuntu-riscv64:/boot# lscpu
Architecture:        riscv64
Byte Order:          Little Endian
CPU(s):              4
On-line CPU(s) list: 0-3
Thread(s) per core:  4
Core(s) per socket:  1
Socket(s):           1
L1d cache:           32 KiB
L1i cache:           32 KiB
L2 cache:            2 MiB
root@Ubuntu-riscv64:/boot# 
```


# dma_map_page_attrs
```
dma_addr_t dma_map_page_attrs(struct device *dev, struct page *page,
                size_t offset, size_t size, enum dma_data_direction dir,
                unsigned long attrs)
{
        const struct dma_map_ops *ops = get_dma_ops(dev);
        dma_addr_t addr;

        pr_info(" %s, %s, %p, %d,%d,%d",dev_driver_string(dev), dev_name(dev), ops, dma_map_direct(dev, ops), dev_is_dma_coherent(dev), attrs & DMA_ATTR_SKIP_CPU_SYNC);
        BUG_ON(!valid_dma_direction(dir));

        if (WARN_ON_ONCE(!dev->dma_mask))
                return DMA_MAPPING_ERROR;

        if (dma_map_direct(dev, ops) ||
            arch_dma_map_page_direct(dev, page_to_phys(page) + offset + size))
        {
                pr_info("call dma_direct_map_page: %s, %s",dev_driver_string(dev), dev_name(dev));
                addr = dma_direct_map_page(dev, page, offset, size, dir, attrs); // macb  execute
        }
        else
        {
                pr_info("call map_page: %s, %s",dev_driver_string(dev), dev_name(dev));
                addr = ops->map_page(dev, page, offset, size, dir, attrs);
        }
        debug_dma_map_page(dev, page, offset, size, dir, addr, attrs);

        return addr;
}
EXPORT_SYMBOL(dma_map_page_attrs);

```

```
root@Ubuntu-riscv64:~# dmesg  | tail -n 30
[  368.539848] call dma_direct_map_page: macb, 10090000.ethernet
[  368.547792]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.547809] call dma_direct_map_page: macb, 10090000.ethernet
[  368.547837]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.547851] call dma_direct_map_page: macb, 10090000.ethernet
[  368.548543]  xhci_hcd, 0000:04:00.0, 0000000000000000, 1,1,0
[  368.548560] call dma_direct_map_page: xhci_hcd, 0000:04:00.0
[  368.548571] software IO TLB: call swiotlb_tbl_map_single: xhci_hcd, 0000:04:00.0
[  368.555788]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.555804] call dma_direct_map_page: macb, 10090000.ethernet
[  368.555835]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.555849] call dma_direct_map_page: macb, 10090000.ethernet
[  368.563790]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.563808] call dma_direct_map_page: macb, 10090000.ethernet
[  368.563836]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.563850] call dma_direct_map_page: macb, 10090000.ethernet
[  368.571788]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.571805] call dma_direct_map_page: macb, 10090000.ethernet
[  368.571833]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.571846] call dma_direct_map_page: macb, 10090000.ethernet
[  368.579791]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.579808] call dma_direct_map_page: macb, 10090000.ethernet
[  368.579839]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.579853] call dma_direct_map_page: macb, 10090000.ethernet
[  368.580538]  xhci_hcd, 0000:04:00.0, 0000000000000000, 1,1,0
[  368.580555] call dma_direct_map_page: xhci_hcd, 0000:04:00.0
[  368.580565] software IO TLB: call swiotlb_tbl_map_single: xhci_hcd, 0000:04:00.0
[  368.587789]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
[  368.587806] call dma_direct_map_page: macb, 10090000.ethernet
[  368.587840]  macb, 10090000.ethernet, 0000000000000000, 1,1,0
root@Ubuntu-riscv64:~# 
```

## dma_direct_map_page
```
static inline dma_addr_t dma_direct_map_page(struct device *dev,
                struct page *page, unsigned long offset, size_t size,
                enum dma_data_direction dir, unsigned long attrs)
{
        phys_addr_t phys = page_to_phys(page) + offset;
        dma_addr_t dma_addr = phys_to_dma(dev, phys);

        if (is_swiotlb_force_bounce(dev))
                return swiotlb_map(dev, phys, size, dir, attrs); // macb not execute

        if (unlikely(!dma_capable(dev, dma_addr, size, true))) {
                if (swiotlb_force != SWIOTLB_NO_FORCE)
                        return swiotlb_map(dev, phys, size, dir, attrs);  // macb not execute

                dev_WARN_ONCE(dev, 1,
                             "DMA addr %pad+%zu overflow (mask %llx, bus limit %llx).\n",
                             &dma_addr, size, *dev->dma_mask, dev->bus_dma_limit);
                return DMA_MAPPING_ERROR;
        }

        if (!dev_is_dma_coherent(dev) && !(attrs & DMA_ATTR_SKIP_CPU_SYNC)) // // macb not execute
                arch_sync_dma_for_device(phys, size, dir);
        return dma_addr;
}
```

*** dma_map_single -->dma_map_page_attrs -->dma_direct_map_page -->swiotlb_map ***

# swiotlb_map


```
dma_addr_t swiotlb_map(struct device *dev, phys_addr_t paddr, size_t size,
                enum dma_data_direction dir, unsigned long attrs)
{
        phys_addr_t swiotlb_addr;
        dma_addr_t dma_addr;

        trace_swiotlb_bounced(dev, phys_to_dma(dev, paddr), size,
                              swiotlb_force);

        pr_info("call swiotlb_tbl_map_single: %s, %s",dev_driver_string(dev), dev_name(dev));
        swiotlb_addr = swiotlb_tbl_map_single(dev, paddr, size, size, dir,
                        attrs);
        if (swiotlb_addr == (phys_addr_t)DMA_MAPPING_ERROR)
                return DMA_MAPPING_ERROR;

        /* Ensure that the address returned is DMA'ble */
        dma_addr = phys_to_dma_unencrypted(dev, swiotlb_addr);
        if (unlikely(!dma_capable(dev, dma_addr, size, true))) {
                swiotlb_tbl_unmap_single(dev, swiotlb_addr, size, dir,
                        attrs | DMA_ATTR_SKIP_CPU_SYNC);
                dev_WARN_ONCE(dev, 1,
                        "swiotlb addr %pad+%zu overflow (mask %llx, bus limit %llx).\n",
                        &dma_addr, size, *dev->dma_mask, dev->bus_dma_limit);
                return DMA_MAPPING_ERROR;
        }

        if (!dev_is_dma_coherent(dev) && !(attrs & DMA_ATTR_SKIP_CPU_SYNC))
                arch_sync_dma_for_device(swiotlb_addr, size, dir);
        return dma_addr;
}
```