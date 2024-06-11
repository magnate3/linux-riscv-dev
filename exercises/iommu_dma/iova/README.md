


#   alloc_iova


+ 1  alloc_iova    

+ 2  dma_map_page   

+ 3  dma_unmap_page   

+ 4 __free_iova    

```
static int allocate_trash_buffer(struct ipu6_mmu *mmu)
{
	unsigned int n_pages = PHYS_PFN(PAGE_ALIGN(IPU6_MMUV2_TRASH_RANGE));
	struct iova *iova;
	unsigned int i;
	dma_addr_t dma;
	unsigned long iova_addr;
	int ret;

	/* Allocate 8MB in iova range */
	iova = alloc_iova(&mmu->dmap->iovad, n_pages,
			  PHYS_PFN(mmu->dmap->mmu_info->aperture_end), 0);
	if (!iova) {
		dev_err(mmu->dev, "cannot allocate iova range for trash\n");
		return -ENOMEM;
	}

	dma = dma_map_page(mmu->dmap->mmu_info->dev, mmu->trash_page, 0,
			   PAGE_SIZE, DMA_BIDIRECTIONAL);
	if (dma_mapping_error(mmu->dmap->mmu_info->dev, dma)) {
		dev_err(mmu->dmap->mmu_info->dev, "Failed to map trash page\n");
		ret = -ENOMEM;
		goto out_free_iova;
	}

	mmu->pci_trash_page = dma;

	/*
	 * Map the 8MB iova address range to the same physical trash page
	 * mmu->trash_page which is already reserved at the probe
	 */
	iova_addr = iova->pfn_lo;
	for (i = 0; i < n_pages; i++) {
		ret = ipu6_mmu_map(mmu->dmap->mmu_info, PFN_PHYS(iova_addr),
				   mmu->pci_trash_page, PAGE_SIZE);
		if (ret) {
			dev_err(mmu->dev,
				"mapping trash buffer range failed\n");
			goto out_unmap;
		}

		iova_addr++;
	}

	mmu->iova_trash_page = PFN_PHYS(iova->pfn_lo);
	dev_dbg(mmu->dev, "iova trash buffer for MMUID: %d is %u\n",
		mmu->mmid, (unsigned int)mmu->iova_trash_page);
	return 0;

out_unmap:
	ipu6_mmu_unmap(mmu->dmap->mmu_info, PFN_PHYS(iova->pfn_lo),
		       PFN_PHYS(iova_size(iova)));
	dma_unmap_page(mmu->dmap->mmu_info->dev, mmu->pci_trash_page,
		       PAGE_SIZE, DMA_BIDIRECTIONAL);
out_free_iova:
	__free_iova(&mmu->dmap->iovad, iova);
	return ret;
}
```
**demo2 iommu_map**

+ 1  __get_free_pages   

+ 2  alloc_iova

+ 3   iova_dma_addr   

+ 4  iommu_map

```
void *tegra_drm_alloc(struct tegra_drm *tegra, size_t size, dma_addr_t *dma)
{
	struct iova *alloc;
	void *virt;
	gfp_t gfp;
	int err;

	if (tegra->domain)
		size = iova_align(&tegra->carveout.domain, size);
	else
		size = PAGE_ALIGN(size);

	gfp = GFP_KERNEL | __GFP_ZERO;
	if (!tegra->domain) {
		/*
		 * Many units only support 32-bit addresses, even on 64-bit
		 * SoCs. If there is no IOMMU to translate into a 32-bit IO
		 * virtual address space, force allocations to be in the
		 * lower 32-bit range.
		 */
		gfp |= GFP_DMA;
	}

	virt = (void *)__get_free_pages(gfp, get_order(size));
	if (!virt)
		return ERR_PTR(-ENOMEM);

	if (!tegra->domain) {
		/*
		 * If IOMMU is disabled, devices address physical memory
		 * directly.
		 */
		*dma = virt_to_phys(virt);
		return virt;
	}

	alloc = alloc_iova(&tegra->carveout.domain,
			   size >> tegra->carveout.shift,
			   tegra->carveout.limit, true);
	if (!alloc) {
		err = -EBUSY;
		goto free_pages;
	}

	*dma = iova_dma_addr(&tegra->carveout.domain, alloc);
	err = iommu_map(tegra->domain, *dma, virt_to_phys(virt),
			size, IOMMU_READ | IOMMU_WRITE, GFP_KERNEL);
	if (err < 0)
		goto free_iova;

	return virt;

free_iova:
	__free_iova(&tegra->carveout.domain, alloc);
free_pages:
	free_pages((unsigned long)virt, get_order(size));

	return ERR_PTR(err);
}
```

```
struct iommu_dma_cookie *cookie = domain->iova_cookie;
struct iova_domain *iovad = &cookie->iovad;
```

# test1

iommu_map 采用固定的地址VBASE + j * PAGE_SIZE     
```
        /* IOMMU Mapping */
        for (j = 0; j < PAGE_NR; j++) {
                unsigned long pfn = page_to_pfn(pages[j]);

                r = iommu_map(domain, VBASE + j * PAGE_SIZE,
                                pfn << PAGE_SHIFT, PAGE_SIZE, IOMMU_READ);
                if (r) {
                        printk("System Error: Faild to iommu map.\n");
                        goto err_iommu_map;
                }
        }
```


```
./dpdk-devbind.py  -u 0000:05:00.0
[root@centos7 iommu]# rmmod hinic
[root@centos7 iommu]# insmod  iommu_test.ko 
[root@centos7 iommu]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
[root@centos7 iommu]# dmesg | tail -n 10
[  350.777470] ioctl_example close was called
[  840.746228] Goodbye, Linux kernel!
[171851.569810] hinic 0000:05:00.0: IO stopped
[171851.619520] hinic 0000:05:00.0 enp5s0: HINIC_INTF is DOWN
[171851.820353] hinic 0000:05:00.0: HiNIC driver - removed
[172006.877200] hinic 0000:06:00.0: IO stopped
[172006.929619] hinic 0000:06:00.0 enp6s0: HINIC_INTF is DOWN
[172007.057763] hinic 0000:06:00.0: HiNIC driver - removed
[172014.320314] pci iommu test successfully 
[172014.324387] pci iommu test successfully 
```

# iova 地址分配   

```
 else if (cmd == VFIO_IOMMU_MAP_DMA) {
                struct vfio_iommu_type1_dma_map map;
                uint32_t mask = VFIO_DMA_MAP_FLAG_READ |
                                VFIO_DMA_MAP_FLAG_WRITE;

                minsz = offsetofend(struct vfio_iommu_type1_dma_map, size);

                if (copy_from_user(&map, (void __user *)arg, minsz))
                        return -EFAULT;

                if (map.argsz < minsz || map.flags & ~mask)
                        return -EINVAL;

                return vfio_dma_do_map(iommu, &map);

```

```
 dma_addr_t iova = map->iova;
```


```

        dma->iova = iova;
        dma->vaddr = vaddr;
        dma->prot = prot;
        get_task_struct(current);
        dma->task = current;
        dma->pfn_list = RB_ROOT;

        /* Insert zero-sized and grow as we map chunks of it */
        vfio_link_dma(iommu, dma);

        /* Don't pin and map if container doesn't contain IOMMU capable domain*/
        if (!IS_IOMMU_CAP_DOMAIN_IN_CONTAINER(iommu))
                dma->size = size;
        else
                ret = vfio_pin_map_dma(iommu, dma, size);
```

+  vfio_pin_map_dma

```
              ret = vfio_iommu_map(iommu, iova + dma->size, pfn, npage,
                                     dma->prot);
                if (ret) {
                        vfio_unpin_pages_remote(dma, iova + dma->size, pfn,
                                                npage, true);
                        break;
                }
```

> ##  IOMMU_DOMAIN_DMA coredump (不支持IOMMU_DOMAIN_DMA)  

```
domain = test_iommu_domain_alloc(&pci_bus_type,IOMMU_DOMAIN_DMA);
```

```
static void __iommu_setup_dma_ops(struct device *dev, u64 dma_base, u64 size,
                                  const struct iommu_ops *ops)
{
        struct iommu_domain *domain;

        if (!ops)
                return;

        /*
         * The IOMMU core code allocates the default DMA domain, which the
         * underlying IOMMU driver needs to support via the dma-iommu layer.
         */
        domain = iommu_get_domain_for_dev(dev);

        if (!domain)
                goto out_err;

        if (domain->type == IOMMU_DOMAIN_DMA) {
                if (iommu_dma_init_domain(domain, dma_base, size, dev))
                        goto out_err;

                dev->dma_ops = &iommu_dma_ops;
        }

        return;

out_err:
         pr_warn("Failed to set up IOMMU for device %s; retaining platform DMA ops\n",
                 dev_name(dev));
}
```

```

[ 2609.453126] [<ffff000008566090>] alloc_iova+0xac/0x1c8
[ 2609.458240] [<ffff000008566f00>] alloc_iova_fast+0x78/0x28c
[ 2609.463792] [<ffff0000010c01cc>] pva_dma_alloc_and_map_at.constprop.8+0x198/0x4cc [iommu_test2]
[ 2609.472449] [<ffff0000010c05ac>] PCIe_probe+0xac/0x168 [iommu_test2]
[ 2609.478774] [<ffff000008475fa8>] local_pci_probe+0x48/0xb0
[ 2609.484234] [<ffff0000080ed870>] work_for_cpu_fn+0x24/0x34
[ 2609.489696] [<ffff0000080f13ac>] process_one_work+0x174/0x3b0
[ 2609.495414] [<ffff0000080f180c>] worker_thread+0x224/0x48c
[ 2609.500875] [<ffff0000080f8638>] kthread+0x10c/0x138
[ 2609.505818] [<ffff000008084f54>] ret_from_fork+0x10/0x18
```

# test2


```
[root@centos7 boot]# grep CONFIG_IOMMU_IOVA config-4.14.0-115.el7a.0.1.aarch64 
CONFIG_IOMMU_IOVA=y
[root@centos7 boot]# 
```

```
[root@centos7 iommu]# ip l set enp5s0 down
[root@centos7 iommu]# ./dpdk-devbind.py  -u 0000:05:00.0
Warning: no supported DPDK kernel modules are loaded
[root@centos7 iommu]# insmod  iommu_test2.ko 
[root@centos7 iommu]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
[root@centos7 iommu]# dmesg | tail -n 10
[   35.720157] bridge: filtering via arp/ip/ip6tables is no longer available by default. Update your scripts to load br_netfilter if you need this.
[   36.492390] Netfilter messages via NETLINK v0.30.
[   36.502525] ip_set: protocol 6
[ 6225.170756] hinic 0000:05:00.0 enp5s0: set rx mode work
[ 6225.531487] hinic 0000:05:00.0: IO stopped
[ 6225.684676] hinic 0000:05:00.0 enp5s0: HINIC_INTF is DOWN
[ 6245.491796] hinic 0000:05:00.0: HiNIC driver - removed
[ 6255.628949] iommu_test2: loading out-of-tree module taints kernel.
[ 6255.635169] iommu_test2: module verification failed: signature and/or required key missing - tainting kernel
[ 6257.792685] test_valid_domain cookie is NULL 
[root@centos7 iommu]# 
```
`test_valid_domain cookie is NULL`     
```

/**
 * Allocate a dma buffer and map it to a specified iova
 * Return valid cpu virtual address on success or NULL on failure
 */
static void *pva_dma_alloc_and_map_at(struct device *dev, size_t size,
				      dma_addr_t iova, gfp_t flags,
				      unsigned long attrs)
{
	struct iommu_domain *domain;
	unsigned long shift, pg_size, mp_size;
	dma_addr_t tmp_iova, offset;
	phys_addr_t pa, pa_new;
	void *cpu_va;
	int ret;

	domain = iommu_get_domain_for_dev(dev);
	if (!domain) {
		dev_err(dev, "IOMMU domain not found");
		return NULL;
	}

	shift = __ffs(domain->pgsize_bitmap);
	pg_size = 1UL << shift;
	mp_size = pg_size;

	/* Reserve iova range */
	tmp_iova = iommu_dma_alloc_iova(dev, size, iova + size - pg_size);
	if (tmp_iova != iova) {
		dev_err(dev, "failed to reserve iova at 0x%llx size 0x%lx\n",
			iova, size);
		return NULL;
	}

	/* Allocate a memory first and get a tmp_iova */
	cpu_va = dma_alloc_attrs(dev, size, &tmp_iova, flags, attrs);
	if (!cpu_va)
		goto fail_dma_alloc;

	/* Use tmp_iova to remap non-contiguous pages to the desired iova */
	for (offset = 0; offset < size; offset += mp_size) {
		dma_addr_t cur_iova = tmp_iova + offset;

		mp_size = pg_size;
		pa = iommu_iova_to_phys(domain, cur_iova);
		/* Checking if next physical addresses are contiguous */
		for ( ; offset + mp_size < size; mp_size += pg_size) {
			pa_new = iommu_iova_to_phys(domain, cur_iova + mp_size);
			if (pa + mp_size != pa_new)
				break;
		}

		/* Remap the contiguous physical addresses together */
		ret = iommu_map(domain, iova + offset, pa, mp_size,
				IOMMU_READ | IOMMU_WRITE);
		if (ret) {
			dev_err(dev, "failed to map pa %llx va %llx size %lx\n",
				pa, iova + offset, mp_size);
			goto fail_map;
		}

		/* Verify if the new iova is correctly mapped */
		if (pa != iommu_iova_to_phys(domain, iova + offset)) {
			dev_err(dev, "mismatched pa 0x%llx <-> 0x%llx\n",
				pa, iommu_iova_to_phys(domain, iova + offset));
			goto fail_map;
		}
	}

	/* Unmap and free the tmp_iova since target iova is linked */
	iommu_unmap(domain, tmp_iova, size);
	iommu_dma_free_iova(dev, tmp_iova, size);

	return cpu_va;

fail_map:
	iommu_unmap(domain, iova, offset);
	dma_free_attrs(dev, size, cpu_va, tmp_iova, attrs);
fail_dma_alloc:
	iommu_dma_free_iova(dev, iova, size);

	return NULL;
}
```