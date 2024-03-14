

```
struct device {
#ifdef CONFIG_DMA_OPS
    const struct dma_map_ops *dma_ops;
#endif
}
[root@centos7 boot]# grep CONFIG_DMA_OPS  config-4.14.0-115.el7a.0.1.aarch64
[root@centos7 boot]# grep CONFIG_IOMMU_DMA  config-4.14.0-115.el7a.0.1.aarch64
CONFIG_IOMMU_DMA=y
[root@centos7 boot]# 
```


# insmod  pci_test2.ko 

```
[root@centos7 iommu]# insmod  pci_test2.ko 
[root@centos7 iommu]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
```

```
[root@centos7 boot]# grep arm_smmu_domain_alloc System.map-4.14.0-115.el7a.0.1.aarch64
ffff000008568b18 t arm_smmu_domain_alloc
ffff00000856b118 t arm_smmu_domain_alloc
[root@centos7 boot]# grep arm_smmu_map System.map-4.14.0-115.el7a.0.1.aarch64
ffff000008567a74 t arm_smmu_map
ffff00000856a788 t arm_smmu_map
[root@centos7 boot]# grep arm_smmu_ops System.map-4.14.0-115.el7a.0.1.aarch64
ffff000008ece920 d arm_smmu_ops
ffff000008ecec38 d arm_smmu_ops
[root@centos7 boot]# 
```

```
[223054.288379] ***************** pci iommu bus number 5, slot 0, devfn 0  ************ 
[223054.296173] group id 25 and group name (null) 
[223054.300694] dbg_added iommu_domain_alloc ok
[223054.304946] iommu_domain_ops: ffff000008ecec38 and iommu_ops->domain_alloc ffff00000856b118, iommu_ops->map ffff00000856a788 
[223054.316300] dbg_added iommu_attach_device ok
[223054.322661] dbg_added iommu_map ok
[223054.326133]  phys_addr: 0000002100000000, my_iova: 0000000050100000 
[223054.342817] ***************** pci iommu bus number 6, slot 0, devfn 0  ************ 
[223054.350614] group id 31 and group name (null) 
[223054.355127] dbg_added iommu_domain_alloc ok
[223054.359383] iommu_domain_ops: ffff000008ecec38 and iommu_ops->domain_alloc ffff00000856b118, iommu_ops->map ffff00000856a788 
[223054.370733] dbg_added iommu_attach_device ok
[223054.377044] dbg_added iommu_map ok
[223054.380516]  phys_addr: 0000002100000000, my_iova: 0000000050100000 
[root@centos7 iommu]#
```




# grep arm_smmu  System.map-4.14.0-115.el7a.0.1.aarch64
```
[root@centos7 boot]# grep ffff00000856b118  System.map-4.14.0-115.el7a.0.1.aarch64
ffff00000856b118 t arm_smmu_domain_alloc
[root@centos7 boot]# grep arm_smmu  System.map-4.14.0-115.el7a.0.1.aarch64
ffff0000085676a0 t arm_smmu_tlb_inv_range_nosync
ffff0000085677a4 t arm_smmu_tlb_inv_vmid_nosync
ffff0000085677d8 t arm_smmu_write_context_bank
ffff000008567964 t arm_smmu_write_s2cr
ffff0000085679f4 t arm_smmu_write_sme
ffff000008567a74 t arm_smmu_map
ffff000008567ad0 t arm_smmu_unmap
ffff000008567b18 t arm_smmu_capable
ffff000008567b5c t arm_smmu_match_node
ffff000008567b94 t arm_smmu_domain_get_attr
ffff000008567c00 t arm_smmu_of_xlate
ffff000008567cbc t arm_smmu_put_resv_regions
ffff000008567d0c t arm_smmu_get_resv_regions
ffff000008567d90 t arm_smmu_domain_set_attr
ffff000008567e34 t __arm_smmu_get_pci_sid
ffff000008567f24 t arm_smmu_global_fault
ffff000008568030 t arm_smmu_context_fault
ffff00000856810c t arm_smmu_device_remove
ffff000008568188 t arm_smmu_device_shutdown
ffff0000085681b4 t arm_smmu_device_cfg_probe
ffff00000856885c t arm_smmu_iova_to_phys
ffff000008568a64 t arm_smmu_domain_free
ffff000008568b18 t arm_smmu_domain_alloc
ffff000008568c08 t arm_smmu_free_sme.isra.12
ffff000008568cac t arm_smmu_add_device
ffff000008569264 t arm_smmu_attach_dev
ffff0000085698f0 t arm_smmu_bus_init
ffff0000085699a0 t arm_smmu_legacy_bus_init
ffff0000085699cc t arm_smmu_remove_device
ffff000008569b40 t arm_smmu_device_group
ffff000008569c04 t __arm_smmu_tlb_sync.isra.19
ffff000008569ca4 t arm_smmu_tlb_sync_global
ffff000008569d38 t arm_smmu_device_reset
ffff000008569f1c t arm_smmu_pm_resume
ffff000008569f4c t arm_smmu_tlb_sync_vmid
ffff000008569f78 t arm_smmu_tlb_inv_context_s2
ffff000008569fb4 t arm_smmu_tlb_sync_context
ffff00000856a064 t arm_smmu_tlb_inv_context_s1
ffff00000856a0b8 t arm_smmu_device_probe
ffff00000856a728 t arm_smmu_cmdq_sync_handler
ffff00000856a744 t arm_smmu_capable
ffff00000856a788 t arm_smmu_map
ffff00000856a7e4 t arm_smmu_unmap
ffff00000856a82c t arm_smmu_iova_to_phys
ffff00000856a884 t arm_smmu_match_node
ffff00000856a8bc t arm_smmu_domain_get_attr
ffff00000856a928 t arm_smmu_write_msi_msg
ffff00000856a9b4 t arm_smmu_of_xlate
ffff00000856a9f0 t arm_smmu_put_resv_regions
ffff00000856aa40 t arm_smmu_get_resv_regions
ffff00000856aac4 t arm_smmu_domain_set_attr
ffff00000856ab68 t arm_smmu_device_group
ffff00000856abb8 t arm_smmu_cmdq_build_cmd
ffff00000856ada8 t arm_smmu_init_one_queue
ffff00000856aec0 t arm_smmu_bitmap_alloc
ffff00000856af44 t arm_smmu_domain_finalise_s2
ffff00000856af9c t arm_smmu_domain_finalise_s1
ffff00000856b070 t arm_smmu_domain_free
ffff00000856b118 t arm_smmu_domain_alloc
ffff00000856b1e8 t arm_smmu_free_msis
ffff00000856b304 t arm_smmu_cmdq_issue_cmd
ffff00000856b544 t arm_smmu_sync_ste_for_sid
ffff00000856b5d0 t arm_smmu_write_strtab_ent
ffff00000856b7d0 t arm_smmu_install_ste_for_dev
ffff00000856b8b0 t arm_smmu_remove_device
ffff00000856b958 t arm_smmu_init_bypass_stes
ffff00000856b9e0 t arm_smmu_add_device
ffff00000856bc00 t arm_smmu_tlb_sync
ffff00000856bc58 t arm_smmu_tlb_inv_range_nosync
ffff00000856bd1c t arm_smmu_tlb_inv_context
ffff00000856be04 t arm_smmu_evtq_thread
ffff00000856bf2c t arm_smmu_priq_thread
ffff00000856c140 t arm_smmu_combined_irq_thread
ffff00000856c18c t arm_smmu_attach_dev
ffff00000856c4b4 t arm_smmu_write_reg_sync.isra.20
ffff00000856c55c t arm_smmu_device_disable
ffff00000856c5b0 t arm_smmu_device_remove
ffff00000856c5e0 t arm_smmu_device_shutdown
ffff00000856c60c t arm_smmu_gerror_handler
ffff00000856c904 t arm_smmu_combined_irq_handler
ffff00000856c93c t arm_smmu_device_probe
ffff0000089556c0 r arm_smmu_s1_tlb_ops
ffff0000089556d8 r arm_smmu_s2_tlb_ops_v2
ffff0000089556f0 r arm_smmu_s2_tlb_ops_v1
ffff000008955720 r arm_smmu_pm_ops
ffff0000089557d8 r arm_smmu_of_match
ffff000008955e00 r arm_smmu_gather_ops
ffff000008955e30 r arm_smmu_options
ffff000008955e60 r arm_smmu_of_match
ffff000008c42870 t arm_smmu_v3_count_resources
ffff000008c428c8 t arm_smmu_v3_is_coherent
ffff000008c428f4 t arm_smmu_count_resources
ffff000008c42920 t arm_smmu_is_coherent
ffff000008c429d0 t arm_smmu_init_resources
ffff000008c42a98 t arm_smmu_v3_init_resources
ffff000008c42c34 t arm_smmu_v3_set_proximity
ffff000008c48638 t arm_smmu_driver_init
ffff000008c48664 t arm_smmu_driver_init
ffff000008c64878 t arm_smmu_driver_exit
ffff000008c64898 t arm_smmu_driver_exit
ffff000008d3ba90 t iort_arm_smmu_v3_cfg
ffff000008d3bac0 t iort_arm_smmu_cfg
ffff000008d41908 t __of_table_arm_smmuv2
ffff000008d419d0 t __of_table_arm_smmuv1
ffff000008d41a98 t __of_table_arm_smmuv3
ffff000008d44f80 t __initcall_arm_smmu_driver_init6
ffff000008d44f88 t __initcall_arm_smmu_driver_init6
ffff000008d451f8 t __initcall_arm_smmu_legacy_bus_init6s
ffff000008ece920 d arm_smmu_ops
ffff000008ece9f8 d arm_smmu_driver
ffff000008eceaf0 d arm_smmu_msi_cfg
ffff000008eceb38 d arm_smmu_driver
ffff000008ecec38 d arm_smmu_ops
[root@centos7 boot]# 
```