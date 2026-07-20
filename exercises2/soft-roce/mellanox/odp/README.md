

# RDMA - ODP 


[RDMA - ODP按需分页设计原理-优点-源码浅析](https://cloud.tencent.com/developer/article/2428026)    
IBV_ACCESS_ON_DEMAND   
```
rc_pingpong.c
...
pp_init_ctx
    if (use_odp)
        const uint32_t rc_caps_mask = IBV_ODP_SUPPORT_SEND | IBV_ODP_SUPPORT_RECV
        ibv_query_device_ex(ctx->context, NULL, &attrx)
        if (!(attrx.odp_caps.general_caps & IBV_ODP_SUPPORT) ||
        attrx.odp_caps.general_caps & IBV_ODP_SUPPORT_IMPLICIT
        access_flags |= IBV_ACCESS_ON_DEMAND
``` 

linux-kernel(参考mlx5)
```
kernel, drivers/infiniband/hw/mlx5/odp.c

module_init(mlx5_ib_init);
    mlx5_ib_odp_init -> IB/mlx5：添加隐式 MR 支持，添加隐式 MR，覆盖整个用户地址空间。MR 实现为由 1GB 直接 MR 组成的间接 KSM MR。页面和直接 MR 由 ODP 添加/删除到 MR
    mlx5_imr_ksm_entries = BIT_ULL(get_order(TASK_SIZE) - MLX5_IMR_MTT_BITS)

ibv_reg_mr -> 注册内存时带上ODP标志
.reg_user_mr = mlx5_ib_reg_user_mr
    if (access_flags & IB_ACCESS_ON_DEMAND)
        return create_user_odp_mr(pd, start, length, iova, access_flags, udata)
            struct ib_umem_odp *odp
            if (!IS_ENABLED(CONFIG_INFINIBAND_ON_DEMAND_PAGING))
            mlx5r_odp_create_eq(dev, &dev->odp_pf_eq)
                INIT_WORK(&eq->work, mlx5_ib_eq_pf_action)
                    mempool_refill
                        mempool_free(mempool_alloc(pool, GFP_KERNEL), pool)
                    mlx5_ib_eq_pf_process
                eq->pool = mempool_create_kmalloc_pool(MLX5_IB_NUM_PF_DRAIN, sizeof(struct mlx5_pagefault))
                eq->wq = alloc_workqueue("mlx5_ib_page_fault", WQ_HIGHPRI | WQ_UNBOUND | WQ_MEM_RECLAIM, MLX5_NUM_CMD_EQE)
                eq->irq_nb.notifier_call = mlx5_ib_eq_pf_int
                    mlx5_ib_eq_pf_int -> net/mlx5：将中断处理程序更改为调用链式通知程序，多个 EQ 可能会在后续补丁中共享同一个 IRQ。EQ 将注册到原子链式通知程序，而不是直接调用 IRQ 处理程序。不使用 Linux 内置共享 IRQ，因为它会强制调用者在调用 free_irq() 之前禁用 IRQ 并清除关联。此补丁是分离 IRQ 和 EQ 逻辑的第一步
                        mlx5_ib_eq_pf_process
                            INIT_WORK(&pfault->work, mlx5_ib_eqe_pf_action)
                                mlx5_ib_pfault
                                    mlx5_ib_page_fault_resume
                                        MLX5_SET(page_fault_resume_in, in, opcode, MLX5_CMD_OP_PAGE_FAULT_RESUME)
                eq->core = mlx5_eq_create_generic(dev->mdev, &param) -> net/mlx5：将 IRQ 请求/释放与 EQ 生命周期分开，不再在 EQ 创建时请求 IRQ，而是在 EQ 表创建之前请求 IRQ。不再在 EQ 销毁后释放 IRQ，而是在 eq 表销毁后释放 IRQ
                    create_async_eq(dev, eq, param)
                        struct mlx5_eq_table *eq_table
                        create_map_eq(dev, eq, param)
                            u8 log_eq_size = order_base_2(param->nent + MLX5_NUM_SPARE_EQE)
                            INIT_RADIX_TREE(&cq_table->tree, GFP_ATOMIC)
                            mlx5_frag_buf_alloc_node(dev, wq_get_byte_sz(log_eq_size, log_eq_stride), &eq->frag_buf, dev->priv.numa_node) -> net/mlx5e：实现碎片工作队列 (WQ)，添加新类型的 struct mlx5_frag_buf，用于分配碎片缓冲区而不是连续缓冲区，并使完成队列 (CQ) 使用它，因为它们很大（Striding RQ 中每个 CQ 的默认值为 2MB）
                            mlx5_init_fbc
                            init_eq_buf
                            mlx5_irq_get_index
                            mlx5_fill_page_frag_array
                            MLX5_SET(create_eq_in, in, opcode, MLX5_CMD_OP_CREATE_EQ)
                            mlx5_cmd_exec(dev, in, inlen, out, sizeof(out))
                            mlx5_debug_eq_add(dev, eq)
                mlx5_eq_enable(dev->mdev, eq->core, &eq->irq_nb)
            if (!start && length == U64_MAX) -> start = 0 and max len
                mlx5_ib_alloc_implicit_mr(to_mpd(pd), access_flags)
                    ib_init_umem_odp
                        mmu_interval_notifier_insert -> 先注册, 再通过 hmm_range_fault 填充, 参考异构内存管理HMM: https://blog.csdn.net/Rong_Toa/article/details/117910321
            if (!mlx5r_umr_can_load_pas(dev, length))
            odp = ib_umem_odp_get(&dev->ib_dev, start, length, access_flags, &mlx5_mn_ops)
                ib_init_umem_odp(umem_odp, ops)
            mr = alloc_cacheable_mr(pd, &odp->umem, iova, access_flags)
                page_size = mlx5_umem_dmabuf_default_pgsz(umem, iova)
                or mlx5_umem_find_best_pgsz
                    ib_umem_find_best_pgsz
                rb_key.ats = mlx5_umem_needs_ats(dev, umem, access_flags)
                ent = mkey_cache_ent_from_rb_key(dev, rb_key) -> RDMA/mlx5：引入 mlx5r_cache_rb_key，从使用 mkey 顺序切换到使用新结构作为缓存条目 RB 树的键。该键是 UMR 操作无法修改的所有 mkey 属性。使用此键定义缓存条目并搜索和创建缓存 mkey
                mr = reg_create(pd, umem, iova, access_flags, page_size, false) -> no cache
                    mr = kzalloc(sizeof(*mr), GFP_KERNEL)
                    mr->page_shift = order_base_2(page_size)
                    mlx5_ib_create_mkey(dev, &mr->mmkey, in, inlen)
                        mlx5_core_create_mkey
                            MLX5_SET(create_mkey_in, in, opcode, MLX5_CMD_OP_CREATE_MKEY)
                    set_mr_fields(dev, mr, umem->length, access_flags, iova)
                mr = _mlx5_mr_cache_alloc(dev, ent, access_flags)
                    if (!ent->mkeys_queue.ci)
                        create_cache_mkey(ent, &mr->mmkey.key)
                            mlx5_core_create_mkey -> 创建缓存
                    else
                        mr->mmkey.key = pop_mkey_locked(ent) -> 从缓存中获取
                            last_page = list_last_entry(&ent->mkeys_queue.pages_list, struct mlx5_mkeys_page, list)
                            ent->mkeys_queue.ci--
                        queue_adjust_cache_locked
                set_mr_fields(dev, mr, umem->length, access_flags, iova)
            xa_init(&mr->implicit_children)
            mlx5r_store_odp_mkey(dev, &mr->mmkey) -> RDMA/mlx5：从 ODP 流中清除 synchronize_srcu()，从 ODP 流中清除 synchronize_srcu()，因为作为 dereg_mr 的一部分，它被发现非常耗时。例如，注销 10000 个 ODP MR，每个 MR 的大小为 2M 大页面，耗时 19.6 秒，相比之下，注销相同数量的非 ODP MR 耗时 172 毫秒。新的锁定方案使用 wait_event() 机制，该机制遵循 MR 的使用计数，而不是使用 synchronize_srcu()。通过这一改变，上述测试所需的时间为 95 毫秒，这甚至比非 ODP 流更好。一旦完全放弃 srcu 使用，就必须使用锁来保护 XA 访问。作为使用上述机制的一部分，我们还可以清理 num_deferred_work 内容并改为遵循使用计数
                xa_store(&dev->odp_mkeys, mlx5_base_mkey(mmkey->key), mmkey, GFP_KERNEL)
            mlx5_ib_init_odp_mr(mr)
                pagefault_real_mr(mr, to_ib_umem_odp(mr->umem), mr->umem->address, mr->umem->length, NULL, MLX5_PF_FLAGS_SNAPSHOT | MLX5_PF_FLAGS_ENABLE) -> RDMA/mlx5：从 pagefault_mr 中分离隐式处理，单个例程在处理隐式父级时，前进到下一个子 MR 的方案非常混乱。此方案只能在处理隐式父级时使用，在处理正常 MR 时不得触发。通过将所有单个 MR 内容直接放入一个函数并在隐式情况下循环调用它来重新安排事物。简化新 pagefault_real_mr() 中的一些错误处理以删除不需要的 goto
                    ib_umem_odp_map_dma_and_lock -> DMA 映射 ODP MR 中的用户空间内存并锁定它。将参数中传递的范围映射到 DMA 地址。映射页面的 DMA 地址在 umem_odp->dma_list 中更新。成功后，ODP MR 将被锁定，以让调用者完成其设备页表更新。成功时返回映射的页面数，失败时返回负错误代码
                        current_seq = range.notifier_seq = mmu_interval_read_begin(&umem_odp->notifier)
                        hmm_range_fault(&range) -> 设备驱动程序填充一系列虚拟地址
                        mmu_interval_read_retry
                        hmm_order = hmm_pfn_to_map_order(range.hmm_pfns[pfn_index])
                        ib_umem_odp_map_dma_single_page(umem_odp, dma_index, hmm_pfn_to_page(range.hmm_pfns[pfn_index]),access_mask)
                            *dma_addr = ib_dma_map_page(dev, page, 0, 1 << umem_odp->page_shift, DMA_BIDIRECTIONAL)
                                dma_map_page(dev->dma_device, page, offset, size, direction)
                    mlx5r_umr_update_xlt
                        mlx5r_umr_create_xlt
                            mlx5r_umr_alloc_xlt
                            dma = dma_map_single(ddev, xlt, sg->length, DMA_TO_DEVICE)
                        mlx5r_umr_set_update_xlt_ctrl_seg
                        mlx5r_umr_set_update_xlt_mkey_seg
                        mlx5r_umr_set_update_xlt_data_seg
                    return np << (page_shift - PAGE_SHIFT)
            return &mr->ibmr

const struct mmu_interval_notifier_ops mlx5_mn_ops = {
  .invalidate = mlx5_ib_invalidate_range,
};


mlx5_ib_eq_pf_process
    switch (eqe->sub_type)
    ...
    INIT_WORK(&pfault->work, mlx5_ib_eqe_pf_action)
    cc = mlx5_eq_update_cc(eq->core, ++cc)
    mlx5_eq_update_ci(eq->core, cc, 1)
```