
  

```
root@ubuntux86:# lsmod | grep rdma
root@ubuntux86:# modprobe nvme_rdma
root@ubuntux86:# lsmod | grep rdma
nvme_rdma              40960  0
nvme_fabrics           24576  1 nvme_rdma
rdma_cm               114688  1 nvme_rdma
iw_cm                  53248  1 rdma_cm
ib_cm                 122880  1 rdma_cm
ib_core               360448  4 rdma_cm,nvme_rdma,iw_cm,ib_cm
nvme_core             126976  5 nvme,nvme_rdma,nvme_fabrics
root@ubuntux86:# insmod  rdma_krping.ko debug=1
root@ubuntux86:# 
```

# bug

```
static inline void ib_dma_free_coherent(struct ib_device *dev, size_t size, void *cpu_addr, u64 dma_handle)
{
                        //dma_free_coherent(dev->dma_device, size, cpu_addr, dma_handle);

}
static inline void *ib_dma_alloc_coherent(struct ib_device *dev, size_t size, u64 *dma_handle, gfp_t flag)
{
         //return dma_alloc_coherent(dev->dma_device, size, dma_handle, flag);
}
```