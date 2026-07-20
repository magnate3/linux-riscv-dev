
# api



```C
  err = iommu_map(pd->domain, va_start, pa_start,
                                                size, flags, GFP_ATOMIC);
```

```C
struct usnic_uiom_pd *usnic_uiom_alloc_pd(struct device *dev)
{
        struct usnic_uiom_pd *pd;
        void *domain;

        pd = kzalloc(sizeof(*pd), GFP_KERNEL);
        if (!pd)
                return ERR_PTR(-ENOMEM);

        pd->domain = domain = iommu_domain_alloc(dev->bus);
        if (!domain) {
                usnic_err("Failed to allocate IOMMU domain");
                kfree(pd);
                return ERR_PTR(-ENOMEM);
        }

        iommu_set_fault_handler(pd->domain, usnic_uiom_dma_fault, NULL);

        spin_lock_init(&pd->lock);
        INIT_LIST_HEAD(&pd->devs);

        return pd;
}
```


```C
iommu_attach_device(pd->domain, dev)
```