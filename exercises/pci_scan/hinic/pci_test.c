/*
 * PCIe Configuration Space
 *
 * (C) 2019.10.24  <>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/version.h>
#include <linux/slab.h>
#include <linux/aer.h>
#include <linux/interrupt.h>
#include <linux/msi.h>
#include <linux/irq.h>
#include <linux/irqdesc.h>
#include <linux/irqdomain.h>
#include <linux/delay.h>
#define DEV_NAME		"PCIe_demo"
//8086:15f9
static const struct pci_device_id PCIe_ids[] = {
	{ PCI_DEVICE(0x19e5,0x0200 ), },
};

int print_bars(struct pci_dev *pdev)
{
        int i, iom, iop;
        unsigned long flags;
        unsigned long addr, len;
        static const char *bar_names[PCI_STD_RESOURCE_END + 1]  = {
                "BAR0",
                "BAR1",
                "BAR2",
                "BAR3",
                "BAR4",
                "BAR5",
        };

        iom = 0;
        iop = 0;

        for (i = 0; i < ARRAY_SIZE(bar_names); i++) {
             pr_err("******bar name : %s", bar_names[i]);
                addr = pci_resource_start(pdev, i);
                len = pci_resource_len(pdev, i);
                if (len != 0 && addr != 0) {
                        flags = pci_resource_flags(pdev, i);
                        if (flags & IORESOURCE_MEM) {
                                iom++;
                        } else if (flags & IORESOURCE_IO) {
                                iop++;
                        }
                        pr_info("flags  %lx,  and addr %lx, and len %lx \n", flags & IORESOURCE_MEM,  addr, len);
                }

        }
        return 0;
}
void show_pci_info(struct pci_bus *bus)
{
     struct pci_dev *pdev, *dev;
     if (!bus)
     {
         return ;
     }
     pr_err("bus name : %s, bus ops %p \n", bus->name, bus->ops);
     pdev = bus->self;
     if (pdev)
     {
         printk("Vendor: %#x Device: %#x, devfun %x, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
#if 0
         print_bars(pdev);
#endif
     }

#if 0
    list_for_each_entry(dev, &bus->devices, bus_list) {
                int i;
                if (pci_is_bridge(dev))
                {
                    pr_err("pci is bridge \t ");
                } 
                for (i = 0; i < PCI_NUM_RESOURCES; i++) {
                        struct resource *r = &dev->resource[i];

                        if (r->parent || !r->start || !r->flags)
                                continue;

                        pr_err("PCI: Claiming %s: ", pci_name(dev));
                        pr_err("Resource %d: %016llx..%016llx [%x]\n",
                                 i, (unsigned long long)r->start,
                                 (unsigned long long)r->end,
                                 (unsigned int)r->flags);
                }
    }
#endif
     show_pci_info(bus->parent);
}
#define E1000_BAR0 0
static void pci_msix_clear_and_set_ctrl(struct pci_dev *dev, u16 clear, u16 set)
{
       u16 ctrl;

       pci_read_config_word(dev, dev->msix_cap + PCI_MSIX_FLAGS, &ctrl);
       ctrl &= ~clear;
       ctrl |= set;
       pci_write_config_word(dev, dev->msix_cap + PCI_MSIX_FLAGS, ctrl);
}

static void test_msix_map_region(struct pci_dev *dev, unsigned nr_entries)
{
        resource_size_t phys_addr;
        u32 table_offset;
        unsigned long flags;
        u8 bir;

        pci_read_config_dword(dev, dev->msix_cap + PCI_MSIX_TABLE,
                              &table_offset);
        bir = (u8)(table_offset & PCI_MSIX_TABLE_BIR);
        flags = pci_resource_flags(dev, bir);
        if (!flags || (flags & IORESOURCE_UNSET))
        {
                pr_info("msix phys_addr cant not readed \n");
                return ;
        }
        table_offset &= PCI_MSIX_TABLE_OFFSET;
        phys_addr = pci_resource_start(dev, bir) + table_offset;
        pr_info("msix phys_addr : %p \n", phys_addr);
        //return ioremap_nocache(phys_addr, nr_entries * PCI_MSIX_ENTRY_SIZE);
}

static int show_irq_domain(struct irq_domain *domain)
{
      struct msi_domain_info *info;
      struct irq_chip *chip ; 
      if (!domain)
      {
	  return 0;
      }
#if 0
      // to be careful, the driver not call msi_domain_alloc -->irq_domain_set_hwirq_and_chip to set 
      // call this will cause coredump
      info = domain->host_data;
      if (info)
      {
	   chip = info->chip;
	   if(chip)
           {
	        pr_err("chip->name %s  \t", chip->name);
           }
           pr_info("info->handler_name : %s \t ", info->handler_name);
      }
#else
      //info = domain->host_data;
      //if (info)
      //{
      //     pr_info("host_data : %p \t ", info);

      //}
#endif 
      pr_info("domain->name : %s \n ", domain->name);
      show_irq_domain(domain->parent);
      return 0;
}
static int test_pci_msi_setup_msi_irqs(struct pci_dev *dev)
{
       struct irq_domain *domain;
       domain = dev_get_msi_domain(&dev->dev);
       if (domain && irq_domain_is_hierarchy(domain))
	       pr_info("irq_domain_is_hierarchy \t ");
       show_irq_domain(domain);
       return 0;
}
static int test_msi_capability_init(struct pci_dev *dev, unsigned nr_entries)
{
    int bars,err, pos;
    u16 ctrl;
    void __iomem * base;
    // call pci_disable_device reversely
    err = pci_enable_device_mem(dev);
    if (err)
        return err;
    //if (!pci_msi_supported(dev, nvec) || dev->current_state != PCI_D0)
    if ( dev->current_state != PCI_D0)
    {
         pr_info("x86 e1000e pci not in PCI D0 \n");
         return -EINVAL;
    }
    bars = pci_select_bars(dev, IORESOURCE_MEM);
    // call pci_release_mem_regions reversely
    err = pci_request_selected_regions_exclusive(dev, bars, "e1000e");
    if (err)
       goto disable_pci_err;
    /* AER (Advanced Error Reporting) hooks */
    pci_enable_pcie_error_reporting(dev);
    pr_err("no error repoorting ************");
    pci_set_master(dev);
    /* PCI config space info */
    err = pci_save_state(dev);
    if (err)
        goto request_region_err;
    // 
    pr_info("x86 e1000e not support msix ,and  pci_msix_vec_count : %d \n", pci_msix_vec_count(dev));
    pr_info("x86 e1000e supprot msi, pci_msi_vec_count : %d, and dev->msi_cap  %d \n", pci_msi_vec_count(dev), dev->msi_cap);
    test_pci_msi_setup_msi_irqs(dev);
    pos =  pci_find_capability(dev, PCI_CAP_ID_MSI);
    pci_read_config_word(dev, pos + PCI_MSI_FLAGS, &ctrl);
    pr_info("is_64bit_address %d \n", (ctrl & PCI_MSI_FLAGS_64BIT));
#if 0
    /* Ensure MSI-X is disabled while it is set up */
    pci_msix_clear_and_set_ctrl(dev, PCI_MSIX_FLAGS_ENABLE, 0);
    pci_read_config_word(dev, dev->msix_cap + PCI_MSIX_FLAGS, &control);
    test_msix_map_region(dev, 64);
#endif
request_region_err:
     pci_release_mem_regions(dev);
disable_pci_err:
     pci_disable_device(dev);
    return 0;
}
void test_pci_ioremap_bar(struct pci_dev *pdev, int bar)
{
        struct resource *res = &pdev->resource[bar];

        /*
 *          * Make sure the BAR is actually a memory resource, not an IO resource
 *                   */
        if (res->flags & IORESOURCE_UNSET || !(res->flags & IORESOURCE_MEM)) {
                dev_warn(&pdev->dev, "can't ioremap BAR %d: %pR\n", bar, res);
                return ;
        }
        pr_info("bar %d phys_addr : %p, len : %lld \n",bar, res->start, resource_size(res));
        //return ioremap_nocache(res->start, resource_size(res));
}
static bool pci_bus_crs_vendor_id(u32 l)
{
	        return (l & 0xffff) == 0x0001;
}
static bool pci_bus_wait_crs(struct pci_bus *bus, int devfn, u32 *l,
		                             int timeout)
{
	        int delay = 1;
		if (!pci_bus_crs_vendor_id(*l))
	                return true;    /* not a CRS completion */

	        if (!timeout)
	             return false;   /* CRS, but caller doesn't want to wait */

				        /*
					 *          * We got the reserved Vendor ID that indicates a completion with
					 *                   * Configuration Request Retry Status (CRS).  Retry until we get a
					 *                            * valid Vendor ID or we time out.
					 *                                     */
	        while (pci_bus_crs_vendor_id(*l)) {
	                if (delay > timeout) {
	        pr_warn("pci %04x:%02x:%02x.%d: not ready after %dms; giving up\n",
		                                pci_domain_nr(bus), bus->number,
		                                PCI_SLOT(devfn), PCI_FUNC(devfn), delay - 1);

                return false;
                }
                if (delay >= 1000)
                   pr_info("pci %04x:%02x:%02x.%d: not ready after %dms; waiting\n",
                              pci_domain_nr(bus), bus->number,
			                                PCI_SLOT(devfn), PCI_FUNC(devfn), delay - 1);

                msleep(delay);
                delay *= 2;

               if (pci_bus_read_config_dword(bus, devfn, PCI_VENDOR_ID, l))
                   return false;
	        }

	        if (delay >= 1000)
                pr_info("pci %04x:%02x:%02x.%d: ready after %dms\n",
		                        pci_domain_nr(bus), bus->number,
                        PCI_SLOT(devfn), PCI_FUNC(devfn), delay - 1);

               return true;
}

bool pci_bus_generic_read_dev_vendor_id(struct pci_bus *bus, int devfn, u32 *l,
		                                        int timeout)
{
	        if (pci_bus_read_config_dword(bus, devfn, PCI_VENDOR_ID, l))
			                return false;
		        /* Some broken boards return 0 or ~0 if a slot is empty: */
		        if (*l == 0xffffffff || *l == 0x00000000 ||
					            *l == 0x0000ffff || *l == 0xffff0000)
				                return false;

			        if (pci_bus_crs_vendor_id(*l))
					                return pci_bus_wait_crs(bus, devfn, l, timeout);

				        return true;
}
bool test_pci_bus_read_dev_vendor_id(struct pci_bus *bus, int devfn, u32 *l,
		                                int timeout)
{

     return pci_bus_generic_read_dev_vendor_id(bus, devfn, l, timeout);
}
static int test_pci_scan_device(struct pci_bus *bus, int devfn)
{
    struct pci_dev *dev;
    u32 l;
    if (!test_pci_bus_read_dev_vendor_id(bus, devfn, &l, 60*1000))
    {
	 pr_err("pci_bus_read_dev_vendor_id fail \n");
	 return 0;
    }
    pr_err("vendor %x, deivce %x, fn %d \n", l & 0xffff, (l >> 16) & 0xffff, devfn);
    return 0;
}
/* PCIe device probe */
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	pr_info("***************** pci bus  info show ************ \n");
        printk("Vendor: %04x Device: %04x, devfun %d, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
        //printk("Vendor: %#x Device: %#x, devfun %x, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
	struct pci_bus *bus = pdev->bus;
	show_pci_info(bus);
        pr_info("***************** pci scan ************ \n");
	//test_pci_scan_device(bus, 254);
	//test_pci_scan_device(bus, 8);
	return 0;
}

/* PCIe device remove */
static void PCIe_remove(struct pci_dev *pdev)
{
}

static struct pci_driver PCIe_demo_driver = {
	.name		= DEV_NAME,
	.id_table	= PCIe_ids,
	.probe		= PCIe_probe,
	.remove		= PCIe_remove,
};

static int __init PCIe_demo_init(void)
{
	return pci_register_driver(&PCIe_demo_driver);
}

static void __exit PCIe_demo_exit(void)
{
	pci_unregister_driver(&PCIe_demo_driver);
}

module_init(PCIe_demo_init);
module_exit(PCIe_demo_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("dda");
MODULE_DESCRIPTION("PCIe Configuration Space Module");
