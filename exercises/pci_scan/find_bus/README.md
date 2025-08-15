

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci_scan/find_bus/find.png)


```
static int PCIe_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
        pr_info("***************** pci bus  info show ************ \n");
        u8 index = 0;
        printk("Vendor: %04x Device: %04x, devfun %d, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
        //printk("Vendor: %#x Device: %#x, devfun %x, and name %s \n", pdev->vendor, pdev->device, pdev->devfn, pci_name(pdev));
        struct pci_bus *bus = pdev->bus, *bus2;
        show_pci_info(bus);
        pr_info("***************** pci bus************ \n");
        for(index = 0; index != 0xff; index++)
        {
            bus2 = pci_find_bus(pci_domain_nr(bus), index);
            if (bus2)
            {
                   pr_err("bus2 name : %s, bus2 ops %p \n", bus2->name, bus2->ops);
            }
        }
        //pr_info("***************** pci scan ************ \n");
        //test_pci_scan_device(bus, 254);
        //test_pci_scan_device(bus, 8);
        return 0;
}
```