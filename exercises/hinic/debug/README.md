# hinic_hw_if.c
```
static int
print_bars(struct pci_dev *pdev)
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
```