
#  Multi-function device

```
        pci_bus_read_config_byte(ctrl->pci_bus, PCI_DEVFN(new_slot->device, 0), 0x0B, &class_code);
        pci_bus_read_config_byte(ctrl->pci_bus, PCI_DEVFN(new_slot->device, 0), PCI_HEADER_TYPE, &header_type);

        if (header_type & 0x80) /* Multi-function device */
                max_functions = 8;
        else
                max_functions = 1
```