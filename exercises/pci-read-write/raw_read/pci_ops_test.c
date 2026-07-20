#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <asm/io.h>

const u32 PCI_ENABLE_BIT     = 0x80000000;
const u32 PCI_CONFIG_ADDRESS = 0xCF8;
const u32 PCI_CONFIG_DATA    = 0xCFC;

// func - 0-7
// slot - 0-31
// bus - 0-255
u32 r_pci_32(u8 bus, u8 device, u8 func, u8 pcireg) {
        // unsigned long flags;
        // local_irq_save(flags)

        outl(PCI_ENABLE_BIT | (bus << 16) | (device << 11) | (func << 8) | (pcireg << 2), PCI_CONFIG_ADDRESS);
        u32 ret = inl(PCI_CONFIG_DATA);

        // local_irq_restore(flags);
        return ret;
}

static __init int init_pcilist(void) {
        u8 bus, device, func;
        u32 data;

        for(bus = 0; bus != 0xff; bus++) {
                for(device = 0; device < 32; device++) {
                        for(func = 0; func < 8; func++) {
                                data = r_pci_32(bus, device, func, 0);

                                if(data != 0xffffffff) {
                                        printk(KERN_INFO "bus %d, device %d, func %d: vendor=0x%08x\n", bus, device, func, data);
                                }
                        }
                }
        }
        return 0;
}

static __exit void exit_pcilist(void) {
        return;
}

MODULE_LICENSE("GPL");
module_init(init_pcilist);
module_exit(exit_pcilist);
