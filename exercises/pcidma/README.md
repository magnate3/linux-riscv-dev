Description
---
This kernel module is a PCI device driver. Its purpose is to control the DMA / bus mastering. A program can enable the DMA of a PCI device given its location. The kernel module makes sure that DMA will be disabled when the program terminates.


Usage
---
In order to use this kernel module, you have first to compile it and load it:
```
make
sudo insmod pcidma.ko
```

Let's assume that you have a PCI device on your system, such as:
```
$ lspci -D | grep Ethernet
0000:42:00.1 Ethernet controller: Intel Corporation Ethernet 10G 2P X520 Adapter (rev 01)
```

An example program that uses the kernel module to enable DMA is the following:
```
#include "pcidma.h"

int main(int argc, char **argv)
{
    int fd;
    struct args_enable args;

    args.pci_loc.domain = 0;
    args.pci_loc.bus = 0x42;
    args.pci_loc.slot = 0;
    args.pci_loc.func = 1;

    fd = open("/dev/pcidma", O_RDONLY);
    ioctl(fd, PCIDMA_ENABLE, &args);
    /* do_work(); */
    return 0;
}
```

Note: In order for the `ioctl` to succeed, the specified device must not be assigned to another device driver.

# test


```
[root@centos7 pcidma]#  lspci -D | grep Ethernet
0000:05:00.0 Ethernet controller: Huawei Technologies Co., Ltd. Hi1822 Family (2*100GE) (rev 45)
0000:06:00.0 Ethernet controller: Huawei Technologies Co., Ltd. Hi1822 Family (2*100GE) (rev 45)
0000:7d:00.0 Ethernet controller: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE RDMA Network Controller (rev 21)
0000:7d:00.1 Ethernet controller: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE Network Controller (rev 21)
0000:7d:00.2 Ethernet controller: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE RDMA Network Controller (rev 21)
0000:7d:00.3 Ethernet controller: Huawei Technologies Co., Ltd. HNS GE/10GE/25GE Network Controller (rev 21)
[root@centos7 pcidma]# 
```


## pci_enable_device


```
Before touching any device registers, the driver needs to enable
the PCI device by calling pci_enable_device(). This will:
	o wake up the device if it was in suspended state,
	o allocate I/O and memory regions of the device (if BIOS did not),
	o allocate an IRQ (if BIOS did not).

```

## pci_set_master

```
pci_set_master() will enable DMA by setting the bus master bit
in the PCI_COMMAND register. It also fixes the latency timer value if
it's set to something bogus by the BIOS.  pci_clear_master() will
disable DMA by clearing the bus master bit.
 
```

##  PCI_COMMAND register

```
pci_write_config_word(dev, PCI_COMMAND,cmd & ~PCI_COMMAND_INTX_DISABLE)
```

##  expected expression before ‘struct’   PCIDMA_ENABLE _IOR('a', 0x01, struct args_enable)

```
[root@centos7 pcidma]# gcc  test1.c -o test1
In file included from test1.c:4:0:
test1.c: In function ‘main’:
pcidma.h:29:39: error: expected expression before ‘struct’
 #define PCIDMA_ENABLE _IOR('a', 0x01, struct args_enable)
                                       ^
test1.c:17:15: note: in expansion of macro ‘PCIDMA_ENABLE’
     ioctl(fd, PCIDMA_ENABLE, &args);
```

*** add #include <sys/ioctl.h> ***
