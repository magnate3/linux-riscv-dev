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

## 查看PCIe设备的MSI和MSI-X的配置

```
[root@centos7 ~]# lspci -s 0000:05:00.0  -vv | grep MSI
        Capabilities: [40] Express (v2) Endpoint, MSI 00
        Capabilities: [80] MSI: Enable- Count=1/32 Maskable+ 64bit+
        Capabilities: [a0] MSI-X: Enable- Count=32 Masked-
[root@centos7 ~]# lspci -s 0000:05:00.0  -xxx
05:00.0 Ethernet controller: Huawei Technologies Co., Ltd. Hi1822 Family (2*100GE) (rev 45)
00: e5 19 00 02 02 00 10 00 45 00 00 02 08 00 00 00
10: 0c 00 b0 07 00 08 00 00 0c 00 a2 08 00 08 00 00
20: 0c 00 20 00 00 08 00 00 00 00 00 00 e5 19 39 d1
30: 00 00 20 e9 40 00 00 00 00 00 00 00 ff 00 00 00
40: 10 80 02 00 e2 8f 00 10 37 29 10 00 03 f1 43 00
50: 08 00 03 01 00 00 00 00 00 00 00 00 00 00 00 00
60: 00 00 00 00 92 03 00 00 00 00 00 00 0e 00 00 00
70: 03 00 1f 00 00 00 00 00 00 00 00 00 00 00 00 00
80: 05 a0 8a 01 00 00 00 00 00 00 00 00 00 00 00 00
90: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
a0: 11 b0 1f 00 02 00 00 00 02 40 00 00 00 00 00 00
b0: 01 c0 03 f8 00 00 00 00 00 00 00 00 00 00 00 00
c0: 03 00 28 80 37 32 78 ff 00 00 00 00 00 00 00 00
d0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
e0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
f0: 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00

[root@centos7 ~]# 

[root@centos7 igb-uio]# lspci -s 0000:05:00.0  -vv 
05:00.0 Ethernet controller: Huawei Technologies Co., Ltd. Hi1822 Family (2*100GE) (rev 45)
        Subsystem: Huawei Technologies Co., Ltd. Hi1822 SP572 (2*100GE)
        Control: I/O- Mem+ BusMaster- SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        NUMA node: 0
        Region 0: Memory at 80007b00000 (64-bit, prefetchable) [size=128K]
        Region 2: Memory at 80008a20000 (64-bit, prefetchable) [size=32K]
        Region 4: Memory at 80000200000 (64-bit, prefetchable) [size=1M]
        Expansion ROM at e9200000 [disabled] [size=1M]
        Capabilities: [40] Express (v2) Endpoint, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0, Latency L0s unlimited, L1 unlimited
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+ SlotPowerLimit 0.000W
                DevCtl: Report errors: Correctable+ Non-Fatal+ Fatal+ Unsupported-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+ FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- UncorrErr- FatalErr- UnsuppReq- AuxPwr+ TransPend-
                LnkCap: Port #0, Speed 8GT/s, Width x16, ASPM not supported, Exit Latency L0s unlimited, L1 unlimited
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 128 bytes Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 8GT/s, Width x16, TrErr- Train- SlotClk- DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range B, TimeoutDis+, LTR-, OBFF Not Supported
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled
                LnkCtl2: Target Link Speed: 8GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance De-emphasis: -6dB
                LnkSta2: Current De-emphasis Level: -3.5dB, EqualizationComplete+, EqualizationPhase1+
                         EqualizationPhase2+, EqualizationPhase3+, LinkEqualizationRequest-
        Capabilities: [80] MSI: Enable- Count=1/32 Maskable+ 64bit+
                Address: 0000000000000000  Data: 0000
                Masking: 00000000  Pending: 00000000
        Capabilities: [a0] MSI-X: Enable- Count=32 Masked-
                Vector table: BAR=2 offset=00000000
                PBA: BAR=2 offset=00004000
        Capabilities: [b0] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1+,D2+,D3hot+,D3cold+)
                Status: D0 NoSoftRst- PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [c0] Vital Product Data
                Product Name: Huawei IN200 2*100GE Adapter
                Read-only fields:
                        [PN] Part number: SP572
                End
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                AERCap: First Error Pointer: 00, GenCap+ CGenEn- ChkCap+ ChkEn-
        Capabilities: [150 v1] Alternative Routing-ID Interpretation (ARI)
                ARICap: MFVC- ACS-, Next Function: 0
                ARICtl: MFVC- ACS-, Function Group: 0
        Capabilities: [200 v1] Single Root I/O Virtualization (SR-IOV)
                IOVCap: Migration-, Interrupt Message Number: 000
                IOVCtl: Enable- Migration- Interrupt- MSE- ARIHierarchy+
                IOVSta: Migration-
                Initial VFs: 120, Total VFs: 120, Number of VFs: 0, Function Dependency Link: 00
                VF offset: 1, stride: 1, Device ID: 375e
                Supported Page Size: 00000553, System Page Size: 00000010
                Region 0: Memory at 0000080007b20000 (64-bit, prefetchable)
                Region 2: Memory at 00000800082a0000 (64-bit, prefetchable)
                Region 4: Memory at 0000080000300000 (64-bit, prefetchable)
                VF Migration: offset: 00000000, BIR: 0
        Capabilities: [310 v1] #19
        Capabilities: [4e0 v1] Device Serial Number 44-a1-91-ff-ff-a4-9c-0b
        Capabilities: [4f0 v1] Transaction Processing Hints
                Device specific mode supported
                No steering table available
        Capabilities: [600 v1] Vendor Specific Information: ID=0000 Rev=0 Len=028 <?>
        Capabilities: [630 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
        Kernel driver in use: igb_uio
        Kernel modules: hinic
```
***Vector table of  MSI-X***



