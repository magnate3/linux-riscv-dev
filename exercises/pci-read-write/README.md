

```

#define PCI_byte_BAD 0
#define PCI_word_BAD (pos & 1)
#define PCI_dword_BAD (pos & 3)

#ifdef CONFIG_PCI_LOCKLESS_CONFIG
# define pci_lock_config(f)     do { (void)(f); } while (0)
# define pci_unlock_config(f)   do { (void)(f); } while (0)
#else
# define pci_lock_config(f)     raw_spin_lock_irqsave(&pci_lock, f)
# define pci_unlock_config(f)   raw_spin_unlock_irqrestore(&pci_lock, f)
#endif

#define PCI_OP_READ(size, type, len) \
int pci_bus_read_config_##size \
        (struct pci_bus *bus, unsigned int devfn, int pos, type *value) \
{                                                                       \
        int res;                                                        \
        unsigned long flags;                                            \
        u32 data = 0;                                                   \
        if (PCI_##size##_BAD) return PCIBIOS_BAD_REGISTER_NUMBER;       \
        pci_lock_config(flags);                                         \
        res = bus->ops->read(bus, devfn, pos, len, &data);              \
        *value = (type)data;                                            \
        pci_unlock_config(flags);                                       \
        return res;                                                     \
}

#define PCI_OP_WRITE(size, type, len) \
int pci_bus_write_config_##size \
        (struct pci_bus *bus, unsigned int devfn, int pos, type value)  \
{                                                                       \
        int res;                                                        \
        unsigned long flags;                                            \
        if (PCI_##size##_BAD) return PCIBIOS_BAD_REGISTER_NUMBER;       \
        pci_lock_config(flags);                                         \
        res = bus->ops->write(bus, devfn, pos, len, value);             \
        pci_unlock_config(flags);                                       \
        return res;                                                     \
}

PCI_OP_READ(byte, u8, 1)
PCI_OP_READ(word, u16, 2)
PCI_OP_READ(dword, u32, 4)
PCI_OP_WRITE(byte, u8, 1)
PCI_OP_WRITE(word, u16, 2)
PCI_OP_WRITE(dword, u32, 4)

EXPORT_SYMBOL(pci_bus_read_config_byte);
EXPORT_SYMBOL(pci_bus_read_config_word);
EXPORT_SYMBOL(pci_bus_read_config_dword);
EXPORT_SYMBOL(pci_bus_write_config_byte);
EXPORT_SYMBOL(pci_bus_write_config_word);
EXPORT_SYMBOL(pci_bus_write_config_dword);
```


#  HINIC_PCI_DB_BAR and msix bar

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/bar.png)


## msix为什么在bar2

```
  Capabilities: [a0] MSI-X: Enable- Count=32 Masked-
                Vector table: BAR=2 offset=00000000
                PBA: BAR=2 offset=00004000
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/msix.png)

# e1000e

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/e1000.png)

```
root@ubuntux86:/home/ubuntu# ethtool -i enp0s31f6
driver: e1000e
version: 5.13.0-39-generic
firmware-version: 0.4-4
expansion-rom-version: 
bus-info: 0000:00:1f.6
supports-statistics: yes
supports-test: yes
supports-eeprom-access: yes
supports-register-dump: yes
supports-priv-flags: yes
root@ubuntux86:/home/ubuntu# lspci -s 0000:00:1f.6 -vv 
00:1f.6 Ethernet controller: Intel Corporation Ethernet Connection (14) I219-LM (rev 11)
        Subsystem: Dell Ethernet Connection (14) I219-LM
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Interrupt: pin A routed to IRQ 125
        Region 0: Memory at 70280000 (32-bit, non-prefetchable) [size=128K]
        Capabilities: [c8] Power Management version 3
                Flags: PMEClk- DSI+ D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold+)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=1 PME-
        Capabilities: [d0] MSI: Enable+ Count=1/1 Maskable- 64bit+
                Address: 00000000fee00298  Data: 0000
        Kernel driver in use: e1000e
        Kernel modules: e1000e
root@ubuntux86:/work/pci_test# lspci -s 0000:00:1f.6 -n
00:1f.6 0200: 8086:15f9 (rev 11)
```

# bridge

```
 echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
```




```
 
static inline const char *pci_name(const struct pci_dev *pdev)
{
        return dev_name(&pdev->dev);
}

static inline const char *dev_name(const struct device *dev)
{
        /* Use the init name until the kobject becomes available */
        if (dev->init_name)
                return dev->init_name;

        return kobject_name(&dev->kobj);
}


```  


 (bridge) 0000:00:0c.0 
          --- (bridge)0000:03:00.0
             --- (bridge)0000:04:00.0
                --- (nic)05:00.0
				
![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/br.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/tree.png)


![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/pci-read-write/br_addr.png)

````

[root@centos7 pci_test2]# insmod  pci_test2.ko 
[root@centos7 pci_test2]# echo 0x19e5 0x0200 > /sys/bus/pci/drivers/PCIe_demo/new_id
[root@centos7 pci_test2]# dmesg | tail -n 20
[  171.893973] pci is bridge     
[  171.896840] pci is bridge     
[  171.899706] pci is bridge     
[  171.902573] pci is bridge     
[  392.515989] ***************** pci info show ************ 
[  392.524233] bus name : PCI Bus 0000:05, bus ops ffff000008eb6a90 
[  392.530303] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:04:00.0 
[  392.537231] bus name : PCI Bus 0000:04, bus ops ffff000008eb6a90 
[  392.543299] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:03:00.0 
[  392.550230] bus name : PCI Bus 0000:03, bus ops ffff000008eb6a90 
[  392.556295] Vendor: 0x19e5 Device: 0xa120, devfun 60, and name 0000:00:0c.0 
[  392.563312] bus name : , bus ops ffff000008eb6a90 
[  392.568113] ***************** pci info show ************ 
[  392.573489] bus name : PCI Bus 0000:06, bus ops ffff000008eb6a90 
[  392.579558] Vendor: 0x19e5 Device: 0x371e, devfun 8, and name 0000:04:01.0 
[  392.586486] bus name : PCI Bus 0000:04, bus ops ffff000008eb6a90 
[  392.592554] Vendor: 0x19e5 Device: 0x371e, devfun 0, and name 0000:03:00.0 
[  392.599485] bus name : PCI Bus 0000:03, bus ops ffff000008eb6a90 
[  392.605550] Vendor: 0x19e5 Device: 0xa120, devfun 60, and name 0000:00:0c.0 
[  392.612568] bus name : , bus ops ffff000008eb6a90 
[root@centos7 pci_test2]# lspci -vvs 0000:04:00.0
04:00.0 PCI bridge: Huawei Technologies Co., Ltd. Hi1822 Family Virtual Bridge (rev 45) (prog-if 00 [Normal decode])
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 32 bytes
        NUMA node: 0
        Bus: primary=04, secondary=05, subordinate=05, sec-latency=0
        Memory behind bridge: e9200000-e92fffff
        Prefetchable memory behind bridge: 0000080000200000-0000080008afffff
        Secondary status: 66MHz- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- <SERR- <PERR-
        BridgeCtl: Parity+ SERR+ NoISA- VGA- MAbort- >Reset- FastB2B-
                PriDiscTmr- SecDiscTmr- DiscTmrStat- DiscTmrSERREn-
        Capabilities: [40] Express (v2) Downstream Port (Slot-), MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0
                        ExtTag+ RBE+
                DevCtl: Report errors: Correctable- Non-Fatal- Fatal- Unsupported-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr+ UncorrErr- FatalErr- UnsuppReq+ AuxPwr+ TransPend-
                LnkCap: Port #1, Speed 8GT/s, Width x16, ASPM not supported, Exit Latency L0s unlimited, L1 unlimited
                        ClockPM- Surprise+ LLActRep+ BwNot+ ASPMOptComp+
                LnkCtl: ASPM Disabled; Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 8GT/s, Width x16, TrErr- Train- SlotClk- DLActive+ BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis-, LTR-, OBFF Not Supported ARIFwd+
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled ARIFwd+
                LnkCtl2: Target Link Speed: 8GT/s, EnterCompliance- SpeedDis-, Selectable De-emphasis: -6dB
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance De-emphasis: -6dB
                LnkSta2: Current De-emphasis Level: -3.5dB, EqualizationComplete+, EqualizationPhase1+
                         EqualizationPhase2+, EqualizationPhase3+, LinkEqualizationRequest-
        Capabilities: [b0] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1+,D2+,D3hot+,D3cold+)
                Status: D0 NoSoftRst- PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [d0] Subsystem: Huawei Technologies Co., Ltd. Device d129
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                AERCap: First Error Pointer: 00, GenCap+ CGenEn- ChkCap+ ChkEn-
        Capabilities: [280 v1] Access Control Services
                ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans+
                ACSCtl: SrcValid+ TransBlk- ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans-
        Capabilities: [310 v1] #19

[root@centos7 pci_test2]# lspci -vvs 0000:03:00.0
03:00.0 PCI bridge: Huawei Technologies Co., Ltd. Hi1822 Family Virtual Bridge (rev 45) (prog-if 00 [Normal decode])
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx-
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 32 bytes
        NUMA node: 0
        Bus: primary=03, secondary=04, subordinate=06, sec-latency=0
        Memory behind bridge: e9200000-e93fffff
        Prefetchable memory behind bridge: 0000080000200000-00000800113fffff
        Secondary status: 66MHz- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort+ <SERR- <PERR-
        BridgeCtl: Parity+ SERR+ NoISA- VGA- MAbort- >Reset- FastB2B-
                PriDiscTmr- SecDiscTmr- DiscTmrStat- DiscTmrSERREn-
        Capabilities: [40] Express (v2) Upstream Port, MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ SlotPowerLimit 25.000W
                DevCtl: Report errors: Correctable- Non-Fatal- Fatal- Unsupported-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr+ UncorrErr- FatalErr- UnsuppReq+ AuxPwr+ TransPend-
                LnkCap: Port #0, Speed 8GT/s, Width x16, ASPM L0s L1, Exit Latency L0s unlimited, L1 unlimited
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM Disabled; Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 8GT/s, Width x8, TrErr- Train- SlotClk- DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis-, LTR-, OBFF Not Supported
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled
                LnkCtl2: Target Link Speed: 8GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance De-emphasis: -6dB
                LnkSta2: Current De-emphasis Level: -3.5dB, EqualizationComplete+, EqualizationPhase1+
                         EqualizationPhase2+, EqualizationPhase3+, LinkEqualizationRequest-
        Capabilities: [b0] Power Management version 3
                Flags: PMEClk- DSI- D1+ D2+ AuxCurrent=0mA PME(D0+,D1+,D2+,D3hot+,D3cold+)
                Status: D0 NoSoftRst- PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [d0] Subsystem: Huawei Technologies Co., Ltd. Device d129
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr+
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                AERCap: First Error Pointer: 14, GenCap+ CGenEn- ChkCap+ ChkEn-
        Capabilities: [310 v1] #19
        Capabilities: [380 v1] Power Budgeting <?>

[root@centos7 pci_test2]# lspci -vvs 0000:00:0c.0 
00:0c.0 PCI bridge: Huawei Technologies Co., Ltd. HiSilicon PCIe Root Port with Gen4 (rev 21) (prog-if 00 [Normal decode])
        Control: I/O+ Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 32 bytes
        Interrupt: pin A routed to IRQ 102
        NUMA node: 0
        Bus: primary=00, secondary=03, subordinate=06, sec-latency=0
        Memory behind bridge: e9200000-e93fffff
        Prefetchable memory behind bridge: 0000080000200000-00000800113fffff
        Secondary status: 66MHz- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort+ <SERR- <PERR-
        BridgeCtl: Parity+ SERR+ NoISA- VGA- MAbort- >Reset- FastB2B-
                PriDiscTmr- SecDiscTmr- DiscTmrStat- DiscTmrSERREn-
        Capabilities: [40] Express (v2) Root Port (Slot+), MSI 00
                DevCap: MaxPayload 512 bytes, PhantFunc 0
                        ExtTag+ RBE+
                DevCtl: Report errors: Correctable+ Non-Fatal+ Fatal+ Unsupported+
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop+
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr- UncorrErr- FatalErr- UnsuppReq- AuxPwr- TransPend-
                LnkCap: Port #0, Speed 16GT/s, Width x8, ASPM not supported, Exit Latency L0s unlimited, L1 unlimited
                        ClockPM- Surprise+ LLActRep+ BwNot+ ASPMOptComp+
                LnkCtl: ASPM Disabled; RCB 128 bytes Disabled- CommClk-
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 8GT/s, Width x8, TrErr- Train- SlotClk+ DLActive+ BWMgmt- ABWMgmt+
                SltCap: AttnBtn- PwrCtrl- MRL- AttnInd- PwrInd- HotPlug- Surprise-
                        Slot #12, PowerLimit 25.000W; Interlock- NoCompl-
                SltCtl: Enable: AttnBtn- PwrFlt- MRL- PresDet- CmdCplt- HPIrq- LinkChg-
                        Control: AttnInd Off, PwrInd Off, Power- Interlock-
                SltSta: Status: AttnBtn- PowerFlt- MRL- CmdCplt- PresDet- Interlock-
                        Changed: MRL- PresDet- LinkState+
                RootCtl: ErrCorrectable- ErrNon-Fatal- ErrFatal- PMEIntEna+ CRSVisible+
                RootCap: CRSVisible+
                RootSta: PME ReqID 0000, PMEStatus- PMEPending-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis-, LTR-, OBFF Not Supported ARIFwd+
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled ARIFwd-
                LnkCtl2: Target Link Speed: 16GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance De-emphasis: -6dB
                LnkSta2: Current De-emphasis Level: -3.5dB, EqualizationComplete+, EqualizationPhase1+
                         EqualizationPhase2+, EqualizationPhase3+, LinkEqualizationRequest-
        Capabilities: [80] MSI: Enable+ Count=2/32 Maskable+ 64bit+
                Address: 0000000202110040  Data: 0000
                Masking: fffffffc  Pending: 00000000
        Capabilities: [b0] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1+,D2+,D3hot+,D3cold+)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [d0] Subsystem: Huawei Technologies Co., Ltd. Device 371e
        Capabilities: [100 v2] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- NonFatalErr-
                AERCap: First Error Pointer: 00, GenCap+ CGenEn- ChkCap+ ChkEn-
        Capabilities: [310 v1] #19
        Capabilities: [630 v1] Access Control Services
                ACSCap: SrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans+
                ACSCtl: SrcValid+ TransBlk- ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans-
        Capabilities: [700 v1] #25
        Capabilities: [70c v1] #27
        Capabilities: [880 v1] #26
        Kernel driver in use: pcieport

```
 





