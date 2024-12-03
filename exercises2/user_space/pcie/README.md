

```
root@134node:~# lshw -class network -businfo
Bus info          Device      Class          Description
========================================================
pci@0000:3b:00.0  enp59s0     network        Ethernet Controller XL710 for 40GbE QSFP+
pci@0000:3f:00.0              network        Ethernet Connection X722
pci@0000:3f:00.2  enp63s0f2   network        Ethernet Connection X722 for 1GbE
pci@0000:3f:00.3  enp63s0f3   network        Ethernet Connection X722 for 1GbE
pci@0000:af:00.0  enp175s0f0  network        82599ES 10-Gigabit SFI/SFP+ Network Connection
pci@0000:af:00.1  enp175s0f1  network        82599ES 10-Gigabit SFI/SFP+ Network Connection
root@134node:~# 
```




```
 lspci -vvv -s af:00.0 
af:00.0 Ethernet controller: Intel Corporation Ethernet Controller XL710 for 40GbE QSFP+ (rev 02)
        Subsystem: Intel Corporation Ethernet Converged Network Adapter XL710-Q1
        Physical Slot: 106-1
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr+ Stepping- SERR+ FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0, Cache Line Size: 32 bytes
        Interrupt: pin A routed to IRQ 314
        NUMA node: 1
        Region 0: Memory at ed000000 (64-bit, prefetchable) [size=8M]
        Region 3: Memory at ee000000 (64-bit, prefetchable) [size=32K]
        Expansion ROM at ee600000 [disabled] [size=512K]
        Capabilities: [40] Power Management version 3
                Flags: PMEClk- DSI+ D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold+)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=1 PME-
        Capabilities: [50] MSI: Enable- Count=1/1 Maskable+ 64bit+
                Address: 0000000000000000  Data: 0000
                Masking: 00000000  Pending: 00000000
        Capabilities: [70] MSI-X: Enable+ Count=129 Masked-
                Vector table: BAR=3 offset=00000000
                PBA: BAR=3 offset=00001000
        Capabilities: [a0] Express (v2) Endpoint, MSI 00
                DevCap: MaxPayload 2048 bytes, PhantFunc 0, Latency L0s <512ns, L1 <64us
                        ExtTag+ AttnBtn- AttnInd- PwrInd- RBE+ FLReset+ SlotPowerLimit 0.000W
                DevCtl: CorrErr+ NonFatalErr+ FatalErr+ UnsupReq-
                        RlxdOrd+ ExtTag+ PhantFunc- AuxPwr- NoSnoop- FLReset-
                        MaxPayload 256 bytes, MaxReadReq 512 bytes
                DevSta: CorrErr+ NonFatalErr- FatalErr- UnsupReq+ AuxPwr- TransPend-
                LnkCap: Port #0, Speed 8GT/s, Width x8, ASPM L1, Exit Latency L1 <16us
                        ClockPM- Surprise- LLActRep- BwNot- ASPMOptComp+
                LnkCtl: ASPM L1 Enabled; RCB 64 bytes, Disabled- CommClk+
                        ExtSynch- ClockPM- AutWidDis- BWInt- AutBWInt-
                LnkSta: Speed 8GT/s (ok), Width x8 (ok)
                        TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-
                DevCap2: Completion Timeout: Range ABCD, TimeoutDis+ NROPrPrP- LTR-
                         10BitTagComp- 10BitTagReq- OBFF Not Supported, ExtFmt- EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS- TPHComp- ExtTPHComp-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis- LTR- OBFF Disabled,
                         AtomicOpsCtl: ReqEn-
                LnkCap2: Supported Link Speeds: 2.5-8GT/s, Crosslink- Retimer- 2Retimers- DRS-
                LnkCtl2: Target Link Speed: 2.5GT/s, EnterCompliance- SpeedDis-
                         Transmit Margin: Normal Operating Range, EnterModifiedCompliance- ComplianceSOS-
                         Compliance De-emphasis: -6dB
                LnkSta2: Current De-emphasis Level: -3.5dB, EqualizationComplete+ EqualizationPhase1+
                         EqualizationPhase2+ EqualizationPhase3+ LinkEqualizationRequest-
                         Retimer- 2Retimers- CrosslinkRes: unsupported
        Capabilities: [e0] Vital Product Data
                Product Name: XL710 40GbE Controller
                Read-only fields:
                        [PN] Part number: 
                        [EC] Engineering changes: 
                        [FG] Unknown: 
                        [LC] Unknown: 
                        [MN] Manufacture ID: 
                        [PG] Unknown: 
                        [SN] Serial number: 
                        [V0] Vendor specific: 
                        [RV] Reserved: checksum good, 0 byte(s) reserved
                Read/write fields:
                        [V1] Vendor specific: 
                End
        Capabilities: [100 v2] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq+ ACSViol-
                UESvrt: DLP+ SDES+ TLP- FCP+ CmpltTO- CmpltAbrt- UnxCmplt- RxOF+ MalfTLP+ ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr+
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr+
                AERCap: First Error Pointer: 00, ECRCGenCap+ ECRCGenEn- ECRCChkCap+ ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [140 v1] Device Serial Number a8-6b-af-ff-ff-fe-fd-3c
        Capabilities: [150 v1] Alternative Routing-ID Interpretation (ARI)
                ARICap: MFVC- ACS-, Next Function: 0
                ARICtl: MFVC- ACS-, Function Group: 0
        Capabilities: [160 v1] Single Root I/O Virtualization (SR-IOV)
                IOVCap: Migration-, Interrupt Message Number: 000
                IOVCtl: Enable- Migration- Interrupt- MSE- ARIHierarchy+
                IOVSta: Migration-
                Initial VFs: 128, Total VFs: 128, Number of VFs: 0, Function Dependency Link: 00
                VF offset: 16, stride: 1, Device ID: 154c
                Supported Page Size: 00000553, System Page Size: 00000001
                Region 0: Memory at 00000000ed800000 (64-bit, prefetchable)
                Region 3: Memory at 00000000ee008000 (64-bit, prefetchable)
                VF Migration: offset: 00000000, BIR: 0
        Capabilities: [1a0 v1] Transaction Processing Hints
                Device specific mode supported
                No steering table available
        Capabilities: [1b0 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
        Capabilities: [1d0 v1] Secondary PCI Express
                LnkCtl3: LnkEquIntrruptEn- PerformEqu-
                LaneErrStat: 0
        Kernel driver in use: i40e
        Kernel modules: i40e
```

# 安装 lstopo 命令

+ ubuntu   
```
sudo apt install hwloc
```
+ centos    
```
sudo yum install hwloc-gui
```

Show the summarized system topology in a graphical window (or print to console if no graphical display is available):   
lstopo    
Show the full system topology without summarizations:   
lstopo --no-factorize   
Show the summarized system topology with only [p]hysical indices (i.e. as seen by the OS):   
lstopo --physical   
Write the full system topology to a file in the specified format:    
lstopo --no-factorize --output-format console|ascii|tex|fig|svg|pdf|ps|png|xml path/to/file   


```
 lstopo --no-factorize --output-format png 220.png
```

![images](220.png)