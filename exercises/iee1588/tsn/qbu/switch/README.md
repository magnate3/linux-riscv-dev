


# driver

```
root@riscv-0101:/# uname -a
Linux riscv-0101 5.4.3-rt1 #24 SMP PREEMPT_RT Thu May 5 11:21:57 CST 2022 aarch64 aarch64 aarch64 GNU/Linux
root@riscv-0101:/#
```

```
root@riscv-0101:~# ethtool -i eno0
driver: fsl_enetc
version: 5.4.3-rt1
firmware-version: 
expansion-rom-version: 
bus-info: 0000:00:00.0
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: no
root@riscv-0101:~# ethtool -i eno2
driver: fsl_enetc
version: 5.4.3-rt1
firmware-version: 
expansion-rom-version: 
bus-info: 0000:00:00.2
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: no
root@riscv-0101:~# cat /sys/class/net/eno2/device/sriov_numvfs
cat: /sys/class/net/eno2/device/sriov_numvfs: No such file or directory
root@riscv-0101:~# cat /sys/class/net/eno0/device/sriov_numvfs
0
root@riscv-0101:~# 
root@riscv-0101:~# ethtool -i swp0
driver: dsa
version: 5.4.3-rt1
firmware-version: N/A
expansion-rom-version: 
bus-info: platform
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: no
supports-priv-flags: no
root@riscv-0101:~# 
```
# pci bridge

```
root@riscv-0101:/# lspci -vvv  | grep 'bridge'      
lspci: Unable to load libkmod resources: error -12
0001:00:00.0 PCI bridge: Freescale Semiconductor Inc Device 82c0 (rev 10) (prog-if 00 [Normal decode])
        I/O behind bridge: 0000f000-00000fff [disabled]
        Memory behind bridge: fff00000-000fffff [disabled]
        Prefetchable memory behind bridge: 00000000fff00000-00000000000fffff [disabled]
0002:00:00.0 PCI bridge: Freescale Semiconductor Inc Device 82c0 (rev 10) (prog-if 00 [Normal decode])
        I/O behind bridge: 0000f000-00000fff [disabled]
        Memory behind bridge: 40000000-400fffff [size=1M]
        Prefetchable memory behind bridge: 00000000fff00000-00000000000fffff [disabled]
```

# eno1

```
root@riscv-0101:/# ethtool -i eno0
driver: fsl_enetc
version: 5.4.3-rt1
firmware-version: 
expansion-rom-version: 
bus-info: 0000:00:00.0
supports-statistics: yes
supports-test: no
supports-eeprom-access: no
supports-register-dump: yes
supports-priv-flags: no
root@riscv-0101:/# lspci -vvv  | grep '0000:00:00.0' -A 100
lspci: Unable to load libkmod resources: error -12
0000:00:00.0 Ethernet controller: Freescale Semiconductor Inc Device e100 (rev 01) (prog-if 01)
        Subsystem: Freescale Semiconductor Inc Device e100
        Device tree node: /sys/firmware/devicetree/base/soc/pcie@1f0000000/ethernet@0,0
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Region 0: Memory at 1f8000000 (32-bit, non-prefetchable) [enhanced] [size=256K]
        Region 2: Memory at 1f8160000 (32-bit, non-prefetchable) [enhanced] [size=64K]
        Capabilities: [40] Express (v2) Root Complex Integrated Endpoint, MSI 00
                DevCap: MaxPayload 128 bytes, PhantFunc 0
                        ExtTag- RBE- FLReset+
                DevCtl: CorrErr- NonFatalErr- FatalErr- UnsupReq-
                        RlxdOrd- ExtTag- PhantFunc- AuxPwr- NoSnoop- FLReset-
                        MaxPayload 128 bytes, MaxReadReq 128 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis-, NROPrPrP-, LTR-
                         10BitTagComp-, 10BitTagReq-, OBFF Not Supported, ExtFmt-, EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled
                         AtomicOpsCtl: ReqEn-
        Capabilities: [80] MSI-X: Enable+ Count=32 Masked-
                Vector table: BAR=2 offset=00000000
                PBA: BAR=2 offset=00000200
        Capabilities: [90] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [9c] Enhanced Allocation (EA): NumEntries=4
                Entry 0: Enable+ Writable- EntrySize=3
                         BAR Equivalent Indicator: BAR 0
                         PrimaryProperties: memory space, non-prefetchable
                         SecondaryProperties: entry unavailable for use, PrimaryProperties should be used
                         Base: 1f8000000
                         MaxOffset: 0003ffff
                Entry 1: Enable+ Writable- EntrySize=3
                         BAR Equivalent Indicator: BAR 2
                         PrimaryProperties: memory space, prefetchable
                         SecondaryProperties: memory space, non-prefetchable
                         Base: 1f8160000
                         MaxOffset: 0000ffff
                Entry 2: Enable+ Writable- EntrySize=3
                         BAR Equivalent Indicator: VF-BAR 0
                         PrimaryProperties: VF memory space, non-prefetchable
                         SecondaryProperties: entry unavailable for use, PrimaryProperties should be used
                         Base: 1f81d0000
                         MaxOffset: 0000ffff
                Entry 3: Enable+ Writable- EntrySize=3
                         BAR Equivalent Indicator: VF-BAR 2
                         PrimaryProperties: VF memory space, prefetchable
                         SecondaryProperties: VF memory space, prefetchable
                         Base: 1f81f0000
                         MaxOffset: 0000ffff
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UESvrt: DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 16, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [130 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
        Capabilities: [140 v1] Single Root I/O Virtualization (SR-IOV)
                IOVCap: Migration-, Interrupt Message Number: 000
                IOVCtl: Enable- Migration- Interrupt- MSE- ARIHierarchy-
                IOVSta: Migration-
                Initial VFs: 2, Total VFs: 2, Number of VFs: 0, Function Dependency Link: 00
                VF offset: 8, stride: 1, Device ID: ef00
                Supported Page Size: 00000013, System Page Size: 00000010
                VF Migration: offset: 00000000, BIR: 0
        Kernel driver in use: fsl_enetc
```

# switch




![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/sw2.png)




## eno2

### pci

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/pci2.png)

```
root@riscv-0101:/# lspci -vvv  | grep '0000:00:00.2' -A 100
lspci: Unable to load libkmod resources: error -12
0000:00:00.2 Ethernet controller: Freescale Semiconductor Inc Device e100 (rev 01) (prog-if 01)
        Subsystem: Freescale Semiconductor Inc Device e100
        Device tree node: /sys/firmware/devicetree/base/soc/pcie@1f0000000/ethernet@0,2
        Control: I/O- Mem+ BusMaster+ SpecCycle- MemWINV- VGASnoop- ParErr- Stepping- SERR- FastB2B- DisINTx+
        Status: Cap+ 66MHz- UDF- FastB2B- ParErr- DEVSEL=fast >TAbort- <TAbort- <MAbort- >SERR- <PERR- INTx-
        Latency: 0
        Region 0: Memory at 1f8080000 (32-bit, non-prefetchable) [enhanced] [size=256K]
        Region 2: Memory at 1f8180000 (32-bit, non-prefetchable) [enhanced] [size=64K]
        Capabilities: [40] Express (v2) Root Complex Integrated Endpoint, MSI 00
                DevCap: MaxPayload 128 bytes, PhantFunc 0
                        ExtTag- RBE- FLReset+
                DevCtl: CorrErr- NonFatalErr- FatalErr- UnsupReq-
                        RlxdOrd- ExtTag- PhantFunc- AuxPwr- NoSnoop- FLReset-
                        MaxPayload 128 bytes, MaxReadReq 128 bytes
                DevSta: CorrErr- NonFatalErr- FatalErr- UnsupReq- AuxPwr- TransPend-
                DevCap2: Completion Timeout: Not Supported, TimeoutDis-, NROPrPrP-, LTR-
                         10BitTagComp-, 10BitTagReq-, OBFF Not Supported, ExtFmt-, EETLPPrefix-
                         EmergencyPowerReduction Not Supported, EmergencyPowerReductionInit-
                         FRS-
                         AtomicOpsCap: 32bit- 64bit- 128bitCAS-
                DevCtl2: Completion Timeout: 50us to 50ms, TimeoutDis-, LTR-, OBFF Disabled
                         AtomicOpsCtl: ReqEn-
        Capabilities: [80] MSI-X: Enable+ Count=16 Masked-
                Vector table: BAR=2 offset=00000000
                PBA: BAR=2 offset=00000100
        Capabilities: [90] Power Management version 3
                Flags: PMEClk- DSI- D1- D2- AuxCurrent=0mA PME(D0+,D1-,D2-,D3hot+,D3cold-)
                Status: D0 NoSoftRst+ PME-Enable- DSel=0 DScale=0 PME-
        Capabilities: [9c] Enhanced Allocation (EA): NumEntries=3
                Entry 0: Enable+ Writable- EntrySize=3
                         BAR Equivalent Indicator: BAR 0
                         PrimaryProperties: memory space, non-prefetchable
                         SecondaryProperties: entry unavailable for use, PrimaryProperties should be used
                         Base: 1f8080000
                         MaxOffset: 0003ffff
                Entry 1: Enable+ Writable- EntrySize=3
                         BAR Equivalent Indicator: BAR 2
                         PrimaryProperties: memory space, prefetchable
                         SecondaryProperties: memory space, non-prefetchable
                         Base: 1f8180000
                         MaxOffset: 0000ffff
                Entry 2: Enable- Writable- EntrySize=3
                         BAR Equivalent Indicator: BAR 4
                         PrimaryProperties: memory space, non-prefetchable
                         SecondaryProperties: entry unavailable for use, PrimaryProperties should be used
                         Base: 1fc000000
                         MaxOffset: 003fffff
        Capabilities: [100 v1] Advanced Error Reporting
                UESta:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UEMsk:  DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                UESvrt: DLP- SDES- TLP- FCP- CmpltTO- CmpltAbrt- UnxCmplt- RxOF- MalfTLP- ECRC- UnsupReq- ACSViol-
                CESta:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                CEMsk:  RxErr- BadTLP- BadDLLP- Rollover- Timeout- AdvNonFatalErr-
                AERCap: First Error Pointer: 16, ECRCGenCap- ECRCGenEn- ECRCChkCap- ECRCChkEn-
                        MultHdrRecCap- MultHdrRecEn- TLPPfxPres- HdrLogCap-
                HeaderLog: 00000000 00000000 00000000 00000000
        Capabilities: [130 v1] Access Control Services
                ACSCap: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
                ACSCtl: SrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-
        Kernel driver in use: fsl_enetc
```
### dts

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/dts.png)


## tc qdisc
```
root@riscv-0101:~# tc qdisc show dev  eno2  
qdisc mq 0: root 
qdisc pfifo_fast 0: parent :8 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :7 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :6 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :5 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :4 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :3 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :2 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :1 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
```

```
root@riscv-0101:~# tc -g  class show dev eno2
+---(:8) mq 
+---(:7) mq 
+---(:6) mq 
+---(:5) mq 
+---(:4) mq 
+---(:3) mq 
+---(:2) mq 
+---(:1) mq 

root@riscv-0101:~# tc -g  class show dev eno0
+---(:8) mq 
+---(:7) mq 
+---(:6) mq 
+---(:5) mq 
+---(:4) mq 
+---(:3) mq 
+---(:2) mq 
+---(:1) mq 

root@riscv-0101:~# tc qdisc show dev  eno0
qdisc mq 0: root 
qdisc pfifo_fast 0: parent :8 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :7 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :6 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :5 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :4 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :3 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :2 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :1 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
root@riscv-0101:~# 
```

   $ tc qdisc (...) \
        sched-row <row number> <cmd> <gate mask> <interval>  \
        [base-time <interval>] [cycle-time <interval>] \
        [extension-time <interval>]


 * Frame Preemption (802.1Qbu):

   To control even further the latency, it may prove useful to signal which
   traffic classes are marked as preemptable. For that, 'taprio' provides the
   preemption command so you set each traffic class as preemptable or not:

   $ tc qdisc (...) \
        preemption 0 1 1 1


 * Time-aware shaper + Preemption:

   As an example of how Qbv and Qbu can be used together, we may specify
   both the schedule and the preempt-mask, and this way we may also
   specify the Set-Gates-and-Hold and Set-Gates-and-Release commands as
   specified in the Qbu spec:

   $ tc qdisc add dev ens4 parent root handle 100 taprio num_tc 4 \
     	   map 2 2 1 0 3 3 3 3 3 3 3 3 3 3 3 3                    \
	   queues 0 1 2 3                                         \
     	   preemption 0 1 1 1                                     \
	   sched-file preempt_gates.sched

    <file> is multi-line, with each line being of the following format:
    <cmd> <gate mask> <interval in nanoseconds>

    For this case, two new commands are introduced:

    "H" for 'set gates and hold'
    "R" for 'set gates and release'

    H 0x01 300
    R 0x03 500
tc qdisc replace dev eth0 parent root handle 100 taprio \
                     num_tc 3 \
                     map 2 2 1 0 2 2 2 2 2 2 2 2 2 2 2 2 \
                     queues 1@0 1@0 1@0 \
                     base-time 1528743495910289987 \
                     sched-entry S 01 300000 \
                     sched-entry S 02 300000 \
                     sched-entry S 04 400000 \
                     flags 0x1 \
                     txtime-delay 200000 \
                     clockid CLOCK_TAI
The clockid parameter specifies which clock is utilized to set the transmission timestamps from frames.

Below shows the configuration used for the Time Aware Shaper which is used to configure each MAC port in AM64xx device.

              |<------------- Cycle time = 250 us --------------->|
 
              +------------+------------+------------+------------+
              |            |            |            |            |
              +------------+------------+------------+------------+
 
              |<---------->|<---------->|<---------->|<---------->|
Window index       0             1            2            3
Duration         62.5 us       62.5 us      62.5 us      62.5 us
Gate mask      0b00000011    0b00001100   0b00110000   0b11000000



## tsntool 

```
root@riscv-0101:~# tsntool regtool 0 0x11f18 
mmap(0, 4096, 0x3, 0x1, 4, 0x11000)
PCI Memory mapped access 0x 8ED06F18.
READ Value at offset 0x11F18 (0xffff8ed06f18): 0x0
root@riscv-0101:~# 
```

```
tsntool> help

      help              show funciton
      version           show version 
      verbose           debug on/off
      quit              quit 
      tsncapget         get tsn capability 
      qcicapget         get stream parameters 
      qbvset            set time gate scheduling config for <ifname>
      qbvget            <ifname> : get time scheduling entrys for <ifname>
      cbstreamidset     set stream identify table
      cbstreamidget     get stream identify table and counters
      qcisfiset         set stream filter instance 
      qcisfiget         get stream filter instance 
      qcisgiset         set stream gate instance 
      qcisgiget         get stream gate instance 
      qcisficounterget  get stream filter counters
      qcifmiset         set flow metering instance
      qcifmiget         get flow metering instance
      cbsset            set TCs credit-based shaper configure
      cbsget            get TCs credit-based shaper status
      qbuset            set one 8-bits vector showing the preemptable traffic class
      qbugetstatus      get qbu preemption setings
      tsdset            set tsd configure
      tsdget            get tsd configure
      ctset             set cut through queue status
      cbgen             set sequence generate configure
      cbrec             set sequence recover configure
      cbget             get 802.1CB config status
      dscpset           set DSCP to queues and dpl
      sendpkt           send ptp broadcast packet every 5 second
      regtool           register read/write of bar0 of PFs
      ptptool           ptp timer set tool
tsntool> tsncapget --device eno0

echo reply:eno0
tsn: len: 0018 type: 0014 data:
   Qbv 
   Qci 
   Qbu 
   Qav Credit-based Shapter 
   time based schedule 
json structure:
 {
        "Qbv":  "YES",
        "Qci":  "YES",
        "Qbu":  "YES",
        "Qav Credit-based Shapter":     "YES",
        "time based schedule":  "YES"
}
tsntool> 
tsntool> tsncapget --device eno2

echo reply:eno2
tsn: len: 0018 type: 0014 data:
   Qbv 
   Qci 
   Qbu 
   Qav Credit-based Shapter 
   time based schedule 
json structure:
 {
        "Qbv":  "YES",
        "Qci":  "YES",
        "Qbu":  "YES",
        "Qav Credit-based Shapter":     "YES",
        "time based schedule":  "YES"
}
tsntool> tsncapget --device swp0

echo reply:swp0
tsn: len: 0020 type: 0014 data:
   Qbv 
   Qci 
   Qbu 
   Qav Credit-based Shapter 
   8021CB 
   time based schedule 
   cut through forward 
json structure:
 {
        "Qbv":  "YES",
        "Qci":  "YES",
        "Qbu":  "YES",
        "Qav Credit-based Shapter":     "YES",
        "8021CB":       "YES",
        "time based schedule":  "YES",
        "cut through forward":  "YES"
}
tsntool> 
```

### qcisgiget
```
tsntool> qcisgiget  --device eno2

echo reply:eno2
tsn: len: 0024 type: 0007 data:
   max stream filter instances = 30
   max stream gate instances = 30
   max flow meter instances = 20
   supported list max = 4
json structure:
 {
        "max stream filter instances":  48,
        "max stream gate instances":    48,
        "max flow meter instances":     32,
        "supported list max":   4
}
sgi: memsize is 576
echo reply:eno2
tsn: len: 0038 type: 0009 data:
   index = 0
   disable 
  level2: nla->_len: 40 type: 9

   cycle time = 0
   cycle time extend = 0
   basetime = 0
   initial ipv = ff
json structure:
 {
        "index":        0,
        "disable":      "ON",
        "adminentry":   {
                "cycle time":   0,
                "cycle time extend":    0,
                "basetime":     0,
                "initial ipv":  255
        }
}
tsntool> qcisgiget  --device swp0

echo reply:swp0
tsn: len: 0024 type: 0007 data:
   max stream filter instances = b0
   max stream gate instances = b8
   max flow meter instances = b8
   supported list max = 4
json structure:
 {
        "max stream filter instances":  176,
        "max stream gate instances":    184,
        "max flow meter instances":     184,
        "supported list max":   4
}
sgi: memsize is 2208
echo reply:swp0
tsn: len: 0038 type: 0009 data:
   index = 0
   disable 
  level2: nla->_len: 40 type: 9

   cycle time = 0
   cycle time extend = 0
   basetime = 0
   initial ipv = ff
json structure:
 {
        "index":        0,
        "disable":      "ON",
        "adminentry":   {
                "cycle time":   0,
                "cycle time extend":    0,
                "basetime":     0,
                "initial ipv":  255
        }
}
```

### qbuget

```
tsntool> qbuget  --device swp0

echo reply:swp0
tsn: len: 0024 type: 000d data:
   preemtable = 00
   holdadvance = 7f
   releaseadvance = 0
   holdrequest = 00
json structure:
 {
        "preemtable":   0,
        "holdadvance":  127,
        "releaseadvance":       0,
        "holdrequest":  0
}
tsntool> qbuget  --device swp3

echo reply:swp3
tsn: len: 0028 type: 000d data:
   preemtable = 00
   holdadvance = 7f
   releaseadvance = 0
   active 
   holdrequest = 00
json structure:
 {
        "preemtable":   0,
        "holdadvance":  127,
        "releaseadvance":       0,
        "active":       "ON",
        "holdrequest":  0
}
tsntool> qbuget  --device eno2

echo reply:eno2
tsn: len: 0028 type: 000d data:
   preemtable = 00
   holdadvance = 7f
   releaseadvance = 7f
   active 
   holdrequest = 02
json structure:
 {
        "preemtable":   0,
        "holdadvance":  127,
        "releaseadvance":       127,
        "active":       "ON",
        "holdrequest":  2
}
tsntool> 
```

###  preemtable

```
root@riscv-0101:~#  tsntool  qbuget  --device swp3 
echo reply:swp3
tsn: len: 0028 type: 000d data:
   preemtable = 03
   holdadvance = 7f
   releaseadvance = 0
   active 
   holdrequest = 00
json structure:
 {
        "preemtable":   3,
        "holdadvance":  127,
        "releaseadvance":       0,
        "active":       "ON",
        "holdrequest":  0
}
```

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/enable.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/enable2.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/enable3.png)


```
root@riscv-0101:/# cat /proc/kallsyms | grep qbu
ffffa01b9124b460 t felix_qbu_set
ffffa01b9124b700 t felix_qbu_get
ffffa01b9137fdb0 T enetc_qbu_set
ffffa01b91380330 T enetc_qbu_get
```

#### ethtool --show-frame-preemption eno0  

```
root@riscv-0101:/# ethtool --show-frame-preemption eno0   
Frame preemption settings for eno0:
        support: supported
        active: active
        supported queues: 0xff
        supported queues: 0
        minimum fragment size: 64
root@riscv-0101:/# ethtool --show-frame-preemption swp3
Frame preemption settings for swp3:
        support: supported
        active: not active
        supported queues: 0xff
        supported queues: 0
        minimum fragment size: 60
root@riscv-0101:/# 
```

#### frame preemption  ENETC_FPE

```
NETIF_F_HW_PREEMPTION_BIT,	/* Hardware TC frame preemption */
#define NETIF_F_PREEMPTION	__NETIF_F(HW_PREEMPTION)
#define PREEMPTION_DISABLE     0x0
[NETIF_F_HW_PREEMPTION_BIT] =	 "tx-preemption",
```


```
int enetc_qbu_set(struct net_device *ndev, u8 ptvector)
{
	u32 temp;
	int i;
	struct enetc_ndev_priv *priv = netdev_priv(ndev);

	temp = enetc_rd(&priv->si->hw, ENETC_QBV_PTGCR_OFFSET);
	if (temp & ENETC_QBV_TGE)
		enetc_wr(&priv->si->hw, ENETC_QBV_PTGCR_OFFSET,
			 temp & (~ENETC_QBV_TGPE));

	for (i = 0; i < 8; i++) {
		/* 1 Enabled. Traffic is transmitted on the preemptive MAC. */
		temp = enetc_port_rd(&priv->si->hw, ENETC_PTCFPR(i));

		if ((ptvector >> i) & 0x1)
			enetc_port_wr(&priv->si->hw,
				      ENETC_PTCFPR(i),
				      temp | ENETC_FPE);
		else
			enetc_port_wr(&priv->si->hw,
				      ENETC_PTCFPR(i),
				      temp & ~ENETC_FPE);
	}

	return 0;
}
```

#### enetc_qbu_set



```
static struct tsn_ops enetc_tsn_ops_full = {
	.device_init = enetc_tsn_init,
	.device_deinit = enetc_tsn_deinit,
	.get_capability = enetc_tsn_get_capability,
	.qbv_set = enetc_qbv_set,
	.qbv_get = enetc_qbv_get,
	.qbv_get_status = enetc_qbv_get_status,
	.cb_streamid_set = enetc_cb_streamid_set,
	.cb_streamid_get = enetc_cb_streamid_get,
	.cb_streamid_counters_get = enetc_cb_streamid_counters_get,
	.qci_get_maxcap = enetc_get_max_capa,
	.qci_sfi_set = enetc_qci_sfi_set,
	.qci_sfi_get = enetc_qci_sfi_get,
	.qci_sfi_counters_get = enetc_qci_sfi_counters_get,
	.qci_sgi_set = enetc_qci_sgi_set,
	.qci_sgi_get = enetc_qci_sgi_get,
	.qci_sgi_status_get = enetc_qci_sgi_status_get,
	.qci_fmi_set = enetc_qci_fmi_set,
	.qci_fmi_get = enetc_qci_fmi_get,
	.qbu_set = enetc_qbu_set,
	.qbu_get = enetc_qbu_get,
	.cbs_set = enetc_set_cbs,
	.cbs_get = enetc_get_cbs,
	.tsd_set = enetc_set_tsd,
	.tsd_get = enetc_get_tsd,
};

static struct tsn_ops enetc_tsn_ops_part = {
	.device_init = enetc_tsn_init,
	.device_deinit = enetc_tsn_deinit,
	.get_capability = enetc_tsn_get_capability,
	.cb_streamid_set = enetc_cb_streamid_set,
	.cb_streamid_get = enetc_cb_streamid_get,
	.cb_streamid_counters_get = enetc_cb_streamid_counters_get,
	.qci_get_maxcap = enetc_get_max_capa,
	.qci_sfi_set = enetc_qci_sfi_set,
	.qci_sfi_get = enetc_qci_sfi_get,
	.qci_sfi_counters_get = enetc_qci_sfi_counters_get,
	.qci_sgi_set = enetc_qci_sgi_set,
	.qci_sgi_get = enetc_qci_sgi_get,
	.qci_sgi_status_get = enetc_qci_sgi_status_get,
	.qci_fmi_set = enetc_qci_fmi_set,
	.qci_fmi_get = enetc_qci_fmi_get,
};
```

#### ENETC_F_QBU
```
static void enetc_pf_netdev_setup(struct enetc_si *si, struct net_device *ndev,
				  const struct net_device_ops *ndev_ops)
{
	struct enetc_ndev_priv *priv = netdev_priv(ndev);

	SET_NETDEV_DEV(ndev, &si->pdev->dev);
	priv->ndev = ndev;
	priv->si = si;
	priv->dev = &si->pdev->dev;
	si->ndev = ndev;

	priv->msg_enable = (NETIF_MSG_WOL << 1) - 1;
	ndev->netdev_ops = ndev_ops;
	enetc_set_ethtool_ops(ndev);
	ndev->watchdog_timeo = 5 * HZ;
	ndev->max_mtu = ENETC_MAX_MTU;

	ndev->hw_features = NETIF_F_SG | NETIF_F_RXCSUM | NETIF_F_HW_CSUM |
			    NETIF_F_HW_VLAN_CTAG_TX | NETIF_F_HW_VLAN_CTAG_RX |
			    NETIF_F_HW_VLAN_CTAG_FILTER | NETIF_F_LOOPBACK;
	ndev->features = NETIF_F_HIGHDMA | NETIF_F_SG |
			 NETIF_F_RXCSUM | NETIF_F_HW_CSUM |
			 NETIF_F_HW_VLAN_CTAG_TX |
			 NETIF_F_HW_VLAN_CTAG_RX;

	if (si->num_rss)
		ndev->hw_features |= NETIF_F_RXHASH;

	if (si->errata & ENETC_ERR_TXCSUM) {
		ndev->hw_features &= ~NETIF_F_HW_CSUM;
		ndev->features &= ~NETIF_F_HW_CSUM;
	}

	ndev->priv_flags |= IFF_UNICAST_FLT;

	if (si->hw_features & ENETC_SI_F_QBV)
		priv->active_offloads |= ENETC_F_QBV; ///ENETC_F_QBV

	if (si->hw_features & ENETC_SI_F_PSFP && !enetc_psfp_enable(priv)) {
		priv->active_offloads |= ENETC_F_QCI;
		ndev->features |= NETIF_F_HW_TC;
		ndev->hw_features |= NETIF_F_HW_TC;
	}

	if (enetc_tsn_is_enabled() && (si->hw_features & ENETC_SI_F_QBU))
		priv->active_offloads |= ENETC_F_QBU; ///ENETC_F_QBU

	/* pick up primary MAC address from SI */
	enetc_get_primary_mac_addr(&si->hw, ndev->dev_addr);
}

```

### ENETC_F_QBV

```
root@riscv-0101:/# cat /proc/kallsyms | grep qbv
ffffa01b9124c860 t felix_qbv_set
ffffa01b9124cce0 t felix_qbv_get
ffffa01b9124cea0 t felix_qbv_get_status
ffffa01b913804f0 T enetc_qbv_get
ffffa01b913815a0 T enetc_qbv_get_status
ffffa01b91382090 T enetc_qbv_set
ffffa01b91cee8b0 t cmd_qbv_status_get
ffffa01b91ceede0 t tsn_qbv_status_get
ffffa01b91ceee10 t tsn_qbv_get
ffffa01b91cef240 t cmd_qbv_set
ffffa01b91cef570 t tsn_qbv_set
ffffa01b921ef0f0 d qbv_policy
ffffa01b921ef1c0 d qbv_ctrl_policy
ffffa01b921ef230 d qbv_entry_policy
root@riscv-0101:/# 
```

```
static int enetc_setup_taprio(struct net_device *ndev,
			      struct tc_taprio_qopt_offload *admin_conf)
{
 

	if (!err)
		priv->active_offloads |= ENETC_F_QBV;

	return err;
}
```


```
{
 tx_swbd->qbv_en = !!(priv->active_offloads & ENETC_F_QBV);
 tx_swbd->check_wb = tx_swbd->do_twostep_tstamp || tx_swbd->qbv_en;
}
```

# enetc_setup_tc_taprio

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/taprio.png)

```
int enetc_setup_tc_taprio(struct net_device *ndev, void *type_data)
					   priv->tx_ring[i]->index,
					   taprio->enable ? 0 : i);

	/* preemption off if TC priority is all 0 */
	if ((err && taprio->enable) || !(err || taprio->enable))
		enetc_preemption_set(ndev, 0);

	return err;
}
```

## enetc_preemption_set

```
int enetc_preemption_set(struct net_device *ndev, u32 ptvector)
{
       struct enetc_ndev_priv *priv = netdev_priv(ndev);
       u8 tc_num;
       u32 temp;
       int i;

       if (ptvector & ~ENETC_QBU_TC_MASK)
               return -EINVAL;

       temp = enetc_rd(&priv->si->hw, ENETC_QBV_PTGCR_OFFSET);
       if (temp & ENETC_QBV_TGE)
               enetc_wr(&priv->si->hw, ENETC_QBV_PTGCR_OFFSET,
                        temp & (~ENETC_QBV_TGPE));

       tc_num = enetc_get_tc_num(priv->si);

       for (i = 0; i < tc_num; i++) {
               temp = enetc_port_rd(&priv->si->hw, ENETC_PTCFPR(i));

               if ((ptvector >> i) & 0x1)
                       enetc_port_wr(&priv->si->hw,
                                     ENETC_PTCFPR(i),
                                     temp | ENETC_FPE);
               else
                       enetc_port_wr(&priv->si->hw,
                                     ENETC_PTCFPR(i),
                                     temp & ~ENETC_FPE);
       }

       return 0;
}

static u32 enetc_preemption_get(struct net_device *ndev)
{
       struct enetc_ndev_priv *priv = netdev_priv(ndev);
       u32 ptvector = 0;
       u8 tc_num;
       int i;

       /* If preemptable MAC is not enable return 0 */
       if (!(enetc_port_rd(&priv->si->hw, ENETC_PFPMR) & ENETC_PFPMR_PMACE))
               return 0;

       tc_num = enetc_get_tc_num(priv->si);

       for (i = 0; i < tc_num; i++)
               if (enetc_port_rd(&priv->si->hw, ENETC_PTCFPR(i)) & ENETC_FPE)
                       ptvector |= 1 << i;

       return ptvector;
}
```

# pMac eMac MMsl

There are two MACs in ENETC. One is express MAC which traffic
classes in it are advanced transmition. Another is preemptable
MAC which traffic classes are frame preemptable.
The hardware need to initialize the MACs at initial stage.
And then set the preemption enable registers of traffic
classes when ethtool set .get_link_ksettings/.set_link_ksettings stage.

To test the ENETC preemption capability, user need to set mqprio
or taprio to mapping the traffic classes with priorities. Then
use ethtool command to set 'preemption' with a 8 bits value.
MSB represent high number traffic class.

```
enetc_pmac_counters[] = {
	{ ENETC_PM1_RFRM,   "PMAC rx frames" },
	{ ENETC_PM1_RPKT,   "PMAC rx packets" },
	{ ENETC_PM1_RDRP,   "PMAC rx dropped packets" },
	{ ENETC_PM1_RFRG,   "PMAC rx fragment packets" },
	{ ENETC_PM1_TFRM,   "PMAC tx frames" },
	{ ENETC_PM1_TERR,   "PMAC tx error frames" },
	{ ENETC_PM1_TPKT,   "PMAC tx packets" },
	{ ENETC_MAC_MERGE_MMFCRXR,   "MAC merge fragment rx counter" },
	{ ENETC_MAC_MERGE_MMFCTXR,   "MAC merge fragment tx counter"},
};
enetc_port_counters[] = {
	{ ENETC_PM0_REOCT,  "MAC rx ethernet octets" },
	{ ENETC_PM0_RALN,   "MAC rx alignment errors" },
	{ ENETC_PM0_RXPF,   "MAC rx valid pause frames" },
	{ ENETC_PM0_RFRM,   "MAC rx valid frames" },
	{ ENETC_PM0_RFCS,   "MAC rx fcs errors" },
	{ ENETC_PM0_RVLAN,  "MAC rx VLAN frames" },
	{ ENETC_PM0_RERR,   "MAC rx frame errors" },
	{ ENETC_PM0_RUCA,   "MAC rx unicast frames" },
	{ ENETC_PM0_RMCA,   "MAC rx multicast frames" },
	{ ENETC_PM0_RBCA,   "MAC rx broadcast frames" },
	{ ENETC_PM0_RDRP,   "MAC rx dropped packets" },
	{ ENETC_PM0_RPKT,   "MAC rx packets" },
	{ ENETC_PM0_RUND,   "MAC rx undersized packets" },
	{ ENETC_PM0_R64,    "MAC rx 64 byte packets" },
	{ ENETC_PM0_R127,   "MAC rx 65-127 byte packets" },
	{ ENETC_PM0_R255,   "MAC rx 128-255 byte packets" },
	{ ENETC_PM0_R511,   "MAC rx 256-511 byte packets" },
	{ ENETC_PM0_R1023,  "MAC rx 512-1023 byte packets" },
	{ ENETC_PM0_R1518,  "MAC rx 1024-1518 byte packets" },
	{ ENETC_PM0_R1519X, "MAC rx 1519 to max-octet packets" },
	{ ENETC_PM0_ROVR,   "MAC rx oversized packets" },
	{ ENETC_PM0_RJBR,   "MAC rx jabber packets" },
	{ ENETC_PM0_RFRG,   "MAC rx fragment packets" },
	{ ENETC_PM0_RCNP,   "MAC rx control packets" },
	{ ENETC_PM0_RDRNTP, "MAC rx fifo drop" },
	{ ENETC_PM0_TEOCT,  "MAC tx ethernet octets" },
	{ ENETC_PM0_TOCT,   "MAC tx octets" },
	{ ENETC_PM0_TCRSE,  "MAC tx carrier sense errors" },
	{ ENETC_PM0_TXPF,   "MAC tx valid pause frames" },
	{ ENETC_PM0_TFRM,   "MAC tx frames" },
	{ ENETC_PM0_TFCS,   "MAC tx fcs errors" },
	{ ENETC_PM0_TVLAN,  "MAC tx VLAN frames" },
	{ ENETC_PM0_TERR,   "MAC tx frames" },
	{ ENETC_PM0_TUCA,   "MAC tx unicast frames" },
	{ ENETC_PM0_TMCA,   "MAC tx multicast frames" },
	{ ENETC_PM0_TBCA,   "MAC tx broadcast frames" },
	{ ENETC_PM0_TPKT,   "MAC tx packets" },
	{ ENETC_PM0_TUND,   "MAC tx undersized packets" },
	{ ENETC_PM0_T127,   "MAC tx 65-127 byte packets" },
	{ ENETC_PM0_T1023,  "MAC tx 512-1023 byte packets" },
	{ ENETC_PM0_T1518,  "MAC tx 1024-1518 byte packets" },
	{ ENETC_PM0_TCNP,   "MAC tx control packets" },
	{ ENETC_PM0_TDFR,   "MAC tx deferred packets" },
	{ ENETC_PM0_TMCOL,  "MAC tx multiple collisions" },
	{ ENETC_PM0_TSCOL,  "MAC tx single collisions" },
	{ ENETC_PM0_TLCOL,  "MAC tx late collisions" },
	{ ENETC_PM0_TECOL,  "MAC tx excessive collisions" },
	{ ENETC_UFDMF,      "SI MAC nomatch u-cast discards" },
	{ ENETC_MFDMF,      "SI MAC nomatch m-cast discards" },
	{ ENETC_PBFDSIR,    "SI MAC nomatch b-cast discards" },
	{ ENETC_PUFDVFR,    "SI VLAN nomatch u-cast discards" },
	{ ENETC_PMFDVFR,    "SI VLAN nomatch m-cast discards" },
	{ ENETC_PBFDVFR,    "SI VLAN nomatch b-cast discards" },
	{ ENETC_PFDMSAPR,   "SI pruning discarded frames" },
	{ ENETC_PICDR(0),   "ICM DR0 discarded frames" },
	{ ENETC_PICDR(1),   "ICM DR1 discarded frames" },
	{ ENETC_PICDR(2),   "ICM DR2 discarded frames" },
	{ ENETC_PICDR(3),   "ICM DR3 discarded frames" },
};
root@riscv-0101:/# ethtool -S swp0
NIC statistics:
     tx_packets: 0
     tx_bytes: 0
     rx_packets: 0
     rx_bytes: 0
     rx_octets: 0
     rx_unicast: 0
     rx_multicast: 0
     rx_broadcast: 0
     rx_shorts: 0
     rx_fragments: 0
     rx_jabbers: 0
     rx_crc_align_errs: 0
     rx_sym_errs: 0
     rx_frames_below_65_octets: 0
     rx_frames_65_to_127_octets: 0
     rx_frames_128_to_255_octets: 0
     rx_frames_256_to_511_octets: 0
     rx_frames_512_to_1023_octets: 0
     rx_frames_1024_to_1526_octets: 0
     rx_frames_over_1526_octets: 0
     rx_pause: 0
     rx_control: 0
     rx_longs: 0
     rx_classified_drops: 0
     rx_red_prio_0: 0
     rx_red_prio_1: 0
     rx_red_prio_2: 0
     rx_red_prio_3: 0
     rx_red_prio_4: 0
     rx_red_prio_5: 0
     rx_red_prio_6: 0
     rx_red_prio_7: 0
     rx_yellow_prio_0: 0
     rx_yellow_prio_1: 0
     rx_yellow_prio_2: 0
     rx_yellow_prio_3: 0
     rx_yellow_prio_4: 0
     rx_yellow_prio_5: 0
     rx_yellow_prio_6: 0
     rx_yellow_prio_7: 0
     rx_green_prio_0: 0
     rx_green_prio_1: 0
     rx_green_prio_2: 0
     rx_green_prio_3: 0
     rx_green_prio_4: 0
     rx_green_prio_5: 0
     rx_green_prio_6: 0
     rx_green_prio_7: 0
     rx_assembly_err: 0
     rx_smd_err: 0
     rx_assembly_ok: 0
     rx_merge_fragments: 0
     rx_pmac_octets: 0
     rx_pmac_unicast: 0
     rx_pmac_multicast: 0
     rx_pmac_broadcast: 0
     rx_pmac_short: 0
     rx_pmac_fragments: 0
     rx_pmac_jabber: 0
     rx_pmac_crc: 0
     rx_pmac_symbol_err: 0
     rx_pmac_sz_64: 0
     rx_pmac_sz_65_127: 0
     rx_pmac_sz_128_255: 0
     rx_pmac_sz_256_511: 0
     rx_pmac_sz_512_1023: 0
     rx_pmac_sz_1024_1526: 0
     rx_pmac_sz_jumbo: 0
     rx_pmac_pause: 0
     rx_pmac_control: 0
     rx_pmac_long: 0
     tx_octets: 0
     tx_unicast: 0
     tx_multicast: 0
     tx_broadcast: 0
     tx_collision: 0
     tx_drops: 0
     tx_pause: 0
     tx_frames_below_65_octets: 0
     tx_frames_65_to_127_octets: 0
     tx_frames_128_255_octets: 0
     tx_frames_256_511_octets: 0
     tx_frames_1024_1526_octets: 0
     tx_frames_over_1526_octets: 0
     tx_yellow_prio_0: 0
     tx_yellow_prio_1: 0
     tx_yellow_prio_2: 0
     tx_yellow_prio_3: 0
     tx_yellow_prio_4: 0
     tx_yellow_prio_5: 0
     tx_yellow_prio_6: 0
     tx_yellow_prio_7: 0
     tx_green_prio_0: 0
     tx_green_prio_1: 0
     tx_green_prio_2: 0
     tx_green_prio_3: 0
     tx_green_prio_4: 0
     tx_green_prio_5: 0
     tx_green_prio_6: 0
     tx_green_prio_7: 0
     tx_aged: 0
     tx_mm_hold: 0
     tx_merge_fragments: 0
     tx_pmac_octets: 0
     tx_pmac_unicast: 0
     tx_pmac_multicast: 0
     tx_pmac_broadcast: 0
     tx_pmac_pause: 0
     tx_pmac_sz_64: 0
     tx_pmac_sz_65_127: 0
     tx_pmac_sz_128_255: 0
     tx_pmac_sz_256_511: 0
     tx_pmac_sz_512_1023: 0
     tx_pmac_sz_1024_1526: 0
     tx_pmac_sz_jumbo: 0
     drop_local: 0
     drop_tail: 0
     drop_yellow_prio_0: 0
     drop_yellow_prio_1: 0
     drop_yellow_prio_2: 0
     drop_yellow_prio_3: 0
     drop_yellow_prio_4: 0
     drop_yellow_prio_5: 0
     drop_yellow_prio_6: 0
     drop_yellow_prio_7: 0
     drop_green_prio_0: 0
     drop_green_prio_1: 0
     drop_green_prio_2: 0
     drop_green_prio_3: 0
     drop_green_prio_4: 0
     drop_green_prio_5: 0
     drop_green_prio_6: 0
     drop_green_prio_7: 0
root@riscv-0101:/# ethtool -S eno2
NIC statistics:
     SI rx octets: 18013
     SI rx frames: 124
     SI rx u-cast frames: 0
     SI rx m-cast frames: 0
     SI tx octets: 3712
     SI tx frames: 44
     SI tx u-cast frames: 20
     SI tx m-cast frames: 24
     Rx ring  0 discarded frames: 0
     Rx ring  1 discarded frames: 0
     Rx ring  2 discarded frames: 0
     Rx ring  3 discarded frames: 0
     Rx ring  4 discarded frames: 0
     Rx ring  5 discarded frames: 0
     Rx ring  6 discarded frames: 0
     Rx ring  7 discarded frames: 0
     Rx ring  8 discarded frames: 0
     Rx ring  9 discarded frames: 0
     Rx ring 10 discarded frames: 0
     Rx ring 11 discarded frames: 0
     Rx ring 12 discarded frames: 0
     Rx ring 13 discarded frames: 0
     Rx ring 14 discarded frames: 0
     Rx ring 15 discarded frames: 0
     Tx ring  0 frames: 0
     Tx ring  1 frames: 0
     Tx ring  2 frames: 1
     Tx ring  3 frames: 21
     Tx ring  4 frames: 2
     Tx ring  5 frames: 5
     Tx ring  6 frames: 15
     Tx ring  7 frames: 0
     Rx ring  0 frames: 124
     Rx ring  0 alloc errors: 0
     Rx ring  1 frames: 0
     Rx ring  1 alloc errors: 0
     MAC rx ethernet octets: 18509 ///////////////eMac
     MAC rx alignment errors: 0
     MAC rx valid pause frames: 0
     MAC rx valid frames: 124
     MAC rx fcs errors: 0
     MAC rx VLAN frames: 0
     MAC rx frame errors: 0
     MAC rx unicast frames: 0
     MAC rx multicast frames: 0
     MAC rx broadcast frames: 124
     MAC rx dropped packets: 0
     MAC rx packets: 124
     MAC rx undersized packets: 0
     MAC rx 64 byte packets: 0
     MAC rx 65-127 byte packets: 51
     MAC rx 128-255 byte packets: 68
     MAC rx 256-511 byte packets: 5
     MAC rx 512-1023 byte packets: 0
     MAC rx 1024-1518 byte packets: 0
     MAC rx 1519 to max-octet packet: 0
     MAC rx oversized packets: 0
     MAC rx jabber packets: 0
     MAC rx fragment packets: 0
     MAC rx control packets: 0
     MAC rx fifo drop: 0
     MAC tx ethernet octets: 3888
     MAC tx octets: 3888
     MAC tx carrier sense errors: 0
     MAC tx valid pause frames: 0
     MAC tx frames: 44
     MAC tx fcs errors: 0
     MAC tx VLAN frames: 0
     MAC tx frames: 0
     MAC tx unicast frames: 20
     MAC tx multicast frames: 24
     MAC tx broadcast frames: 0
     MAC tx packets: 44
     MAC tx undersized packets: 0
     MAC tx 65-127 byte packets: 40
     MAC tx 512-1023 byte packets: 0
     MAC tx 1024-1518 byte packets: 0
     MAC tx control packets: 0
     MAC tx deferred packets: 0
     MAC tx multiple collisions: 0
     MAC tx single collisions: 0
     MAC tx late collisions: 0
     MAC tx excessive collisions: 0
     SI MAC nomatch u-cast discards: 0
     SI MAC nomatch m-cast discards: 0
     SI MAC nomatch b-cast discards: 0
     SI VLAN nomatch u-cast discards: 0
     SI VLAN nomatch m-cast discards: 0
     SI VLAN nomatch b-cast discards: 0
     SI pruning discarded frames: 0
     ICM DR0 discarded frames: 0
     ICM DR1 discarded frames: 0
     ICM DR2 discarded frames: 0
     ICM DR3 discarded frames: 0
     PMAC rx frames: 0
     PMAC rx packets: 0
     PMAC rx dropped packets: 0
     PMAC rx fragment packets: 0
     PMAC tx frames: 0
     PMAC tx error frames: 0
     PMAC tx packets: 0
     MAC merge fragment rx counter: 0
     MAC merge fragment tx counter: 0
     Tx window drop  0 frames: 0
     Tx window drop  1 frames: 0
     Tx window drop  2 frames: 0
     Tx window drop  3 frames: 0
     Tx window drop  4 frames: 0
     Tx window drop  5 frames: 0
     Tx window drop  6 frames: 0
     Tx window drop  7 frames: 0
     p04_rx_octets: 3888
     p04_rx_unicast: 20
     p04_rx_multicast: 24
     p04_rx_broadcast: 0
     p04_rx_shorts: 0
     p04_rx_fragments: 0
     p04_rx_jabbers: 0
     p04_rx_crc_align_errs: 0
     p04_rx_sym_errs: 0
     p04_rx_frames_below_65_octets: 0
     p04_rx_frames_65_to_127_octets: 40
     p04_rx_frames_128_to_255_octets: 4
     p04_rx_frames_256_to_511_octets: 0
     p04_rx_frames_512_to_1023_octets: 0
     p04_rx_frames_1024_to_1526_octet: 0
     p04_rx_frames_over_1526_octets: 0
     p04_rx_pause: 0
     p04_rx_control: 0
     p04_rx_longs: 0
     p04_rx_classified_drops: 0
     p04_rx_red_prio_0: 0
     p04_rx_red_prio_1: 0
     p04_rx_red_prio_2: 0
     p04_rx_red_prio_3: 0
     p04_rx_red_prio_4: 0
     p04_rx_red_prio_5: 0
     p04_rx_red_prio_6: 0
     p04_rx_red_prio_7: 0
     p04_rx_yellow_prio_0: 0
     p04_rx_yellow_prio_1: 0
     p04_rx_yellow_prio_2: 0
     p04_rx_yellow_prio_3: 0
     p04_rx_yellow_prio_4: 0
     p04_rx_yellow_prio_5: 0
     p04_rx_yellow_prio_6: 0
     p04_rx_yellow_prio_7: 0
     p04_rx_green_prio_0: 24
     p04_rx_green_prio_1: 0
     p04_rx_green_prio_2: 0
     p04_rx_green_prio_3: 0
     p04_rx_green_prio_4: 0
     p04_rx_green_prio_5: 0
     p04_rx_green_prio_6: 0
     p04_rx_green_prio_7: 0
     p04_rx_assembly_err: 0
     p04_rx_smd_err: 0
     p04_rx_assembly_ok: 0
     p04_rx_merge_fragments: 0
     p04_rx_pmac_octets: 0
     p04_rx_pmac_unicast: 0
     p04_rx_pmac_multicast: 0
     p04_rx_pmac_broadcast: 0
     p04_rx_pmac_short: 0
     p04_rx_pmac_fragments: 0
     p04_rx_pmac_jabber: 0
     p04_rx_pmac_crc: 0
     p04_rx_pmac_symbol_err: 0
     p04_rx_pmac_sz_64: 0
     p04_rx_pmac_sz_65_127: 0
     p04_rx_pmac_sz_128_255: 0
     p04_rx_pmac_sz_256_511: 0
     p04_rx_pmac_sz_512_1023: 0
     p04_rx_pmac_sz_1024_1526: 0
     p04_rx_pmac_sz_jumbo: 0
     p04_rx_pmac_pause: 0
     p04_rx_pmac_control: 0
     p04_rx_pmac_long: 0
     p04_tx_octets: 18509
     p04_tx_unicast: 0
     p04_tx_multicast: 0
     p04_tx_broadcast: 124
     p04_tx_collision: 0
     p04_tx_drops: 0
     p04_tx_pause: 0
     p04_tx_frames_below_65_octets: 0
     p04_tx_frames_65_to_127_octets: 51
     p04_tx_frames_128_255_octets: 68
     p04_tx_frames_256_511_octets: 0
     p04_tx_frames_1024_1526_octets: 0
     p04_tx_frames_over_1526_octets: 0
     p04_tx_yellow_prio_0: 0
     p04_tx_yellow_prio_1: 0
     p04_tx_yellow_prio_2: 0
     p04_tx_yellow_prio_3: 0
     p04_tx_yellow_prio_4: 0
     p04_tx_yellow_prio_5: 0
     p04_tx_yellow_prio_6: 0
     p04_tx_yellow_prio_7: 0
     p04_tx_green_prio_0: 124
     p04_tx_green_prio_1: 0
     p04_tx_green_prio_2: 0
     p04_tx_green_prio_3: 0
     p04_tx_green_prio_4: 0
     p04_tx_green_prio_5: 0
     p04_tx_green_prio_6: 0
     p04_tx_green_prio_7: 0
     p04_tx_aged: 0
     p04_tx_mm_hold: 0
     p04_tx_merge_fragments: 0
     p04_tx_pmac_octets: 0
     p04_tx_pmac_unicast: 0
     p04_tx_pmac_multicast: 0
     p04_tx_pmac_broadcast: 0
     p04_tx_pmac_pause: 0
     p04_tx_pmac_sz_64: 0
     p04_tx_pmac_sz_65_127: 0
     p04_tx_pmac_sz_128_255: 0 ///////////////pMac
     p04_tx_pmac_sz_256_511: 0
     p04_tx_pmac_sz_512_1023: 0
     p04_tx_pmac_sz_1024_1526: 0
     p04_tx_pmac_sz_jumbo: 0
     p04_drop_local: 0
     p04_drop_tail: 0
     p04_drop_yellow_prio_0: 0
     p04_drop_yellow_prio_1: 0
     p04_drop_yellow_prio_2: 0
     p04_drop_yellow_prio_3: 0
     p04_drop_yellow_prio_4: 0
     p04_drop_yellow_prio_5: 0
     p04_drop_yellow_prio_6: 0
     p04_drop_yellow_prio_7: 0
     p04_drop_green_prio_0: 0
     p04_drop_green_prio_1: 0
     p04_drop_green_prio_2: 0
     p04_drop_green_prio_3: 0
     p04_drop_green_prio_4: 0
     p04_drop_green_prio_5: 0
     p04_drop_green_prio_6: 0
     p04_drop_green_prio_7: 0
root@riscv-0101:/#
```

## mmsl ENETC_MMCSR

```
static void enetc_configure_port_pmac(struct enetc_hw *hw)
{
	u32 temp;

	/* Set pMAC step lock */
	temp = enetc_port_rd(hw, ENETC_PFPMR);
	enetc_port_wr(hw, ENETC_PFPMR,
		      temp | ENETC_PFPMR_PMACE | ENETC_PFPMR_MWLM);

	temp = enetc_port_rd(hw, ENETC_MMCSR);
	enetc_port_wr(hw, ENETC_MMCSR, temp | ENETC_MMCSR_ME);
}
```

## ENETC_PM0 、ENETC_PM1

***enetc_pf.c***

```
static void enetc_configure_port_mac(struct enetc_hw *hw,
				     phy_interface_t phy_mode)
{
	enetc_port_wr(hw, ENETC_PM0_MAXFRM,
		      ENETC_SET_MAXFRM(ENETC_RX_MAXFRM_SIZE));

	enetc_port_wr(hw, ENETC_PTCMSDUR(0), ENETC_MAC_MAXFRM_SIZE);
	enetc_port_wr(hw, ENETC_PTXMBAR, 2 * ENETC_MAC_MAXFRM_SIZE);

	enetc_port_wr(hw, ENETC_PM0_CMD_CFG, ENETC_PM0_CMD_PHY_TX_EN |
		      ENETC_PM0_CMD_TXP	| ENETC_PM0_PROMISC |
		      ENETC_PM0_TX_EN | ENETC_PM0_RX_EN);////ENETC_PM0

	enetc_port_wr(hw, ENETC_PM1_CMD_CFG, ENETC_PM0_CMD_PHY_TX_EN |
		      ENETC_PM0_CMD_TXP	| ENETC_PM0_PROMISC |
		      ENETC_PM0_TX_EN | ENETC_PM0_RX_EN);////ENETC_PM1
	/* set auto-speed for RGMII */
	if (enetc_port_rd(hw, ENETC_PM0_IF_MODE) & ENETC_PMO_IFM_RG ||
	    phy_mode == PHY_INTERFACE_MODE_RGMII) {
		enetc_port_wr(hw, ENETC_PM0_IF_MODE, ENETC_PM0_IFM_RGAUTO);
		enetc_port_wr(hw, ENETC_PM1_IF_MODE, ENETC_PM0_IFM_RGAUTO);
	}

	if (phy_mode == PHY_INTERFACE_MODE_XGMII ||
	    phy_mode == PHY_INTERFACE_MODE_USXGMII) {
		enetc_port_wr(hw, ENETC_PM0_IF_MODE, ENETC_PM0_IFM_XGMII); ////ENETC_PM0
		enetc_port_wr(hw, ENETC_PM1_IF_MODE, ENETC_PM0_IFM_XGMII);
	}

	/* on LS1028A the MAC Rx FIFO defaults to value 2, which is too high and
	 * may lead to Rx lock-up under traffic.  Set it to 1 instead, as
	 * recommended by the hardware team.
	 */
	enetc_port_wr(hw, ENETC_PM0_RX_FIFO, ENETC_PM0_RX_FIFO_VAL);
}
```

# tsn_ops and qos

```
static struct tsn_ops enetc_tsn_ops_full = {
	.device_init = enetc_tsn_init,
	.device_deinit = enetc_tsn_deinit,
	.get_capability = enetc_tsn_get_capability,
	.qbv_set = enetc_qbv_set,
	.qbv_get = enetc_qbv_get,
	.qbv_get_status = enetc_qbv_get_status,
	.cb_streamid_set = enetc_cb_streamid_set,
	.cb_streamid_get = enetc_cb_streamid_get,
	.cb_streamid_counters_get = enetc_cb_streamid_counters_get,
	.qci_get_maxcap = enetc_get_max_capa,
	.qci_sfi_set = enetc_qci_sfi_set,
	.qci_sfi_get = enetc_qci_sfi_get,
	.qci_sfi_counters_get = enetc_qci_sfi_counters_get,
	.qci_sgi_set = enetc_qci_sgi_set,
	.qci_sgi_get = enetc_qci_sgi_get,
	.qci_sgi_status_get = enetc_qci_sgi_status_get,
	.qci_fmi_set = enetc_qci_fmi_set,
	.qci_fmi_get = enetc_qci_fmi_get,
	.qbu_set = enetc_qbu_set,
	.qbu_get = enetc_qbu_get,
	.cbs_set = enetc_set_cbs,
	.cbs_get = enetc_get_cbs,
	.tsd_set = enetc_set_tsd,
	.tsd_get = enetc_get_tsd,
};
```

# stack

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/enet.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/stack.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/stack2.png)

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/iee1588/tsn/qbu/switch/stack3.png)

# cycle time
IEEE 802.1 Qci全称Per-Stream Filtering and Policing（以下使用简称PSFP），
即对每个数据流采取过滤和控制策略，以确保输入流量符合规范，
从而避免由故障或恶意攻击（如Dos攻击）引起的异常流量问题。
```
     Tx window drop  7 frames: 0
root@riscv-0101:~# tsntool qcisgiget  --device swp2
echo reply:swp2
tsn: len: 0024 type: 0007 data:
   max stream filter instances = b0
   max stream gate instances = b8
   max flow meter instances = b8
   supported list max = 4
json structure:
 {
        "max stream filter instances":  176,
        "max stream gate instances":    184,
        "max flow meter instances":     184,
        "supported list max":   4
}
sgi: memsize is 2208
echo reply:swp2
tsn: len: 0038 type: 0009 data:
   index = 0
   disable 
  level2: nla->_len: 40 type: 9

   cycle time = 0
   cycle time extend = 0
   basetime = 0
   initial ipv = ff
json structure:
 {
        "index":        0,
        "disable":      "ON",
        "adminentry":   {
                "cycle time":   0,
                "cycle time extend":    0,
                "basetime":     0,
                "initial ipv":  255
        }
}
```

```
root@riscv-0101:~# tsntool qcisfiget --device eno2 --index 2
echo reply:eno2
tsn: len: 005c type: 0008 data:
   index = 2
   disable 
   streamhandle = ffffffff
   priority = ff
   gateid = 0

 ==counters==
   match : 0
   pass : 0
   gate_drop : 0
   sdu_pass : 0
   sdu_drop : 0
   red : 0

 === end ===
json structure:
 {
        "index":        2,
        "disable":      "ON",
        "streamhandle": -1,
        "priority":     255,
        "gateid":       0,
        "match":        0,
        "pass": 0,
        "gate_drop":    0,
        "sdu_pass":     0,
        "sdu_drop":     0,
        "red":  0
}
root@riscv-0101:~# tsntool  cbstreamidget --device eno2 --index 2
echo reply:eno2
echo reply:-22
root@riscv-0101:~# tsntool  cbstreamidget --device eno2 --index 1
echo reply:eno2
echo reply:-22
root@riscv-0101:~# 
```

```
sgcl_data->ct = cpu_to_le32(tsn_qci_sgi->admin.cycle_time);
sgcl_data->cte = cpu_to_le32(tsn_qci_sgi->admin.cycle_time_extension);
```

##  stream control list class

```

  stream control list class 9 , cmd 1 data buffer  
struct sgcl_data {
        u32     btl;
        u32 bth;
        u32     ct;
        u32     cte;
        /*struct sgce   *sgcl;*/
};
     dma = dma_map_single(&priv->si->pdev->dev, sgcl_data,
                             data_size, DMA_FROM_DEVICE);
```

```
int enetc_qci_sgi_set(struct net_device *ndev, u32 index,
		      struct tsn_qci_psfp_sgi_conf *tsn_qci_sgi)
{
	sgcl_data->ct = cpu_to_le32(tsn_qci_sgi->admin.cycle_time);
	sgcl_data->cte = cpu_to_le32(tsn_qci_sgi->admin.cycle_time_extension);
		if (!tsn_qci_sgi->admin.base_time) {
		sgcl_data->btl =
			cpu_to_le32(enetc_rd(&priv->si->hw, ENETC_SICTR0));
		sgcl_data->bth =
			cpu_to_le32(enetc_rd(&priv->si->hw, ENETC_SICTR1));
	} else {
		u32 tempu, templ;

		tempu = upper_32_bits(tsn_qci_sgi->admin.base_time);
		templ = lower_32_bits(tsn_qci_sgi->admin.base_time);
		sgcl_data->bth = cpu_to_le32(tempu);
		sgcl_data->btl = cpu_to_le32(templ);
	}

	xmit_cbdr(priv->si, curr_cbd);
}

* PCI IEP device data */
struct enetc_si {
        struct pci_dev *pdev;
        struct enetc_hw hw;
        enum enetc_errata errata;

        struct net_device *ndev; /* back ref. */

        struct enetc_cbdr cbd_ring;

        int num_rx_rings; /* how many rings are available in the SI */
        int num_tx_rings;
        int num_fs_entries;
        int num_rss; /* number of RSS buckets */
        unsigned short pad;
        int hw_features;
#ifdef CONFIG_ENETC_TSN
        struct enetc_cbs *ecbs;
#endif

};

/*                                                              ring by writing the pir register.
 * Update the counters maintained by software.
 */
static int xmit_cbdr(struct enetc_si *si, int i)
{
	struct enetc_cbdr *ring = &si->cbd_ring;
	struct enetc_cbd *dest_cbd;
	int nc, timeout;

	i = (i + 1) % ring->bd_count;

	ring->next_to_use = i;
	/* let H/W know BD ring has been updated */
	enetc_wr_reg(ring->pir, i);

	timeout = ENETC_CBDR_TIMEOUT;

	do {
		if (enetc_rd_reg(ring->cir) == i)
			break;
		usleep_range(10, 20);
		timeout -= 10;
	} while (timeout);

	if (!timeout)
		return -EBUSY;

	nc = ring->next_to_clean;

	while (enetc_rd_reg(ring->cir) != nc) {
		dest_cbd = ENETC_CBD(*ring, nc);
		if (dest_cbd->status_flags & ENETC_CBD_STATUS_MASK)
			WARN_ON(1);

		nc = (nc + 1) % ring->bd_count;
	}

	ring->next_to_clean = nc;

	return 0;
}

```

# qbv

```
root@SIG-0101:~# ethtool -S eno2 | grep 'Tx ring'
     Tx ring  0 frames: 0
     Tx ring  1 frames: 0
     Tx ring  2 frames: 2
     Tx ring  3 frames: 45
     Tx ring  4 frames: 0
     Tx ring  5 frames: 1
     Tx ring  6 frames: 4
     Tx ring  7 frames: 1
root@SIG-0101:~# ethtool -S swp2 | grep 'Tx ring'
root@SIG-0101:~#
root@SIG-0101:~# ethtool -S eno0 | grep 'Tx ring'
     Tx ring  0 frames: 35
     Tx ring  1 frames: 93
     Tx ring  2 frames: 6486
     Tx ring  3 frames: 59
     Tx ring  4 frames: 1242
     Tx ring  5 frames: 695
     Tx ring  6 frames: 78
     Tx ring  7 frames: 77
root@SIG-0101:~# 
```

```
root@riscv-0101:~# tsntool  tsncapget   --device swp2
echo reply:swp2
tsn: len: 0020 type: 0014 data:
   Qbv 
   Qci 
   Qbu 
   Qav Credit-based Shapter 
   8021CB 
   time based schedule 
   cut through forward 
json structure:
 {
        "Qbv":  "YES",
        "Qci":  "YES",
        "Qbu":  "YES",
        "Qav Credit-based Shapter":     "YES",
        "8021CB":       "YES",
        "time based schedule":  "YES",
        "cut through forward":  "YES"
}
```

```
tsntool qcisgiget  --device swp2
root@riscv-0101:~# tsntool qbvget  --device swp2
echo reply:swp2
tsn: len: 002c type: 0005 data:
  level2: nla->_len: 20 type: 11

   listcount = 0
   gatestate = ff
   currenttime = 170a1db5fc66
   listmax = 40
json structure:
 {
        "oper": {
                "listcount":    0,
                "gatestate":    255
        },
        "currenttime":  25332215577702,
        "listmax":      64
}
root@riscv-0101:~# tc -g  class show dev swp2
root@riscv-0101:~# tc -g  class show dev eno2
+---(:8) mq 
+---(:7) mq 
+---(:6) mq 
+---(:5) mq 
+---(:4) mq 
+---(:3) mq 
+---(:2) mq 
+---(:1) mq 

root@riscv-0101:~# tc -g  class show dev swp3
root@riscv-0101:~# tc qdisc  show dev swp3
qdisc noqueue 0: root refcnt 2 
root@riscv-0101:~# tc qdisc  show dev swp2
qdisc noqueue 0: root refcnt 2 
root@riscv-0101:~# tc qdisc  show dev eno2
qdisc mq 0: root 
qdisc pfifo_fast 0: parent :8 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :7 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :6 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :5 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :4 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :3 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :2 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
qdisc pfifo_fast 0: parent :1 bands 3 priomap 1 2 2 2 1 2 0 0 1 1 1 1 1 1 1 1
root@riscv-0101:~# 

```

# references

[qdisc](https://lore.kernel.org/all/20170920055829.bhwn6pd332ldjkeg@localhost/T/)
[09-TSN配置指导](http://www.h3c.com/cn/d_202201/1526943_30005_0.htm)