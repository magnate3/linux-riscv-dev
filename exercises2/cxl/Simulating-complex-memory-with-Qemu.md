# Soft-Reserved Memory

EFI attributes may be used to mark some memory ranges as "soft-reserved" instead of normal RAM so that the kernel doesn't use them by default. This is useful for memory with different performance that should be reserved to specific uses/applications. They are exposed as DAX by default and possibly as NUMA node later.

### Prerequisites

This requires to boot in UEFI (instead of legacy BIOS), see the Qemu command-line below.
For passing something like efi_fake_mem=1G@4G:0x40000(to mark 4-5GB range as soft-reserved), the kernel must have CONFIG_EFI_FAKE_MEMMAP=y (not enabled in Debian kernels by default).

### Choosing which memory range

The 0-4GB physical memory range is quite complicated when booting Qemu since it contains lots of reserved ranges, including 3-4GB reserved for PCI stuff. It's better to use ranges after 4GB to find large ranges of normal memory. So make the first NUMA node 3GB and use other nodes, they will be mapped after the PCI stuff, after 4GB.

If a single memory range is marked as soft-reserved but covers multiple nodes, strange things happen, the kernel creates a DAX covering both (with locality of the first) but fails to entirely register it, and then creates separated DAX as expected. To avoid issues, it's better to specify two ranges even if they are consecutive.

### Configuring Qemu with 2 NUMAs + 2 CPU-less NUMA

```
kvm \
 -drive if=pflash,format=raw,file=./OVMF.fd \
 -drive media=disk,format=qcow2,file=efi.qcow2 \
 -smp 4 -m 6G \
 -object memory-backend-ram,size=3G,id=m0 \
 -object memory-backend-ram,size=1G,id=m1 \
 -object memory-backend-ram,size=1G,id=m2 \
 -object memory-backend-ram,size=1G,id=m3 \
 -numa node,nodeid=0,memdev=m0,cpus=0-1 \
 -numa node,nodeid=1,memdev=m1,cpus=2-3 \
 -numa node,nodeid=2,memdev=m2 \
 -numa node,nodeid=3,memdev=m3
```

OVMF is required for booting in UEFI mode (during both VM install and later).

### Marking NUMA nodes as soft-reserved and getting hmem DAX device

On the kernel boot command-line, pass `efi_fake_mem=1G@4G:0x40000,1G@6G:0x40000` to make NUMA node#1 (one with CPUs) and #3 (CPU-less) as soft-reserved. Their memory disappears, and a DAX device appears.

```
% cat /proc/iomem
100000000-13fffffff : hmem.0              <- node #1 is soft-reserved
  100000000-13fffffff : Soft Reserved
    100000000-13fffffff : dax0.0
140000000-17fffffff : System RAM          <- node #2 is normal memory
180000000-1bfffffff : hmem.1              <- node #3 is soft-reserved
  180000000-1bfffffff : Soft Reserved
    180000000-1bfffffff : dax1.0
```

Those DAX devices under /sys/bus/dax/devices point to platform hmem devices but there isn't much useless in there.
```
dax0.0 -> ../../../devices/platform/hmem.0/dax0.0
dax1.0 -> ../../../devices/platform/hmem.1/dax1.0
```

dax0.0 has `target_node=numa_node=1` in its sysfs attributes because node1 is online thanks to existing CPUs.

dax1.0 is offline since it contains neither CPUs nor RAM. It has `target_node=3` as expected, but `numa_node=0` since this must be a online node during boot. node#0 was chosen because it's close (we didn't specify any distance matrix on the Qemu command-line, the default 10=local, 20=remote is used, hence 20 is the minimal distance from node#3 to online nodes, and node#0 is the first one of those).

### Making NUMA nodes out of soft-reserved memory

```
% daxctl reconfigure-device --mode=system-ram all
% cat /proc/iomem
[...]
100000000-13fffffff : hmem.0
  100000000-13fffffff : Soft Reserved
    100000000-13fffffff : dax0.0
      100000000-13fffffff : System RAM (kmem) <- node#1 is back as a NUMA node
140000000-17fffffff : System RAM
180000000-1bfffffff : hmem.1
  180000000-1bfffffff : Soft Reserved
    180000000-1bfffffff : dax1.0
      180000000-1bfffffff : System RAM (kmem) <- node#3 is back as a NUMA node
```

# NVDIMMs

### NVDIMMs in Qemu

Add `-machine pc,nvdimm=on` to qemu to enable nvdimms, then make `maxmem` in `-m` equal to RAM+NVDIMM size, and `slots` in `-m` equal to number of NVDIMMs. Then create the object and device, for instance:

```
kvm \
 -machine pc,nvdimm=on \
 -drive if=pflash,format=raw,file=./OVMF.fd \
 -drive media=disk,format=qcow2,file=efi.qcow2 \
 -smp 4 \
 -m 6G,slots=1,maxmem=7G \
 -object memory-backend-ram,size=3G,id=ram0 \
 -object memory-backend-ram,size=1G,id=ram1 \
 -object memory-backend-ram,size=1G,id=ram2 \
 -object memory-backend-ram,size=1G,id=ram3 \
 -numa node,nodeid=0,memdev=ram0,cpus=0-1 \
 -numa node,nodeid=1,memdev=ram1,cpus=2-3 \
 -numa node,nodeid=2,memdev=ram2 \
 -numa node,nodeid=3,memdev=ram3 \
 -numa node,nodeid=4 \
 -object memory-backend-file,id=nvdimm1,share=on,mem-path=nvdimm.img,size=1G \
 -device nvdimm,id=nvdimm1,memdev=nvdimm1,unarmed=off,node=4
```

### DAX and NUMA node in Linux

You'll get a `pmem0` in Linux, from namespace2.0 (likely not 0.0 because dax0.0 and dax1.0 are used for soft-reserved memory in this config):
```
% ndctl list
[
  {
    "dev":"namespace2.0",
    "mode":"fsdax",
    "map":"dev",
    "size":1054867456,
    "uuid":"937b5655-a581-4961-bbbc-f6a567a86b0f",
    "sector_size":512,
    "align":2097152,
    "blockdev":"pmem2"
  }
]
```

Convert it to DAX with
```
% ndctl create-namespace -f -e namespace2.0 -p pmem -t devdax
```

That DAX points to ndctl region device now:
```
/sys/bus/dax/devices/dax2.0 -> ../../../devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0012:00/ndbus0/region2/dax2.0/dax2.0
```
That region contains a single mapping since there's only one NVDIMM here, and its type is `nvdimm`:
```
% cat /sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0012:00/ndbus0/region2/mappings
1
% cat /sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0012:00/ndbus0/region2/mapping0
nmem0,0,1073741824,0
% cat /sys/devices/LNXSYSTM:00/LNXSYBUS:00/ACPI0012:00/ndbus0/nmem0/devtype
nvdimm
```

As usual, that DAX can be made a NUMA node:
```
% daxctl reconfigure-device --mode=system-ram dax2.0
```

# NUMA topology and memory performance

See also https://futurewei-cloud.github.io/ARM-Datacenter/qemu/how-to-configure-qemu-numa-nodes/

### SLIT distances

All values must be given individually. To make node#2 (HBM) and node#4 (NVDIMM) close to node#0, and node#3 (HBM) close to node#1:
```
 -numa dist,src=0,dst=0,val=10 -numa dist,src=0,dst=1,val=20 -numa dist,src=0,dst=2,val=12 -numa dist,src=0,dst=3,val=22 -numa dist,src=0,dst=4,val=15 \
-numa dist,src=1,dst=0,val=20 -numa dist,src=1,dst=1,val=10 -numa dist,src=1,dst=2,val=22 -numa dist,src=1,dst=3,val=12 -numa dist,src=1,dst=4,val=25 \
-numa dist,src=2,dst=0,val=12 -numa dist,src=2,dst=1,val=22 -numa dist,src=2,dst=2,val=10 -numa dist,src=2,dst=3,val=25 -numa dist,src=2,dst=4,val=30 \
-numa dist,src=3,dst=0,val=22 -numa dist,src=3,dst=1,val=12 -numa dist,src=3,dst=2,val=25 -numa dist,src=3,dst=3,val=10 -numa dist,src=3,dst=4,val=30 \
-numa dist,src=4,dst=0,val=15 -numa dist,src=4,dst=1,val=25 -numa dist,src=4,dst=2,val=30 -numa dist,src=4,dst=3,val=30 -numa dist,src=4,dst=4,val=10
```

### HMAT initiators

Before Qemu 7.2, `-machine hmat=on` required an `initiator=X` attribute for each NUMA node, which means latency/bandwidth aren't used by Linux to define best initiators. Fixed in 7.2, initiator=X is now optional, so that we can have a memory with 2 best initiators.

### HMAT performance attributes

HMAT latency/bandwidth example for 2 CPU+memory nodes, and a third node with same (slow) performance to both CPUs:
```
qemu-system-x86_64 -accel kvm \
 -machine pc,hmat=on \
 -drive if=pflash,format=raw,file=./OVMF.fd \
 -drive media=disk,format=qcow2,file=efi.qcow2 \
 -smp 4 \
 -m 3G \
 -object memory-backend-ram,size=1G,id=ram0 \
 -object memory-backend-ram,size=1G,id=ram1 \
 -object memory-backend-ram,size=1G,id=ram2 \
 -numa node,nodeid=0,memdev=ram0,cpus=0-1 \
 -numa node,nodeid=1,memdev=ram1,cpus=2-3 \
 -numa node,nodeid=2,memdev=ram2 \
 -numa hmat-lb,initiator=0,target=0,hierarchy=memory,data-type=access-latency,latency=10 \
 -numa hmat-lb,initiator=0,target=0,hierarchy=memory,data-type=access-bandwidth,bandwidth=10485760 \
 -numa hmat-lb,initiator=0,target=1,hierarchy=memory,data-type=access-latency,latency=20 \
 -numa hmat-lb,initiator=0,target=1,hierarchy=memory,data-type=access-bandwidth,bandwidth=5242880 \
 -numa hmat-lb,initiator=0,target=2,hierarchy=memory,data-type=access-latency,latency=30 \
 -numa hmat-lb,initiator=0,target=2,hierarchy=memory,data-type=access-bandwidth,bandwidth=1048576 \
 -numa hmat-lb,initiator=1,target=0,hierarchy=memory,data-type=access-latency,latency=20 \
 -numa hmat-lb,initiator=1,target=0,hierarchy=memory,data-type=access-bandwidth,bandwidth=5242880 \
 -numa hmat-lb,initiator=1,target=1,hierarchy=memory,data-type=access-latency,latency=10 \
 -numa hmat-lb,initiator=1,target=1,hierarchy=memory,data-type=access-bandwidth,bandwidth=10485760 \
 -numa hmat-lb,initiator=1,target=2,hierarchy=memory,data-type=access-latency,latency=30 \
 -numa hmat-lb,initiator=1,target=2,hierarchy=memory,data-type=access-bandwidth,bandwidth=1048576
```
Before the Qemu patch, `memdev=ram2` must be followed by `initiator=0` or `initiator=1`.

# CXL

Linux 6.3 will most of the required patches for PMEM and RAM regions. Qemu branch cxl-2023-01-26 from https://gitlab.com/jic23/qemu works.

### Single PMEM device on single 4-core socket

First we need a CXL hostbridge (pxb-cxl "cxl.1" here), then we attach a root-port (cxl-rp "root_port13" here), then a Type 3 device.

In this case is PMEM device so it needs two "memory-backend-file" objects, one for the memory ("pmem0" here) and one for its label storage area ("cxl-lsa0" here). Finally we need a Fixed Memory Window (cxl-fwm) to map that memory in the host.

There can be multiple PXBs, multiple RPs behind each PXB, but only one device behind each RP unless we place CXL switches between RP and devices.

```
qemu-system-x86_64 \
  -machine q35,accel=kvm,nvdimm=on,cxl=on \
  -drive if=pflash,format=raw,file=$FILES/OVMF.fd \
  -drive media=disk,format=qcow2,file=$FILES/efi.qcow2 \
  -device e1000,netdev=net0,mac=52:54:00:12:34:56 \
  -netdev user,id=net0,hostfwd=tcp::10022-:22 \
  -m 4G,slots=8,maxmem=8G \
  -smp 4 \
  -object memory-backend-ram,size=4G,id=mem0 \
  -numa node,nodeid=0,cpus=0-3,memdev=mem0 \
  -object memory-backend-file,id=pmem0,share=on,mem-path=/tmp/cxltest.raw,size=256M \
  -object memory-backend-file,id=cxl-lsa0,share=on,mem-path=/tmp/lsa.raw,size=256M \
  -device pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.1 \
  -device cxl-rp,port=0,bus=cxl.1,id=root_port13,chassis=0,slot=2 \
  -device cxl-type3,bus=root_port13,persistent-memdev=pmem0,lsa=cxl-lsa0,id=cxl-pmem0 \
  -M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G
```

### Two RAM devices on single 4-core socket, on different PXBs

RAM devices need only one Qemu object ("memory-backend-ram" since its volatile) since they have no LSA.

We define 2 FWM, one per PXB. We'll get one region per device.

Devices are attached to different PXBs here, with a single RP each, although that's not required (see below).

```
qemu-system-x86_64 \
  -machine q35,accel=kvm,nvdimm=on,cxl=on \
  -drive if=pflash,format=raw,file=$FILES/OVMF.fd \
  -drive media=disk,format=qcow2,file=$FILES/efi.qcow2 \
  -device e1000,netdev=net0,mac=52:54:00:12:34:56 \
  -netdev user,id=net0,hostfwd=tcp::10022-:22 \
  -m 2G,slots=8,maxmem=6G \
  -smp cpus=4,cores=2,sockets=2 \
  -object memory-backend-ram,size=1G,id=mem0 \
  -numa node,nodeid=0,cpus=0-1,memdev=mem0 \
  -object memory-backend-ram,size=1G,id=mem1 \
  -numa node,nodeid=1,cpus=2-3,memdev=mem1 \
  -object memory-backend-ram,id=vmem0,share=on,size=256M \
  -device pxb-cxl,numa_node=0,bus_nr=12,bus=pcie.0,id=cxl.1 \
  -device cxl-rp,port=0,bus=cxl.1,id=root_port13,chassis=0,slot=2 \
  -device cxl-type3,bus=root_port13,volatile-memdev=vmem0,id=cxl-vmem0 \
  -object memory-backend-ram,id=vmem1,share=on,size=256M \
  -device pxb-cxl,numa_node=1,bus_nr=14,bus=pcie.0,id=cxl.2 \
  -device cxl-rp,port=0,bus=cxl.2,id=root_port14,chassis=1,slot=2 \
  -device cxl-type3,bus=root_port14,volatile-memdev=vmem1,id=cxl-vmem1 \
  -M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G,cxl-fmw.1.targets.0=cxl.2,cxl-fmw.1.size=4G
```

### Two RAM devices on single 4-core socket, on single PXB

Same as above but we don't want two PXBs. We have 2 RPs on the same PXB, and one device on each RP.

Another solution would be to have a single RP and a switch (see below). However Linux (at least 6.3) doesn't correct propagate decoders unless we have a second RP, hence we'd need to add a second RP anyway.

```
qemu-system-x86_64 \
  -machine q35,accel=kvm,nvdimm=on,cxl=on \
  -drive if=pflash,format=raw,file=$FILES/OVMF.fd \
  -drive media=disk,format=qcow2,file=$FILES/efi.qcow2 \
  -device e1000,netdev=net0,mac=52:54:00:12:34:56 \
  -netdev user,id=net0,hostfwd=tcp::10022-:22 \
  -m 2G,slots=8,maxmem=6G \
  -smp cpus=4,cores=2,sockets=2 \
  -object memory-backend-ram,size=1G,id=mem0 \
  -numa node,nodeid=0,cpus=0-1,memdev=mem0 \
  -object memory-backend-ram,size=1G,id=mem1 \
  -numa node,nodeid=1,cpus=2-3,memdev=mem1 \
  -object memory-backend-ram,id=vmem0,share=on,size=256M \
  -object memory-backend-ram,id=vmem1,share=on,size=256M \
  -device pxb-cxl,numa_node=0,bus_nr=12,bus=pcie.0,id=cxl.1 \
  -device cxl-rp,port=0,bus=cxl.1,id=root_port13,chassis=0,slot=2 \
  -device cxl-type3,bus=root_port13,volatile-memdev=vmem0,id=cxl-vmem0 \
  -device cxl-rp,port=0,bus=cxl.1,id=root_port14,chassis=1,slot=2 \
  -device cxl-type3,bus=root_port14,volatile-memdev=vmem1,id=cxl-vmem1 \
  -M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G
```

There's a single FWM here, can be used either as a single interleaved region or 2 separate regions.

### Single socket with CXL switch in front of 2 switches with 2 regions each, but single region usable

Below the PXB+RP, we create a switch 'interm0' with 'cxl-upstream'. Then we define two ports 'intermport0' and 'intermport1' below that switch with 'cxl-downstream'. Below each port, we attach one switch ('us0' and 'us1' respectively). And finally we put one RAM device and one PMEM device below 'us0', and one RAM device and one RAM+PMEM device below 'us1'.

Single FWM, hence no way to use both RAM and PMEM simultaneously. But multiple RAM (or PMEM respectively) devices may be used simultaneously either as separed RAM regions (or PMEM respectively) or a single interleaved one.

```
qemu-system-x86_64 \
  -machine q35,accel=kvm,nvdimm=on,cxl=on \
  -drive if=pflash,format=raw,file=$FILES/OVMF.fd \
  -drive media=disk,format=qcow2,file=$FILES/efi.qcow2 \
  -device e1000,netdev=net0,mac=52:54:00:12:34:56 \
  -netdev user,id=net0,hostfwd=tcp::10022-:22 \
  -m 4G,slots=8,maxmem=8G \
  -smp 4 \
  -object memory-backend-ram,size=4G,id=mem0 \
  -numa node,nodeid=0,cpus=0-3,memdev=mem0 \
  -object memory-backend-ram,id=cxl-mem0,share=on,size=256M \
  -object memory-backend-file,id=cxl-pmem1,share=on,mem-path=/tmp/cxltest1.raw,size=256M \
  -object memory-backend-ram,id=cxl-mem2,share=on,size=256M \
  -object memory-backend-ram,id=cxl-mem3,share=on,size=256M \
  -object memory-backend-file,id=cxl-pmem3,share=on,mem-path=/tmp/cxltest3.raw,size=256M \
  -object memory-backend-file,id=cxl-lsa1,share=on,mem-path=/tmp/lsa1.raw,size=256M \
  -object memory-backend-file,id=cxl-lsa3,share=on,mem-path=/tmp/lsa3.raw,size=256M \
  -device pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.1 \
  -device cxl-rp,port=0,bus=cxl.1,id=root_port0,chassis=0,slot=0 \
  -device cxl-upstream,bus=root_port0,id=interm0 \
  -device cxl-downstream,port=0,bus=interm0,id=intermport0,chassis=1,slot=1 \
  -device cxl-downstream,port=1,bus=interm0,id=intermport1,chassis=1,slot=2 \
  -device cxl-upstream,bus=intermport0,id=us0 \
  -device cxl-downstream,port=0,bus=us0,id=swport0,chassis=2,slot=4 \
  -device cxl-type3,bus=swport0,volatile-memdev=cxl-mem0,id=cxl-mem0 \
  -device cxl-downstream,port=1,bus=us0,id=swport1,chassis=2,slot=5 \
  -device cxl-type3,bus=swport1,persistent-memdev=cxl-pmem1,lsa=cxl-lsa1,id=cxl-pmem1 \
  -device cxl-upstream,bus=intermport1,id=us1 \
  -device cxl-downstream,port=2,bus=us1,id=swport2,chassis=3,slot=6 \
  -device cxl-type3,bus=swport2,volatile-memdev=cxl-mem2,id=cxl-mem2 \
  -device cxl-downstream,port=3,bus=us1,id=swport3,chassis=3,slot=7 \
  -device cxl-type3,bus=swport3,volatile-memdev=cxl-mem3,persistent-memdev=cxl-pmem3,lsa=cxl-lsa3,id=cxl-pmem3 \
  -M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G
```

### Two sockets with one CXL switch each, and 3 devices between each switch, with 4 FWM

4 PXB. 1st and 4th have a single device. 2nd and 3rd have a switch with 2 devices each.

4 FWM, one per PXB.

```
qemu-system-x86_64 \
  -machine q35,accel=kvm,nvdimm=on,cxl=on \
  -drive if=pflash,format=raw,file=$FILES/OVMF.fd \
  -drive media=disk,format=qcow2,file=$FILES/efi.qcow2 \
  -device e1000,netdev=net0,mac=52:54:00:12:34:56 \
  -netdev user,id=net0,hostfwd=tcp::10022-:22 \
  -m 2G,slots=8,maxmem=6G \
  -smp cpus=4,cores=2,sockets=2 \
  -object memory-backend-ram,size=1G,id=mem0 \
  -numa node,nodeid=0,cpus=0-1,memdev=mem0 \
  -object memory-backend-ram,size=1G,id=mem1 \
  -numa node,nodeid=1,cpus=2-3,memdev=mem1 \
\
  -object memory-backend-ram,id=cxl-mem0,share=on,size=256M \
  -device pxb-cxl,numa_node=0,bus_nr=16,bus=pcie.0,id=cxl.0 \
  -device cxl-rp,port=0,bus=cxl.0,id=root_port16,chassis=0,slot=0 \
  -device cxl-type3,bus=root_port16,volatile-memdev=cxl-mem0,id=cxl-mem0 \
\
  -object memory-backend-file,id=cxl-pmem1,share=on,mem-path=/tmp/cxltest1.raw,size=256M \
  -object memory-backend-file,id=cxl-lsa1,share=on,mem-path=/tmp/lsa1.raw,size=256M \
  -object memory-backend-ram,id=cxl-mem2,share=on,size=256M \
  -object memory-backend-file,id=cxl-pmem2,share=on,mem-path=/tmp/cxltest2.raw,size=256M \
  -object memory-backend-file,id=cxl-lsa2,share=on,mem-path=/tmp/lsa2.raw,size=256M \
  -device pxb-cxl,numa_node=0,bus_nr=24,bus=pcie.0,id=cxl.1 \
  -device cxl-rp,port=0,bus=cxl.1,id=root_port24,chassis=1,slot=0 \
  -device cxl-upstream,bus=root_port24,id=sw0 \
  -device cxl-downstream,port=1,bus=sw0,id=sw0port1,chassis=1,slot=1 \
  -device cxl-type3,bus=sw0port1,persistent-memdev=cxl-pmem1,lsa=cxl-lsa1,id=cxl-pmem1 \
  -device cxl-downstream,port=2,bus=sw0,id=sw0port2,chassis=1,slot=2 \
  -device cxl-type3,bus=sw0port2,volatile-memdev=cxl-mem2,persistent-memdev=cxl-pmem2,lsa=cxl-lsa2,id=cxl-pmem2 \
\
  -device pxb-cxl,numa_node=1,bus_nr=32,bus=pcie.0,id=cxl.2 \
  -device cxl-rp,port=0,bus=cxl.2,id=root_port32,chassis=2,slot=0 \
  -device cxl-upstream,bus=root_port32,id=sw1 \
  -device cxl-downstream,port=0,bus=sw1,id=sw1port0,chassis=2,slot=1 \
  -object memory-backend-ram,id=cxl-mem3,share=on,size=256M \
  -device cxl-type3,bus=sw1port0,volatile-memdev=cxl-mem3,id=cxl-mem3 \
  -device cxl-downstream,port=1,bus=sw1,id=sw1port1,chassis=2,slot=2 \
  -object memory-backend-ram,id=cxl-mem4,share=on,size=256M \
  -device cxl-type3,bus=sw1port1,volatile-memdev=cxl-mem4,id=cxl-mem4 \
\
  -object memory-backend-file,id=cxl-pmem5,share=on,mem-path=/tmp/cxltest5.raw,size=256M \
  -object memory-backend-file,id=cxl-lsa5,share=on,mem-path=/tmp/lsa5.raw,size=256M \
  -device pxb-cxl,numa_node=1,bus_nr=40,bus=pcie.0,id=cxl.3 \
  -device cxl-rp,port=0,bus=cxl.3,id=root_port40,chassis=3,slot=0 \
  -device cxl-type3,bus=root_port40,persistent-memdev=cxl-pmem5,lsa=cxl-lsa5,id=cxl-pmem5 \
  -M \
cxl-fmw.0.targets.0=cxl.0,cxl-fmw.0.size=4G,\
cxl-fmw.1.targets.0=cxl.1,cxl-fmw.1.size=4G,\
cxl-fmw.2.targets.0=cxl.2,cxl-fmw.2.size=4G,\
cxl-fmw.3.targets.0=cxl.3,cxl-fmw.3.size=4G
```

### Interleaving across PXBs

It's possible to interleave regions across multiple devices, very easily when on the same PXB, but also on different PXB. In the latter case, the FWM must specify multiple targets and interleaving commands (see below) will have to strictly match these targets.

This configuration will interleave the first FWM between buses cxl.1 and cxl.2 with granularity 8k:
```
-M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.targets.1=cxl.2,cxl-fmw.0.size=4G,cxl-fmw.0.interleave-granularity=8k
```

### Using CXL devices on Linux

The CXL command-line tool comes with ndctl. However release 0.76 doesn't contain support for volatile regions yet. Use the vv/volatile-regions branch from the https://github.com/pmem/ndctl

To see the list of devices:
```
$ cxl list -M
```

To see the list of decoders:
```
$ cxl list -D
```

#### Decoders

Using regions requires a "decoder" at each level of the hierarchy. In practice, only the "root" decoder is required at region creation. Other level will get their decoder automatically (except on Linux 6.3 where single RP can cause some decoder shortage...).

The cxl tool is supposed to find the right decoder automatically but it doesn't do it yet, hence we have to manually specifying it.

Root decoders are decoder0.0 ... decoder0.3 if there are 4 PXB. To see which PCI bus is associated with a decoder, run `cxl list -v` and look for `decoders:root`. Attribute `"target":"pci0000:10"` means all PCI/CXL devices behing that bus may use that decoder.

Otherwise just try decoder0.X until it works.

#### RAM regions

To create a region using RAM device "mem0" and decoder "decoder0.0":
```
$ cxl create-region -m -t ram -d decoder0.0 mem0
```
To create a region interleaved between 2 RAM device "mem0" and "mem1" and decoder "decoder0.0":
```
$ cxl create-region -m -t ram -d decoder0.0 -w 2 mem0 mem1
```
Once a RAM region is created, a new DAX device should appear under /sys/bus/dax/devices. Linux 6.3 may make it an offline NUMA node by default. Use `daxctl online-memory dax0.0` to make it online. Older kernels may make it a DAX device only instead. Use `daxctl reconfigure-device --mode=devdax dax0.0` to make it an (online) NUMA node.

A single decoder may be used for multiple regions, but it cannot mix RAM and PMEM regions.

#### PMEM regions

If using PMEM device, change into `-t pmem` (and change the decoder if already used for a RAM region).
```
$ cxl create-region -m -t pmem -d decoder0.1 mem1
```
PMEM regions need a namespace before they are usable:
```
$ ndctl create-namespace -t pmem -m devdax -r region0 -f
```
This will create DAX device such as dax0.0, which is not converted to NUMA node automatically. Use `daxctl reconfigure-device --mode=devdax dax0.0` to do so.
