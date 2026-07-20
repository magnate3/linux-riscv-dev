


```
static int amdxdna_drm_open(struct drm_device *ddev, struct drm_file *filp)
{

	client->sva = iommu_sva_bind_device(xdna->ddev.dev, current->mm);

	client->pasid = iommu_sva_get_pasid(client->sva);

}

```

ATS    PRI    PASID   
```
$ lspci -v
...
0a:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Ellesmere [Radeon RX 470/480] (rev e7) (prog-if 00 [VGA controller])
	Subsystem: XFX Pine Group Inc. Ellesmere [Radeon RX 470/480/570/570X/580/580X/590]
	Flags: bus master, fast devsel, latency 0, IRQ 160
	Memory at c0000000 (64-bit, prefetchable) [size=256M]
	Memory at d0000000 (64-bit, prefetchable) [size=2M]
	I/O ports at f000 [size=256]
	Memory at fce00000 (32-bit, non-prefetchable) [size=256K]
	Expansion ROM at fce40000 [disabled] [size=128K]
	Capabilities: [48] Vendor Specific Information: Len=08 <?>
	Capabilities: [50] Power Management version 3
	Capabilities: [58] Express Legacy Endpoint, MSI 00
	Capabilities: [a0] MSI: Enable+ Count=1/1 Maskable- 64bit+
	Capabilities: [100] Vendor Specific Information: ID=0001 Rev=1 Len=010 <?>
	Capabilities: [150] Advanced Error Reporting
	Capabilities: [200] #15
	Capabilities: [270] #19
	Capabilities: [2b0] Address Translation Service (ATS)
	Capabilities: [2c0] Page Request Interface (PRI)
	Capabilities: [2d0] Process Address Space ID (PASID)
	Capabilities: [320] Latency Tolerance Reporting
	Capabilities: [328] Alternative Routing-ID Interpretation (ARI)
	Capabilities: [370] L1 PM Substates
	Kernel driver in use: vfio-pci
	Kernel modules: amdgpu
```