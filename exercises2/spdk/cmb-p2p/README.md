



# The P2P API for NVMe
The functions that provide access to the NVMe CMBs for P2P capabilities are given in the table below.

Key Functions	Description   
```
spdk_nvme_ctrlr_map_cmb()	Map a previously reserved controller memory buffer so that it's data is visible from the CPU.
spdk_nvme_ctrlr_unmap_cmb()	Free a controller memory I/O buffer.
spdk_nvme_ctrlr_get_regs_cmbsz()	Get the NVMe controller CMBSZ (Controller Memory Buffer Size) register.
```


# Determining device support
SPDK's identify example application displays whether a device has a controller memory buffer and which operations it supports. Run it as follows:

spdk_nvme_identify -r traddr:<pci id of ssd>
# cmb_copy: An example P2P Application
Run the cmb_copy example application.

./build/examples/cmb_copy -r <pci id of write ssd>-1-0-1 -w <pci id of write ssd>-1-0-1 -c <pci id of the ssd with cmb>

This should copy a single LBA (LBA 0) from namespace 1 on the read NVMe SSD to LBA 0 on namespace 1 on the write SSD using the CMB as the DMA buffer.