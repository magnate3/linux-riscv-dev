
obj-$(CONFIG_BLK_DEV_NVME)     += nvme.o nv_pinbuf.o

lightnvm-$(CONFIG_NVM)			:= lightnvm.o
nvme-y					+= core.o pci.o $(lightnvm-y)
nvme-$(CONFIG_BLK_DEV_NVME_SCSI)        += scsi.o
nv_pinbuf-y := 
ccflags-y := -I$(src)/include -I$(NVIDIA_SRC)