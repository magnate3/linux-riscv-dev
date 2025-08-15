obj-m := 
#obj-m +=   pci_test2.o
obj-m +=   iommu_test.o
obj-m +=   iommu_test2.o
EXTRA_CFLAGS += -DCONFIG_IOMMU_API=1 -DCONFIG_IOMMU_IOVA=1
#ccflags-y += -DCONFIG_IOMMU_API

all:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) modules
clean:
	make -C /lib/modules/$(shell uname -r)/build \
	M=$(PWD) clean

