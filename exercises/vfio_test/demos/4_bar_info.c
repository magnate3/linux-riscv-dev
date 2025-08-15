#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <endian.h>
#include <linux/vfio.h>
#include <linux/pci_regs.h>
#include <sys/mman.h>
#include <inttypes.h>
#include <sys/types.h>
#include <linux/vfio.h>
/* HW interface registers */
#define HINIC_CSR_FUNC_ATTR0_ADDR                       0x0
#define HINIC_CSR_FUNC_ATTR1_ADDR                       0x4
#define VFIO_GET_REGION_ADDR(x) ((uint64_t) x << 40ULL)
#define rte_bswap32(x) __builtin_bswap32(x)
#define rte_be_to_cpu_32(x) rte_bswap32(x)
#define rte_wmb() do { asm volatile ("dmb st" : : : "memory"); } while (0)
#define rte_rmb() __sync_synchronize()
#define rte_io_rmb() rte_rmb()
uint32_t
rte_read32_relaxed(const volatile void *addr)
{
        return *(const volatile uint32_t *)addr;
}
uint32_t
rte_read32(const volatile void *addr)
{
        uint32_t val;
        val = rte_read32_relaxed(addr);
        rte_io_rmb();
        return val;
}
static inline uint32_t hinic_hwif_read_reg(void *  reg)
{
        return  rte_be_to_cpu_32(rte_read32(reg));
}
//https://github.com/ARM-software/revere-demo/blob/master/vfio.c
void *vfio_map_bar0(int device_fd)
{
	int s;
        void *bar0;
        void *addr0, *addr1;
	//struct vfio_region_info bar0_region = (struct vfio_region_info){
	struct vfio_region_info bar0_region = {
		.argsz = sizeof(struct vfio_region_info),
		.index = VFIO_PCI_BAR0_REGION_INDEX,
	};

	s = ioctl(device_fd, VFIO_DEVICE_GET_REGION_INFO,&bar0_region);
	if (s)
		err(1, "get bar0 region info: %d", s);

	if (!(bar0_region.flags & VFIO_REGION_INFO_FLAG_READ))
		errx(1, "bar0 region not readable");

	if (!(bar0_region.flags & VFIO_REGION_INFO_FLAG_WRITE))
		errx(1, "bar0 region not writable");

	if (!(bar0_region.flags & VFIO_REGION_INFO_FLAG_MMAP))
		errx(1, "bar0 region not mmap'able");

	bar0 = mmap(0,bar0_region.size, PROT_READ | PROT_WRITE, MAP_SHARED, device_fd, bar0_region.offset);
	if (bar0 == MAP_FAILED)
		err(1, "mmap bar0 failed");

	printf("BAR0 @%p, size: %lld, offset: %#llx\n", bar0, bar0_region.size, bar0_region.offset);
        addr0 = bar0 + HINIC_CSR_FUNC_ATTR0_ADDR;
        addr1 = bar0 + HINIC_CSR_FUNC_ATTR1_ADDR;
        printf("attr0 @0x%08x = 0x%08x\n",HINIC_CSR_FUNC_ATTR0_ADDR,  hinic_hwif_read_reg(addr0));
        printf("attr1 @0x%08x = 0x%08x\n",HINIC_CSR_FUNC_ATTR1_ADDR,  hinic_hwif_read_reg(addr1));
	return bar0;
}
int main(int argc, char *argv[]) {
    int group = 0, container = 0, device = 0;
    int ret, i;
    uint64_t pci_bar;
    uint64_t config_offset;
    struct vfio_device_info device_info = { .argsz = sizeof(device_info) };
    struct vfio_region_info reg = { .argsz = sizeof(reg) };
    
    if (argc < 3) {
        printf("Usage: %s /dev/vfio/<group> xxxx:xx:xx:x\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    group = open(argv[1], O_RDWR);
    if (group < 0) {
        perror(argv[1]);
        exit(EXIT_FAILURE);
    }

    container = open("/dev/vfio/vfio", O_RDWR);
    if (container < 0) {
        perror("failed to open /dev/vfio/vfio");
        goto fail_group;
    }

    ioctl(group, VFIO_GROUP_SET_CONTAINER, &container);
    ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU);
    device = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, argv[2]);
    ioctl(device, VFIO_DEVICE_GET_INFO, &device_info);
    printf("device_info {num_irqs:%d, num_regions:%d}\n", device_info.num_irqs, device_info.num_regions);

    reg.index = VFIO_PCI_CONFIG_REGION_INDEX;
    ioctl(device, VFIO_DEVICE_GET_REGION_INFO, &reg);
    printf("config region info {argsz:%d, flags:0x%x, cap_offset:%d size:%ld offset:0x%lx}\n",
        reg.argsz, reg.flags, reg.cap_offset, reg.size, reg.offset);

    config_offset = reg.offset;
    for (i = 0; i < 6; i++) {
        char ioport, mem64;
        //ret = pread64(device, &pci_bar, sizeof(pci_bar), VFIO_GET_REGION_ADDR(VFIO_PCI_CONFIG_REGION_INDEX) + PCI_BASE_ADDRESS_0 + (4 * i));
        ret = pread64(device, &pci_bar, sizeof(pci_bar), config_offset + PCI_BASE_ADDRESS_0 + (4 * i));
        //pci_bar = le64toh(pci_bar);
        ioport = pci_bar & PCI_BASE_ADDRESS_SPACE_IO;
        mem64 = ioport ? 0 : (pci_bar & PCI_BASE_ADDRESS_MEM_TYPE_64);
        printf("%d bar:0x%016x ioport:0x%x mem64:0x%x\n", i, pci_bar, ioport, mem64);
    }
    vfio_map_bar0(device);
    close(container);

fail_group:
    close(group);
    exit(EXIT_SUCCESS);
}
