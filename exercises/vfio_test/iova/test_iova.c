#include <linux/vfio.h>
#include <linux/types.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/eventfd.h>

#include <time.h>

#define VFIO_BASE_PATH "/dev/vfio/"
#define VFIO_CONTAINER_PATH  "/dev/vfio/vfio"

#define VFIO_DMA_MAP_FLAG_NOEXEC (1 << 2)
#define phys_addr_t	uint64_t
static bool phys_addrs_available = true;
typedef uint64_t rte_iova_t;
#define RTE_BAD_IOVA ((rte_iova_t)-1)
#define RTE_LOG(l, t, fmt...) printf(fmt)
#define PFN_MASK_SIZE	8
struct vfio_dev_spec {
	char *name;
	char *bus;
	int iommu_group;

	struct vfio_device_info vfio_device_info;
	int device_fd;

	struct vfio_region_info *regions;
	struct vfio_irq_info *irqs;
};

struct vfio_info {
	struct vfio_group_status group_status;
	struct vfio_iommu_type1_info iommu_info;
	struct vfio_dev_spec dev;

	int container;
	int group;
};

void free_vfio_dev(struct vfio_dev_spec *dev)
{
	if (dev->regions) {
		free(dev->regions);
	}

	if (dev->irqs) {
		free(dev->irqs);
	}
}

void init_vfio_info(struct vfio_info *info)
{
	info->group_status.argsz = sizeof(struct vfio_group_status);
	info->iommu_info.argsz = sizeof(struct vfio_iommu_type1_info);
}

void init_vfio_dev_spec(struct vfio_dev_spec *dev)
{
	dev->regions = NULL;
	dev->irqs = NULL;
}

int is_group_viable(struct vfio_info *info)
{
	int ret;

	ret = ioctl(info->group, VFIO_GROUP_GET_STATUS, &info->group_status);
	if (ret)
		return ret;

	return info->group_status.flags & VFIO_GROUP_FLAGS_VIABLE;
}

int check_vfio_version(struct vfio_info *info)
{
	return ioctl(info->container, VFIO_GET_API_VERSION) != VFIO_API_VERSION;
}

int check_iommu_extension(struct vfio_info *info)
{
	return !ioctl(info->container, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU);
}

int set_group_to_container(struct vfio_info *info)
{
	return ioctl(info->group, VFIO_GROUP_SET_CONTAINER, &info->container);
}

int unset_group_to_container(struct vfio_info *info)
{
	return ioctl(info->group, VFIO_GROUP_UNSET_CONTAINER, &info->container);
}

int set_iommu_type(struct vfio_info *info)
{
	return ioctl(info->container, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU);
}
/*
 *  * Get physical address of any mapped virtual address in the current process.
 *   */
phys_addr_t rte_mem_virt2phy(const void *virtaddr)
{
	int fd, retval;
	uint64_t page, physaddr;
	unsigned long virt_pfn;
	int page_size;
	off_t offset;
 
	/* Cannot parse /proc/self/pagemap, no need to log errors everywhere */
	if (!phys_addrs_available)
		return RTE_BAD_IOVA;
 
	/* standard page size */
	page_size = getpagesize();
 
	fd = open("/proc/self/pagemap", O_RDONLY);
	if (fd < 0) {
		RTE_LOG(ERR, EAL, "%s(): cannot open /proc/self/pagemap: %s\n",
			__func__, strerror(errno));
		return RTE_BAD_IOVA;
	}
 
	virt_pfn = (unsigned long)virtaddr / page_size;
	offset = sizeof(uint64_t) * virt_pfn;
	if (lseek(fd, offset, SEEK_SET) == (off_t) -1) {
		RTE_LOG(ERR, EAL, "%s(): seek error in /proc/self/pagemap: %s\n",
				__func__, strerror(errno));
		close(fd);
		return RTE_BAD_IOVA;
	}
 
	retval = read(fd, &page, PFN_MASK_SIZE);
	close(fd);
	if (retval < 0) {
		RTE_LOG(ERR, EAL, "%s(): cannot read /proc/self/pagemap: %s\n",
				__func__, strerror(errno));
		return RTE_BAD_IOVA;
	} else if (retval != PFN_MASK_SIZE) {
		RTE_LOG(ERR, EAL, "%s(): read %d bytes from /proc/self/pagemap "
				"but expected %d:\n",
				__func__, retval, PFN_MASK_SIZE);
		return RTE_BAD_IOVA;
	}
 
	/*
 * 	 * the pfn (page frame number) are bits 0-54 (see
 * 	 	 * pagemap.txt in linux Documentation)
 * 	 	 	 */
	if ((page & 0x7fffffffffffffULL) == 0)
        {
             printf(" (page & 0x7fffffffffffffULL) == 0  \n ");
		return RTE_BAD_IOVA;
        } 
	physaddr = ((page & 0x7fffffffffffffULL) * page_size)
		+ ((unsigned long)virtaddr % page_size);
 
	RTE_LOG(ERR, EAL, "phyaddr %p\n", (void*)physaddr);
 
	return physaddr;
}
 
rte_iova_t rte_mem_virt2iova(const void *virtaddr)
{
	return rte_mem_virt2phy(virtaddr);
}
int dma_do_map(struct vfio_info *info, struct vfio_iommu_type1_dma_map *map,
						    uint64_t iova, int size)
{
        int ret ;
        rte_iova_t phy;
        int fd = open("mmap.txt",  O_CREAT | O_RDWR, 0600);
        if (fd < 0)
                return -1;
        if (ftruncate(fd, size) < 0) {
                close(fd);
                return -1;
        }
	map->argsz = sizeof(*map);
	map->vaddr = (uintptr_t)mmap(0, size, PROT_READ | PROT_WRITE,
			     MAP_PRIVATE | MAP_ANONYMOUS, fd, 0);
        printf(" mmap addr: %p \n", map->vaddr);
        if (MAP_FAILED == (uint32_t)(map->vaddr))
                return -1;
        close(fd);
        // alloc page
        *((char *)map->vaddr) = 1;
	if( RTE_BAD_IOVA == (phy=rte_mem_virt2phy((void *)map->vaddr)))
        {
            printf("rte_mem_virt2phy err \n");
            return -1;
        }
        map->iova = 0;
        //map->iova = map->vaddr;
        //map->iova = phy;
	map->flags = VFIO_DMA_MAP_FLAG_READ | VFIO_DMA_MAP_FLAG_WRITE;
	map->size = size;

	//memset(map->vaddr, 0x55, size);

        printf(" before dma map, vaddr: %p, iova %p \n", map->vaddr, map->iova);
	ret = ioctl(info->container, VFIO_IOMMU_MAP_DMA, map);
                        if (ret) {
                                printf("Failed to map memory \n");
                                return ret;
                        }
                        else {
                            printf(" after dma map, vaddr: %p, iova %p \n", map->vaddr, map->iova);
                        }
       return 0;
}

int dma_do_unmap(struct vfio_info *info, struct vfio_iommu_type1_dma_map *map)
{
	int ret;
	struct vfio_iommu_type1_dma_unmap unmap;

	unmap.argsz = sizeof(struct vfio_iommu_type1_dma_unmap);
	unmap.iova = map->iova;
	unmap.size = map->size;
	unmap.flags = map->flags;

	ret = ioctl(info->container, VFIO_IOMMU_UNMAP_DMA, &unmap);

	munmap((void *)map->vaddr, map->size);

	return ret;
}

void get_vfio_device_info(int dev_fd, struct vfio_device_info *info)
{
	info->argsz = sizeof(*info);
	ioctl(dev_fd, VFIO_DEVICE_GET_INFO, info);
}

void populate_device_regions(struct vfio_dev_spec *dev)
{
	int i;
	int num_regs = dev->vfio_device_info.num_regions;

	dev->regions = malloc(num_regs * sizeof(struct vfio_region_info));

	for (i = 0; i < num_regs; i++) {
		struct vfio_region_info *reg = &dev->regions[i];

		reg->argsz = sizeof(*reg);
		reg->index = i;

		ioctl(dev->device_fd, VFIO_DEVICE_GET_REGION_INFO, reg);
	}
}

void populate_device_irqs(struct vfio_dev_spec *dev)
{
	int i;
	int num_irqs = dev->vfio_device_info.num_irqs;

	dev->irqs = malloc(num_irqs * sizeof(struct vfio_irq_info));

	for (i = 0; i < num_irqs; i++) {
		struct vfio_irq_info *irq = &dev->irqs[i];

		irq->argsz = sizeof(*irq);
		irq->index = i;

		ioctl(dev->device_fd, VFIO_DEVICE_GET_IRQ_INFO, irq);
                
                /* if this vector cannot be used with eventfd continue with next*/
                if ((irq->flags & VFIO_IRQ_INFO_EVENTFD) == 0) {
                        printf("IRQ doesn't support Event FD");
                        continue;
                }
                
	}
}


int vfio_irqfd_clean(int device, unsigned int index)
{
        struct vfio_irq_set irq_set = {
                .argsz = sizeof(irq_set),
                .flags = VFIO_IRQ_SET_DATA_NONE | VFIO_IRQ_SET_ACTION_TRIGGER,
                .index = index,
                .start = 0,
                .count = 0,
        };

        int ret = ioctl(device, VFIO_DEVICE_SET_IRQS, &irq_set);

        if (ret) {
                return 1;
        }

        return 0;
}
#define IRQ_SET_BUF_LEN (sizeof(struct vfio_irq_set) + sizeof(int))
#define MAX_INTERRUPT_VECTORS 32
#define MSIX_IRQ_SET_BUF_LEN (sizeof(struct vfio_irq_set) + sizeof(int) * (MAX_INTERRUPT_VECTORS + 1))
/**
 * Enable VFIO MSI-X interrupts.
 * @param device_fd The VFIO file descriptor.
 * @return The event file descriptor.
 */
int vfio_enable_msix(int device_fd, uint32_t interrupt_vector, int fd) {
	printf("Enable MSIX Interrupts \n");
	char irq_set_buf[MSIX_IRQ_SET_BUF_LEN];
	struct vfio_irq_set* irq_set;
	int* fd_ptr;
        int ret;

	irq_set = (struct vfio_irq_set*) irq_set_buf;
	irq_set->argsz = sizeof(irq_set_buf);
	if (!interrupt_vector) {
		interrupt_vector = 1;
	} else if (interrupt_vector > MAX_INTERRUPT_VECTORS)
		interrupt_vector = MAX_INTERRUPT_VECTORS + 1;

	irq_set->count = interrupt_vector;
	irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
	irq_set->index = VFIO_PCI_MSIX_IRQ_INDEX;
	irq_set->start = 0;
	fd_ptr = (int*) &irq_set->data;
	fd_ptr[0] = fd;

	ret = ioctl(device_fd, VFIO_DEVICE_SET_IRQS, irq_set);
        return ret;
}
int vfio_irqfd_init(int device, unsigned int index,unsigned int count, int fd)
{

        struct vfio_irq_set *irq_set;
        int32_t *pfd;
        int ret, argsz;
       
        if (VFIO_PCI_MSI_IRQ_INDEX != index)
        {
            printf(" %s not msi interrupt \n", __func__);
            return 0;
        }

        argsz = sizeof(*irq_set) + sizeof(*pfd);
        irq_set = malloc(argsz);

        if (!irq_set) {
                printf("Failure in %s allocating memory\n", __func__);

                return 1;
        }

        irq_set->argsz = argsz;
        irq_set->flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER;
        //irq_set->index = VFIO_PCI_MSIX_IRQ_INDEX;
        irq_set->index = index;
        irq_set->start = 0;
        irq_set->count = count;
        pfd = (int32_t *)&irq_set->data;
        *pfd = fd;

        ret = ioctl(device, VFIO_DEVICE_SET_IRQS, irq_set);
        free(irq_set);

        if (ret) {
                return 1;
        }

        return 0;
}

/**
 * Initialize device IRQ with @irq_type and and register an event notifier.
 */
int qemu_vfio_pci_init_irq(int device, int irq_type)
{
    int ret;
    struct vfio_irq_set *irq_set;
    size_t irq_set_size;
    struct vfio_irq_info irq_info = { .argsz = sizeof(irq_info) };
    unsigned long long int e;
    int irqfd;
    irq_info.index = irq_type;
    int index;
    if (ioctl(device, VFIO_DEVICE_GET_IRQ_INFO, &irq_info)) {
        return -EINVAL;
    }
    if (!(irq_info.flags & VFIO_IRQ_INFO_EVENTFD)) {
        printf("Device interrupt doesn't support eventfd");
        return -EINVAL;
    }
    irq_set_size = sizeof(*irq_set) + sizeof(int);
    irq_set = malloc(irq_set_size);
    /* Get to a known IRQ state */
    *irq_set = (struct vfio_irq_set) {
        .argsz = irq_set_size,
        .flags = VFIO_IRQ_SET_DATA_EVENTFD | VFIO_IRQ_SET_ACTION_TRIGGER,
        .index = irq_info.index,
        .start = 0,
        .count = 1,
    };
    printf("irq index %d , irq type : %d \n", irq_info.index, irq_type);
    irqfd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
    *(int *)&irq_set->data = irqfd;
    ret = ioctl(device, VFIO_DEVICE_SET_IRQS, irq_set);
    free(irq_set);
    if (ret) {
        printf("Failed to setup device interrupt \n ");
        close(irqfd);
        return -errno;
    }
    else
    {
        printf("succ to setup device interrupt \n");
    }
    ret = read(irqfd, &e, sizeof(e));
     if (ret != -1 || errno != EAGAIN) {
             printf("IRQ %d shouldn't trigger yet.\n", irq_info.index);

             goto error1;
     }
     else
     {
             printf("IRQ %d  trigger .\n", irq_info.index);
     }

     if (vfio_irqfd_clean(device, irq_info.index)) {
             printf("error while cleaning IRQ num.%d\n",
                                             irq_info.index);
             goto error1;

     }
 error1:
                        close(irqfd);
    return 0;
}
int main(int argc, const char **argv)
{
	int i ,ret;
	struct vfio_info dev_vfio_info;
	struct vfio_iommu_type1_dma_map dma_map;
	struct vfio_dev_spec dev;

	init_vfio_info(&dev_vfio_info);
	init_vfio_dev_spec(&dev);

	/* Create a new container */
	dev_vfio_info.container = open(VFIO_CONTAINER_PATH, O_RDWR);

	if (check_vfio_version(&dev_vfio_info)
		|| check_iommu_extension(&dev_vfio_info)) {
		printf("IOMMU Type1 not supported or unknown API\n");

		goto error;
	}

	/* Open the group */
	dev_vfio_info.group = open("/dev/vfio/26", O_RDWR);
	if (!is_group_viable(&dev_vfio_info)) {
		printf("the group is not viable\n");

		goto error;
	}

	if (set_group_to_container(&dev_vfio_info)
			|| set_iommu_type(&dev_vfio_info)) {
		printf("something went wrong\n");

		goto error;
	}

	/* try to map 1MB of memory to the device */
	printf("dma-mapping some memory to the device\n");
	if (dma_do_map(&dev_vfio_info, &dma_map, 0, getpagesize()*4)) {
		printf("error while dma-mapping\n");
	} else {
		printf("dma-map successful\n");
	}

	/* use 0000:05:00.0  for example */
	dev.device_fd = ioctl(dev_vfio_info.group, VFIO_GROUP_GET_DEVICE_FD, "0000:05:00.0");
	if (dev.device_fd < 0) {
		printf("unable to get device fd\n");
		goto error;
	}

	get_vfio_device_info(dev.device_fd, &dev.vfio_device_info);
	populate_device_regions(&dev);

	printf("\nNum regions: %d\n", dev.vfio_device_info.num_regions);
error1:
	dma_do_unmap(&dev_vfio_info, &dma_map);

	free_vfio_dev(&dev);

	unset_group_to_container(&dev_vfio_info);
	close(dev_vfio_info.group);
	close(dev_vfio_info.container);

	return 0;

error:
	exit(1);
}
