#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/vfio.h>

void loop_region_cap(struct vfio_region_info *info) {
    struct vfio_info_cap_header *hdr;
    void *ptr = info;

    if (!(info->flags & VFIO_REGION_INFO_FLAG_CAPS)) {
        return;
    }

    for (hdr = ptr + info->cap_offset; hdr != ptr; hdr = ptr + hdr->next) {
        printf("  hdr {id:%d, version:%d next:%d}\n", hdr->id, hdr->version, hdr->next);
        if (VFIO_REGION_INFO_CAP_SPARSE_MMAP == hdr->id) {
            struct vfio_region_info_cap_sparse_mmap *sparse = (struct vfio_region_info_cap_sparse_mmap *)hdr;
            int i = 0;
            for (i = 0; i < sparse->nr_areas; i++) {
                printf("    sparse->areas %d offset:0x%lx size:0x%lx\n",
                    i, sparse->areas[i].offset, sparse->areas[i].size);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int group = 0, container = 0, device = 0;
    int ret, i;
    struct vfio_group_status status = { .argsz = sizeof(status) };
    struct vfio_device_info device_info = { .argsz = sizeof(device_info) };
    
    if (argc < 3) {
        printf("Usage: %s /dev/vfio/<group> xxxx:xx:xx:x\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    printf("pid:%d\n", getpid());

    group = open(argv[1], O_RDWR);
    if (group < 0) {
        perror(argv[1]);
        exit(EXIT_FAILURE);
    }
    printf("group fd:::%d\n", group);

    ret = ioctl(group, VFIO_GROUP_GET_STATUS, &status);
    if (ret) {
        perror("VFIO_GROUP_GET_STATUS");
        goto fail_group;
    }
    printf("VFIO_GROUP_GET_STATUS status {%d:0x%x}\n", status.argsz, status.flags);
    if (!(status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
        printf("group %s is not viable\n", argv[1]);
        goto fail_group;
    }

    container = open("/dev/vfio/vfio", O_RDWR);
    if (container < 0) {
        perror("failed to open /dev/vfio/vfio");
        goto fail_group;
    }
    printf("container fd:::%d\n", container);

    ret = ioctl(container, VFIO_GET_API_VERSION);
    printf("vfio api version:%d\n", ret);
    if (ret != VFIO_API_VERSION) {
        printf("supported vfio version: %d, reported version: %d\n",
            VFIO_API_VERSION, ret);
    }

    ret = ioctl(container, VFIO_CHECK_EXTENSION, VFIO_TYPE1_IOMMU);
    if (ret) {
        printf("vfio type is VFIO_TYPE1_IOMMU\n");
    } else {
        printf("vfio type is not VFIO_TYPE1_IOMMU!!\n");
    }

    ioctl(group, VFIO_GROUP_SET_CONTAINER, &container);
    ioctl(container, VFIO_SET_IOMMU, VFIO_TYPE1_IOMMU);
    device = ioctl(group, VFIO_GROUP_GET_DEVICE_FD, argv[2]);
    printf("device fd:::%d\n", device);
    ioctl(device, VFIO_DEVICE_GET_INFO, &device_info);
    printf("device_info {num_irqs:%d, num_regions:%d}\n", device_info.num_irqs, device_info.num_regions);

    for (i = 0; i < device_info.num_regions; i++) {
        struct vfio_region_info reg = { .argsz = sizeof(reg) };
        reg.index = i;
        ioctl(device, VFIO_DEVICE_GET_REGION_INFO, &reg);
        printf("%d region info {argsz:%d, flags:0x%x, cap_offset:%d size:%ld offset:0x%lx}\n",
            reg.index, reg.argsz, reg.flags, reg.cap_offset, reg.size, reg.offset);
        loop_region_cap(&reg);
        if (reg.argsz > sizeof(struct vfio_region_info)) {
            struct vfio_region_info *rty = malloc(reg.argsz);
            rty->index = i;
            rty->argsz = reg.argsz;
            ioctl(device, VFIO_DEVICE_GET_REGION_INFO, rty);
            printf("%d region info {argsz:%d, flags:0x%x, cap_offset:%d size:%ld offset:0x%lx}\n",
                rty->index, rty->argsz, rty->flags, rty->cap_offset, rty->size, rty->offset);
            loop_region_cap(rty);
            free(rty);
        }
    }

    getchar();
    close(container);

fail_group:
    close(group);
    exit(EXIT_SUCCESS);
}
