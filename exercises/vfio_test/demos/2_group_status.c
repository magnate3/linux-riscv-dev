#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/vfio.h>

int main(int argc, char *argv[]) {
    int group = 0, ret = 0;
    struct vfio_group_status status = { .argsz = sizeof(status) };
    
    if (argc < 2) {
        printf("Usage: %s /dev/vfio/<group>\n", argv[0]);
        exit(EXIT_FAILURE);
    }
    group = open(argv[1], O_RDWR);
    if (group < 0) {
        perror(argv[1]);
        exit(EXIT_FAILURE);
    }

    ret = ioctl(group, VFIO_GROUP_GET_STATUS, &status);
    if (ret) {
        perror("VFIO_GROUP_GET_STATUS");
        goto fail_fd;
    }
    printf("VFIO_GROUP_GET_STATUS status {%d:0x%x}\n", status.argsz, status.flags);
    if (!(status.flags & VFIO_GROUP_FLAGS_VIABLE)) {
        printf("group %s is not viable\n", argv[1]);
        close(group);
    }

fail_fd:
    close(group);
    exit(EXIT_SUCCESS);
}
