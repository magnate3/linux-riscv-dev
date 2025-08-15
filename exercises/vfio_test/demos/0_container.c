#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <linux/vfio.h>

int main(int argc, char *argv[]) {
    int container = 0, ret = 0;

    container = open("/dev/vfio/vfio", O_RDWR);
    if (container < 0) {
        perror("failed to open /dev/vfio/vfio");
        return -1;
    }

    ret = ioctl(container, VFIO_GET_API_VERSION);
    printf("vfio api version:%d\n", ret);
    if (ret != VFIO_API_VERSION) {
        printf("supported vfio version: %d, reported version: %d\n",
            VFIO_API_VERSION, ret);
    }

    close(container);
    return 0;
}
