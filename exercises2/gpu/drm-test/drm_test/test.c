#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#if 1
#include <xf86drm.h>
#include <xf86drmMode.h>
#endif
int main(int argc, char **argv)
{
        int fd;
        char *vaddr;
        struct drm_mode_create_dumb create_req = {};
        struct drm_mode_destroy_dumb destroy_req = {};
        struct drm_mode_map_dumb map_req = {};

        fd = open("/dev/dri/card0", O_RDWR);

        create_req.bpp = 32;
        create_req.width = 240;
        create_req.height = 320;


        //  在 drm_vma_offset_manager 中分配一个 240 x 320 的显存, 并返回对应 gem obj 的 handle.
        drmIoctl(fd, DRM_IOCTL_MODE_CREATE_DUMB, &create_req);
        printf("create dumb: handle = %u, pitch = %u, size = %llu\n",
                create_req.handle, create_req.pitch, create_req.size);

        // 通过 handle 找到 gem, 返回对应 node 的 start 偏移 == map_req.offset
        map_req.handle = create_req.handle;
        drmIoctl(fd, DRM_IOCTL_MODE_MAP_DUMB, &map_req);
        printf("get mmap offset 0x%llx\n", map_req.offset);

        // 使用这个 offeet 找到对应 node 并映射其描述的内存
        vaddr = mmap(0, create_req.size, PROT_WRITE, MAP_SHARED, fd, map_req.offset);
        strcpy(vaddr, "This is a dumb buffer!");
        munmap(vaddr, create_req.size);

        // 使用这个 offeet 找到对应 node 并映射其描述的内存
        vaddr = mmap(0, create_req.size, PROT_READ, MAP_SHARED, fd, map_req.offset);
        printf("read from mmap: %s\n", vaddr);
        munmap(vaddr, create_req.size);

        getchar();

        destroy_req.handle = create_req.handle;
        drmIoctl(fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_req);
        close(fd);

        return 0;
}
 
