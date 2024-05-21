#include <linux/module.h>
#include <drm/drm_crtc_helper.h>
#include <drm/drm_plane_helper.h>
#include <drm/drm_fb_cma_helper.h>
#include <drm/drm_gem_cma_helper.h>
#include <drm/drmP.h>
#include <drm/drm_mm.h>
#include <drm/drm_vma_manager.h>

#define WIDTH  1920
#define HEIGHT 1080

static struct drm_device *drm;

static const struct file_operations vkms_fops = {
    .owner = THIS_MODULE,
    .open = drm_open,
    .release = drm_release,
    .unlocked_ioctl = drm_ioctl,
    .poll = drm_poll,
    .read = drm_read,
    .mmap = drm_gem_cma_mmap, // 实现 mmap 操作
};

static struct drm_driver vkms_driver = {
    .fops           = &vkms_fops,
    .driver_features    =  DRIVER_GEM,
    .name           = "vkms",
    .desc           = "Virtual Kernel Mode Setting",
    .date           = "20180514",
    .major          = 1,
    .minor          = 0,
    // 在 drm_gem_cma_mmap 中会设置这个回调函数. 在我们驱动中也要实现, 
    // 不然会 mmap 失败. 博主也不知道为啥, 有知道的同学可以告知一下.
    .gem_vm_ops     = &drm_gem_cma_vm_ops,     
    .dumb_create    = drm_gem_cma_dumb_create, // 在 drm_mm 中申请一个 node , 并分配物理内存
    .dumb_map_offset = drm_gem_dumb_map_offset, // 返回 node 中的 start 内存偏移, 即该 node 的索引
};

static int vkms_drm_mm_init(struct drm_device *dev)
{
    struct drm_vma_offset_manager *mgr;

    mgr = kzalloc(sizeof(*mgr), GFP_KERNEL);

    drm->vma_offset_manager = mgr;
    drm_mm_init(&mgr->vm_addr_space_mm, 0, WIDTH * HEIGHT * 2);

    return 0;
}

static void vkms_drm_mm_cleanup(struct drm_device *dev)
{
    kfree(dev->vma_offset_manager);
}

static int __init vkms_init(void)
{

    drm = drm_dev_alloc(&vkms_driver, NULL);

    vkms_drm_mm_init(drm); // 初始化 drm_mm

    drm_dev_register(drm, 0);

    return 0;
}

static void __exit vkms_exit(void)
{
    drm_dev_unregister(drm);
    vkms_drm_mm_cleanup(drm);
    drm_dev_unref(drm);
}

module_init(vkms_init);
module_exit(vkms_exit);

MODULE_AUTHOR("baron");
MODULE_DESCRIPTION("drm mm test drv");
MODULE_LICENSE("GPL");
 