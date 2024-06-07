#ifndef CHARDEV_H
#define CHARDEV_H

#include <linux/ioctl.h>

#define MAJOR_MAGIC 100
#define DEVNAME "simpler"

struct address{
    unsigned long user_addr;
    unsigned long len;
};
#define IOCTL_ALLOC_VMA _IOR(MAJOR_MAGIC, 0, char*)
#define IOCTL_FREE_VMA _IOWR(MAJOR_MAGIC, 1, char*)


#endif
