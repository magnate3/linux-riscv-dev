#ifndef CHARDEV_H
#define CHARDEV_H

#include <linux/ioctl.h>

#define MAJOR_MAGIC 100
#define DEVNAME "simpler"


#define IOCTL_MVE_VMA_TO _IOR(MAJOR_MAGIC, 0, char*)
#define IOCTL_MVE_VMA _IOWR(MAJOR_MAGIC, 1, char*)
#define IOCTL_PROMOTE_VMA _IOR(MAJOR_MAGIC, 2, char*)
#define IOCTL_COMPACT_VMA _IOWR(MAJOR_MAGIC, 3, char*)


#endif
