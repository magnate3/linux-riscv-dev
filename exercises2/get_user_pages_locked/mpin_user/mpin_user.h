/* SPDX-License-Identifier: GPL-2.0+ WITH Linux-syscall-note */
#ifndef _MPIN_USER_H
#define _MPIN_USER_H

#define MPIN_USER_N "mpin_user"
#define MPIN_USER_PATH "/proc/" MPIN_USER_N
/* User does: open(MPIN_USER_PATH, ...) */

/**
 * struct mpin_user_address - Expected pin user space address and size
 * @addr: Address to pin
 * @size: Size of pin address
 */
struct mpin_user_address {
	__u64 addr;		/* virtual address of calling process */
	__u64 size;		/* size in bytes to map */
};

/* MPIN_CMD_PIN: Pin a range of memory */
#define MPIN_CMD_PIN		_IOW('W', 2, struct mpin_user_address)

/* MPIN_CMD_UNPIN: Unpin a range of memory */
#define MPIN_CMD_UNPIN		_IOW('W', 3, struct mpin_user_address)

#ifndef __KERNEL__

#include <sys/ioctl.h>

static inline int mpin_user_add(int fd, void *virt, size_t size)
{
	struct mpin_user_address mua = {
		.addr = (__u64)virt,
		.size = size,
	};

	return ioctl(fd, MPIN_CMD_PIN, &mua);
}

static inline int mpin_user_remove(int fd, void *virt, size_t size)
{
	struct mpin_user_address mua = {
		.addr = (__u64)virt,
		.size = size,
	};

	return ioctl(fd, MPIN_CMD_UNPIN, &mua);
}

#endif /* __KERNEL__ */
#endif /* _MPIN_USER_H */
