// SPDX-License-Identifier: GPL-2.0
/*
 * mpin_user-test.c - Test and example usage of the mpin_user.ko driver
 *
 * Copyright (C) 2023 Weka.io ltd.
 *
 * Author: Boaz Harrosh <boaz@weka.io>
 */
#include <asm-generic/int-ll64.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdarg.h>

#include "mpin_user.h"

bool g_mpin_on = false;
int g_mpin_fd = -1;

static void pr_error(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	vfprintf(stderr, fmt, args);
	va_end(args);
}

static void pr_info(const char *fmt, ...)
{
	va_list args;

	va_start(args, fmt);
	vfprintf(stdout, fmt, args);
	va_end(args);
}

static bool mpin_is_active(int fd)
{
	char msg[8];
	int l;

	l = read(fd, msg, sizeof(msg));
	if (l <= 0) {
		pr_error("Failed to read %s => %d\n", MPIN_USER_PATH, errno);
		return false;
	}

	if (0 == strncmp(msg, "1", 1))
		return true;
	else
		return false;
}

static int mpin_init(void)
{
	g_mpin_fd = open(MPIN_USER_PATH, O_RDWR);
	if (g_mpin_fd < 0) {
		pr_error("Failed to open %s => %d\n", MPIN_USER_PATH, errno);
		g_mpin_fd = -2;
		return errno;
	}

	g_mpin_on = mpin_is_active(g_mpin_fd);
	if (!g_mpin_on)
		pr_info("mpin_user driver loaded but pinning NOT enabeled\n");

	return 0;
}

void *_some_alloc(size_t size)
{
	void *addr;

	addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (addr == MAP_FAILED) {
		pr_error("Failed to alloc 0x%zx bytes => %d\n", size, errno);
		return NULL;
	}

	return addr;
}

/* If driver is not loaded it is an error,
 * if pinning not needed, its just a NO-OP
 */
int _mpin_add(void *addr, __u64 size)
{
	int err;

	if (g_mpin_fd < 0)
		return ENODEV;

	if (!g_mpin_on)
		return 0;

	err = mpin_user_add(g_mpin_fd, addr, size);
	if (err < 0) {
		pr_error("mpin_user_add => %d\n", errno);
		return errno;
	}

	return 0;
}

#define M (1024UL * 1024UL)
#define  ALLOC_SIZE (16UL * M)

int main(int argc, char *argv[])
{
	void *buff;
	int err;

	err = mpin_init();
	if (err)
		return err;

	buff = _some_alloc(ALLOC_SIZE);
	if (!buff)
		return ENOMEM;

	err = _mpin_add(buff, ALLOC_SIZE);
	if (err)
		return err;

	pr_info("mpin_user test all done\n");
	return 0;
}
