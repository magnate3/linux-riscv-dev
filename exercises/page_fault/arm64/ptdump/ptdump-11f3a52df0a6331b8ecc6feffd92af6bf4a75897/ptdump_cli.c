/*
 * Copyright (C) 2017 Hewlett Packard Enterprise Development, L.P.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published by
 * the Free Software Foundation.
 */

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "ptdump.h"

#define PAGE_SIZE 4096

int do_dump(int fd, unsigned int order)
{
	struct ptdump_req req;
        unsigned long buf;
	size_t buf_size = PAGE_SIZE * (1 << order);
	int err = 0;

	/* Allocate a page aligned buffer */
	if (posix_memalign((void **)&buf, PAGE_SIZE, buf_size)) {
                fprintf(stderr, "*** failed to allocate memory (%s)\n",
                        strerror(errno));
		return 1;
        }
	printf("buffer addr: %016lx, size: %ld\n", buf, buf_size);

	/* Beginning of first page */
	req.addr = buf;
	req.order = 0;
	strcpy((char *)req.addr, "!!! Testing 1 !!!");
	if (ioctl(fd, PTDUMP_DUMP, &req) < 0) {
		fprintf(stderr, "*** failed to execute ioctl command\n");
		err |= 1;
	}

	/* Inside first page (offset 0x100) */
	req.addr = buf + 0x100;
	req.order = 0;
	strcpy((char *)req.addr, "!!! Testing 2 !!!");
        if (ioctl(fd, PTDUMP_DUMP, &req) < 0) {
		fprintf(stderr, "*** failed to execute ioctl command\n");
		err |= 1;
	}

	if (order > 0) {
		/* Second page */
		req.addr = buf + PAGE_SIZE;
		req.order = 0;
		strcpy((char *)req.addr, "!!! Testing 3 !!!");
		if (ioctl(fd, PTDUMP_DUMP, &req) < 0) {
			fprintf(stderr, "*** failed to execute ioctl "
				"command\n");
			err |= 1;
		}
	}

	free((void *)buf);
	return err;
}

int do_write(int fd, unsigned int order)
{
	struct ptdump_req req;
        unsigned long buf;
	size_t buf_size = PAGE_SIZE * (1 << order);
	int err = 0;

	/* Allocate a page aligned buffer */
	if (posix_memalign((void **)&buf, PAGE_SIZE, buf_size)) {
                fprintf(stderr, "*** failed to allocate memory (%s)\n",
                        strerror(errno));
		return 1;
        }
	printf("buffer addr: %016lx, size: %ld\n", buf, buf_size);

	req.addr = buf;
	req.order = order;
	strcpy((void *)req.addr, "!!! Testing 1 !!!");
	if (ioctl(fd, PTDUMP_WRITE, &req) < 0) {
		fprintf(stderr, "*** failed to execute ioctl command\n");
		err |= 1;
	}

	free((void *)buf);
	return err;
}

int main(int argc, char *argv[])
{
        int fd;
	int err = 0;

        printf("\n");
        printf("pid of current process: %d\n", getpid());
	
        fd = open("/dev/ptdump", O_RDWR);
        if (fd < 0) {
                fprintf(stderr, "*** failed to open /dev/ptdump\n");
                return 1;
        }

	err |= do_dump(fd, 1);
	err |= do_write(fd, 1);

        printf("\n");

	close(fd);
        return err;
}
