# SPDX-License-Identifier: GPL-2.0

#CROSS_COMPILE := arm-linux-gnueabi-

CC := $(CROSS_COMPILE)gcc
#CFLAGS := -I../../usr/include

PROGS := dmabuf_sync share_fd mmap_dmabuf

all: $(PROGS)
dmabuf_sync:
	gcc  dmabuf_sync.c -o dmabuf_sync -I../
share_fd:
	gcc  share_fd.c -o share_fd -I../
mmap_dmabuf:
	gcc  mmap_dmabuf.c -o  mmap_dmabuf -I../
clean:
	rm -fr $(PROGS)
