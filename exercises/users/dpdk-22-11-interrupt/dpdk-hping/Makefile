# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2010-2014 Intel Corporation

# Build using pkg-config variables if possible
ifneq ($(shell pkg-config --exists libdpdk; echo $$?), 0)
$(error "Cannot find libdpdk using pkg-config")
endif

PKGCONF = pkg-config
SOURCES += $(wildcard *.c)
TARGETS := $(patsubst %.c, %, $(SOURCES))

all: $(TARGETS)

CFLAGS += -g -O3 $(shell $(PKGCONF) --cflags libdpdk)
LDFLAGS += $(shell $(PKGCONF) --libs libdpdk) -lpthread

%: %.c Makefile
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
