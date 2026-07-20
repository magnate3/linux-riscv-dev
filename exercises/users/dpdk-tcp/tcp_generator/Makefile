# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2010-2020 Intel Corporation

# binary name
APP = tcp-generator

# all source are stored in SRCS-y
SRCS-y := main.c util.c tcp_util.c dpdk_util.c

# Build using pkg-config variables if possible
#ifneq ($(shell pkg-config --exists libdpdk && echo 0),0)
#$(error "no installation of DPDK found")
#endif

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif
RTE_TARGET ?=  arm64-armv8a-linuxapp-gcc
include $(RTE_SDK)/mk/rte.vars.mk
DPDK_LIB := $(RTE_SDK)/$(RTE_TARGET)
LDFLAGS +=  -L ${DPDK_LIB}/lib -ldpdk -lpthread -lnuma -lrt -lm -ldl -lpcap
all: shared
.PHONY: shared static
shared: build/$(APP)-shared
	ln -sf $(APP)-shared build/$(APP)
static: build/$(APP)-static
	ln -sf $(APP)-static build/$(APP)

PKGCONF ?= pkg-config

#PC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)
#CFLAGS += -O3 $(shell $(PKGCONF) --cflags libdpdk)
#LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk)
#LDFLAGS_STATIC = $(shell $(PKGCONF) --static --libs libdpdk)

#ifeq ($(MAKECMDGOALS),static)
## check for broken pkg-config
#ifeq ($(shell echo $(LDFLAGS_STATIC) | grep 'whole-archive.*l:lib.*no-whole-archive'),)
#$(warning "pkg-config output list does not contain drivers between 'whole-archive'/'no-whole-archive' flags.")
#$(error "Cannot generate statically-linked binaries with this version of pkg-config")
#endif
#endif

CFLAGS += -DALLOW_EXPERIMENTAL_API -Wall -DNIC_SUPPPORT_FLOW_OFFLOAD=0 -DDBUG_TCP=0 -DDBUG_TCP_GSO=1
include $(RTE_SDK)/mk/rte.extapp.mk

build/$(APP)-shared: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(filter %.c,$^) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED) -lm

build/$(APP)-static: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(filter %.c,$^) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED) -lm

build:
	@mkdir -p $@

.PHONY: clean
clean:
	rm -f build/$(APP) build/$(APP)-static build/$(APP)-shared
	test -d build && rmdir -p build || true
