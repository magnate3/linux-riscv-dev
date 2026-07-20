# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2010-2014 Intel Corporation

# binary name
APP = dpdk-ipv6

# all source are stored in SRCS-y
SRCS-y := main.c util.c  dpdk_ip.c dpdk_icmp6.c dpdk_eth.c dpdk_mbuf.c dpdk_ipv6_ext.c dpdk_ipv6.c  dpdk_ipv6_ndic.c  dpdk_nat46.c

# Build using pkg-config variables if possible
$(shell pkg-config --exists libdpdk)
ifeq ($(.SHELLSTATUS),0)

all: shared
.PHONY: shared static
shared: build/$(APP)-shared
	ln -sf $(APP)-shared build/$(APP)
static: build/$(APP)-static
	ln -sf $(APP)-static build/$(APP)

PC_FILE := $(shell pkg-config --path libdpdk)
CFLAGS += -O3 $(shell pkg-config --cflags libdpdk)
LDFLAGS_SHARED = $(shell pkg-config --libs libdpdk)
LDFLAGS_STATIC = -Wl,-Bstatic $(shell pkg-config --static --libs libdpdk)

build/$(APP)-shared: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED)

build/$(APP)-static: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_STATIC)

build:
	@mkdir -p $@

.PHONY: clean
clean:
	rm -f build/$(APP) build/$(APP)-static build/$(APP)-shared
	rmdir --ignore-fail-on-non-empty build

else

ifeq ($(RTE_SDK),)
$(error "Please define RTE_SDK environment variable")
endif

# Default target, can be overridden by command line or environment
#RTE_TARGET ?= x86_64-native-linuxapp-gcc
RTE_TARGET ?= arm64-armv8a-linuxapp-gcc


include $(RTE_SDK)/mk/rte.vars.mk

CFLAGS += -DDPDK_IPV6_DEBUG -DINET6 -DCONFIG_DPDK_ICMP_DEBUG
CFLAGS += -O3
CFLAGS += $(WERROR_FLAGS)  -Wno-pointer-arith -Wno-unused-parameter -Wno-unused-function
CFLAGS += -I$(SRCDIR)/include
include $(RTE_SDK)/mk/rte.extapp.mk

endif
