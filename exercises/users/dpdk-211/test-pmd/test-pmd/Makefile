# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2010-2014 Intel Corporation

# binary name
APP = testpmd

# all source are stored in SRCS-y
SRCS-y := main.c

PKGCONF ?= pkg-config

# Build using pkg-config variables if possible
ifneq ($(shell $(PKGCONF) --exists libdpdk && echo 0),0)
$(error "no installation of DPDK found")
endif

# all source are stored in SRCS-y
SRCS-y := testpmd.c
SRCS-y += parameters.c
SRCS-y += cmdline.c
SRCS-y += cmdline_flow.c
SRCS-y += cmdline_mtr.c
SRCS-y += cmdline_tm.c
SRCS-y += cmd_flex_item.c
SRCS-y += config.c
SRCS-y += iofwd.c
SRCS-y += macfwd.c
SRCS-y += 5tswap.c
SRCS-y += macswap.c
SRCS-y += flowgen.c
SRCS-y += rxonly.c
SRCS-y += txonly.c
SRCS-y += csumonly.c
SRCS-y += shared_rxq_fwd.c
SRCS-y += icmpecho.c
SRCS-y += noisy_vnf.c
SRCS-y += bpf_cmd.c
SRCS-$(CONFIG_RTE_LIBRTE_IEEE1588) += ieee1588fwd.c
SRCS-$(CONFIG_RTE_LIBRTE_BPF) += bpf_cmd.c
SRCS-y += util.c
ifeq ($(CONFIG_RTE_LIBRTE_PMD_SOFTNIC), y)
SRCS-y += softnicfwd.c
endif
all: shared
.PHONY: shared static
shared: build/$(APP)-shared
	ln -sf $(APP)-shared build/$(APP)
static: build/$(APP)-static
	ln -sf $(APP)-static build/$(APP)

PC_FILE := $(shell $(PKGCONF) --path libdpdk 2>/dev/null)
CFLAGS += -O3 $(shell $(PKGCONF) --cflags libdpdk)
LDFLAGS_SHARED = $(shell $(PKGCONF) --libs libdpdk)
LDFLAGS_STATIC = $(shell $(PKGCONF) --static --libs libdpdk)

ifeq ($(MAKECMDGOALS),static)
# check for broken pkg-config
ifeq ($(shell echo $(LDFLAGS_STATIC) | grep 'whole-archive.*l:lib.*no-whole-archive'),)
$(warning "pkg-config output list does not contain drivers between 'whole-archive'/'no-whole-archive' flags.")
$(error "Cannot generate statically-linked binaries with this version of pkg-config")
endif
endif

CFLAGS += -DALLOW_EXPERIMENTAL_API

LDFLAGS_SHARED += -lrte_net_i40e 
build/$(APP)-shared: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_SHARED)

build/$(APP)-static: $(SRCS-y) Makefile $(PC_FILE) | build
	$(CC) $(CFLAGS) $(SRCS-y) -o $@ $(LDFLAGS) $(LDFLAGS_STATIC)

build:
	@mkdir -p $@


clean:
	rm -f build/$(APP) build/$(APP)-static build/$(APP)-shared
	test -d build && rmdir -p build || true
