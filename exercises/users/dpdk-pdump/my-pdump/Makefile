# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2016 Intel Corporation

include $(RTE_SDK)/mk/rte.vars.mk

ifeq ($(CONFIG_RTE_LIBRTE_PDUMP),y)

APP = dpdk-pdump

CFLAGS += $(WERROR_FLAGS)
CFLAGS += -g

# all source are stored in SRCS-y

SRCS-y := main.c

include $(RTE_SDK)/mk/rte.app.mk

endif
