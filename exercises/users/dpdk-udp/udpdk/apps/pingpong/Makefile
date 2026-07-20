#
# Created by leoll2 on 9/25/20.
# Copyright (c) 2020 Leonardo Lai. All rights reserved.
#

RTE_TARGET ?= arm64-armv8a-linuxapp-gcc
UDPDK_DPDK_INCLUDE=/root/dpdk-19.11/${RTE_TARGET}/include
ROOTDIR=../..
DEPSDIR=${ROOTDIR}/deps
LISTSDIR=${ROOTDIR}/udpdk/list

ifeq ($(RTE_TARGET),)
$(error "Please define RTE_TARGET environment variable")
endif

ifeq ($(UDPDK_PATH),)
	UDPDK_PATH=${ROOTDIR}
endif

# all source are stored in SRCS-y
SRCS= main.c

LIBS+= -L${UDPDK_PATH}/udpdk -Wl,--whole-archive,-ludpdk,--no-whole-archive
LIBS+= -L/root/dpdk-19.11/${RTE_TARGET}/lib -Wl,--whole-archive,-ldpdk,--no-whole-archive
#LIBS+= -L${DEPSDIR}/dpdk/${RTE_TARGET}/lib -Wl,--whole-archive,-ldpdk,--no-whole-archive
LIBS+= -Wl,--no-whole-archive -lrt -lm -ldl -lcrypto -pthread -lnuma  -lpcap

CFLAGS += $(WERROR_FLAGS)  -O0 -g

TARGET="pingpong"
all:
	cc -I${ROOTDIR}/udpdk -I$(UDPDK_DPDK_INCLUDE) -I${LISTSDIR} $(DEPSDIR)/inih/ini.c -o ${TARGET} ${SRCS} ${LIBS}
	#cc -I${ROOTDIR}/udpdk -I${DEPSDIR}/dpdk/${RTE_TARGET}/include -o ${TARGET} ${SRCS} ${LIBS}

.PHONY: clean
clean:
	rm -f *.o ${TARGET}
