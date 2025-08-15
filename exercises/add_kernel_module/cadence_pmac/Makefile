# SPDX-License-Identifier: GPL-2.0
#
# Makefile for the Atmel network device drivers.
#
pmac-y	:= p_macb_main.o
ifeq ($(CONFIG_MACB),n && $(CONFIG_MACB_USE_HWSTAMP),y)
pmac-y	+= macb_ptp.o
endif

obj-$(CONFIG_P_MACB) += pmac.o
obj-$(CONFIG_MACB_PCI) += macb_pci.o
