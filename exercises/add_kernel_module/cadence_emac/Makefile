# SPDX-License-Identifier: GPL-2.0
#
# Makefile for the Atmel network device drivers.
#
emac-y	:= e_macb_main.o
#ifeq ($(CONFIG_MACB_USE_HWSTAMP),y)
ifeq ($(CONFIG_MACB),n && $(CONFIG_P_MACB),n && $(CONFIG_MACB_USE_HWSTAMP),y)
emac-y	+= macb_ptp.o
endif

obj-$(CONFIG_E_MACB) += emac.o
obj-$(CONFIG_MACB_PCI) += macb_pci.o
