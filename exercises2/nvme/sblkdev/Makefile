# SPDX-License-Identifier: GPL-2.0

# Configuration and compile options for standalone module version in a separate
# file. The upstream version should contains the configuration in the Kconfig
# file, and should be free from all branches of conditional compilation.
include ${M}/Makefile-standalone

sblkdev-y := main.o device.o
obj-$(CONFIG_SBLKDEV) += sblkdev.o
