# SPDX-License-Identifier: GPL-2.0
#
# Copyright 2020 UPMEM. All rights reserved.

obj-m += dpu.o

dpu-objs += dpu_region.o dpu_region_address_translation.o
dpu-objs += dpu_rank.o dpu_rank_sysfs.o
dpu-objs += dpu_dax.o
dpu-objs += dpu_control_interface.o
dpu-objs += dpu_mcu_ci_protocol.o
dpu-objs += dpu_region_dev.o dpu_region_srat.o dpu_rank_dmi.o
dpu-objs += ufi_ci.o ufi_bit_config.o ufi_dma_wavegen_config.o ufi.o
dpu-objs += dpu_config.o
dpu-objs += dpu_runner.o
dpu-objs += dpu_management.o
dpu-objs += dpu_memory.o
dpu-objs += dpu_rank_mcu.o
dpu-objs += dpu_power_management.o
dpu-objs += dpu_membo.o
format-source  = modules/dpu_region.c modules/dpu_region_address_translation.c
format-source += modules/dpu_rank.c modules/dpu_rank_sysfs.c
format-source += modules/dpu_dax.c
format-source += modules/dpu_control_interface.c
format-source += modules/dpu_mcu_ci_protocol.c
format-source += modules/dpu_region_dev.c modules/dpu_region_srat.c modules/dpu_rank_dmi.c
format-source += modules/ufi_ci.c modules/ufi_bit_config.c modules/ufi.c
format-source += modules/dpu_config.c
format-source += modules/dpu_runner.c
format-source += modules/dpu_management.c
format-source += modules/dpu_memory.c
format-source += modules/dpu_rank_mcu.c
format-source += modules/dpu_power_management.c
format-source += modules/dpu_membo.c
format-source += modules/dpu_rank_mcu.h
format-source += modules/dpu_region.h modules/dpu_region_address_translation.h
format-source += modules/dpu_rank.h modules/dpu_rank_ioctl.h
format-source += modules/dpu_dax.h
format-source += modules/dpu_control_interface.h
format-source += modules/dpu_mcu_ci_protocol.h modules/dpu_mcu_ci_commands.h modules/dpu_mcu_ci_compat.h
format-source += modules/dpu_region_constants.h modules/dpu_region_dev.h
format-source += modules/dpu_utils.h
format-source += modules/dpu_types.h
format-source += modules/ufi_rank_utils.h
format-source += modules/ufi/ufi_ci.h modules/ufi/ufi_ci_commands.h modules/ufi/ufi_ci_types.h modules/ufi/ufi_bit_config.h
format-source += modules/dpu_config.h
format-source += modules/dpu_pci_ids.h
format-source += modules/dpu_power_management.h
format-source += modules/uapi/dpu.h
format-source += modules/uapi/dpu_runner.h
format-source += modules/uapi/dpu_management.h
format-source += modules/uapi/dpu_memory.h
format-source += modules/dpu_membo.h
format-source += modules/dpu_membo_ioctl.h

# Mappings
dpu-objs 	+= ../mappings/fpga_kc705/dpu_fpga_kc705_translation.o
dpu-objs 	+= ../mappings/fpga_kc705/dpu_fpga_kc705_dma_op.o
dpu-objs 	+= ../mappings/fpga_kc705/dpu_fpga_kc705_spi.o
dpu-objs 	+= ../mappings/fpga_kc705/dpu_fpga_kc705_fs.o
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_translation.c
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_spi.c
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_spi.h
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_device.h
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_fs.c
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_fs.h
format-source 	+= mappings/fpga_kc705/dpu_fpga_kc705_register.h

dpu-objs 	+= ../mappings/fpga_aws/dpu_fpga_aws_translation.o
dpu-objs 	+= ../mappings/fpga_aws/dpu_fpga_aws_libxdma.o
format-source 	+= mappings/fpga_aws/dpu_fpga_aws_translation.c

ifeq (${CONFIG_X86_64},y)
dpu-objs	+= ../mappings/xeon_sp/dpu_xeon_sp_translation.o
endif
format-source	+= mappings/xeon_sp/dpu_xeon_sp_translation.c

ifeq (${CONFIG_PPC64},y)
dpu-objs	+= ../mappings/power9/dpu_power9_translation.o
endif
format-source	+= mappings/power9/dpu_power9_translation.c

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(dir $(mkfile_path))
KDIR ?= /lib/modules/$(shell uname -r)/build
SOURCE_KDIR ?= $(KDIR)

ifneq ("$(wildcard /etc/debian_version)", "")
uapi_path := /usr/src/linux-headers-$(shell uname -r)/include/uapi
uapi_path := $(subst amd64,common,$(uapi_path))
else
uapi_path := /usr/src/kernels/$(shell uname -r)/include/uapi/
endif

ccflags-y += -I$(SOURCE_KDIR)/drivers/dax
ccflags-y += -I$(KDIR)/drivers/dax
ccflags-y += -I$(current_dir)
ccflags-y += -I$(current_dir)/uapi

ccflags-y += -I$(current_dir)/../mappings/fpga_kc705/ -I$(current_dir)/../mappings/fpga_kc705/
ccflags-y += -I$(current_dir)/../mappings/fpga_aws/ -I$(current_dir)/../mappings/fpga_aws/

all:
	$(MAKE) -C $(KDIR) M=$(current_dir) modules

modules_install:
	$(MAKE) -C $(KDIR) M=$(current_dir) modules_install

clean:
	$(MAKE) -C $(KDIR) M=$(current_dir) clean
	rm -f ../mappings/fpga_kc705/dpu_fpga_kc705_translation.o
	rm -f ../mappings/fpga_kc705/dpu_fpga_kc705_dma_op.o
	rm -f ../mappings/fpga_kc705/dpu_fpga_kc705_spi.o
	rm -f ../mappings/fpga_kc705/dpu_fpga_kc705_fs.o
	rm -f ../mappings/fpga_aws/dpu_fpga_aws_translation.o
	rm -f ../mappings/fpga_aws/dpu_fpga_aws_libxdma.o
	rm -f ../mappings/xeon_sp/dpu_xeon_sp_translation.o
	rm -f ../mappings/power9/dpu_power9_translation.o

api_install:
	sudo install -d $(uapi_path)/dpu
	sudo install -m 644 uapi/dpu.h uapi/dpu_management.h uapi/dpu_memory.h uapi/dpu_runner.h $(uapi_path)/dpu/

api_clean:
	sudo rm -rf $(uapi_path)/dpu

format:
	clang-format -style=file -i $(patsubst %,../%,$(format-source))

space := $(subst ,, )

print-format-source:
		@echo "$(subst $(space),|,$(format-source))"

sign: all
	/usr/src/kernels/$(shell uname -r)/scripts/sign-file sha256 ./MOK.priv ./MOK.der dpu.ko
