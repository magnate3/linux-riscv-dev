/*
 * Huawei HiNIC PCI Express Linux driver
 * Copyright(c) 2017 Huawei Technologies Co., Ltd
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms and conditions of the GNU General Public License,
 * version 2, as published by the Free Software Foundation.
 *
 * This program is distributed in the hope it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * for more details.
 *
 */

#include <linux/pci.h>
#include <linux/device.h>
#include <linux/errno.h>
#include <linux/io.h>
#include <linux/types.h>
#include <linux/bitops.h>

#include "hinic_hw_csr.h"
#include "hinic_hw_if.h"

#define PCIE_ATTR_ENTRY         0

#define VALID_MSIX_IDX(attr, msix_index) ((msix_index) < (attr)->num_irqs)

/**
 * hinic_msix_attr_set - set message attribute for msix entry
 * @hwif: the HW interface of a pci function device
 * @msix_index: msix_index
 * @pending_limit: the maximum pending interrupt events (unit 8)
 * @coalesc_timer: coalesc period for interrupt (unit 8 us)
 * @lli_timer: replenishing period for low latency credit (unit 8 us)
 * @lli_credit_limit: maximum credits for low latency msix messages (unit 8)
 * @resend_timer: maximum wait for resending msix (unit coalesc period)
 *
 * Return 0 - Success, negative - Failure
 **/
int hinic_msix_attr_set(struct hinic_hwif *hwif, u16 msix_index,
			u8 pending_limit, u8 coalesc_timer,
			u8 lli_timer, u8 lli_credit_limit,
			u8 resend_timer)
{
	u32 msix_ctrl, addr;

	if (!VALID_MSIX_IDX(&hwif->attr, msix_index))
		return -EINVAL;

	msix_ctrl = HINIC_MSIX_ATTR_SET(pending_limit, PENDING_LIMIT)   |
		    HINIC_MSIX_ATTR_SET(coalesc_timer, COALESC_TIMER)   |
		    HINIC_MSIX_ATTR_SET(lli_timer, LLI_TIMER)           |
		    HINIC_MSIX_ATTR_SET(lli_credit_limit, LLI_CREDIT)   |
		    HINIC_MSIX_ATTR_SET(resend_timer, RESEND_TIMER);

	addr = HINIC_CSR_MSIX_CTRL_ADDR(msix_index);

	hinic_hwif_write_reg(hwif, addr, msix_ctrl);
	return 0;
}

/**
 * hinic_msix_attr_get - get message attribute of msix entry
 * @hwif: the HW interface of a pci function device
 * @msix_index: msix_index
 * @pending_limit: the maximum pending interrupt events (unit 8)
 * @coalesc_timer: coalesc period for interrupt (unit 8 us)
 * @lli_timer: replenishing period for low latency credit (unit 8 us)
 * @lli_credit_limit: maximum credits for low latency msix messages (unit 8)
 * @resend_timer: maximum wait for resending msix (unit coalesc period)
 *
 * Return 0 - Success, negative - Failure
 **/
int hinic_msix_attr_get(struct hinic_hwif *hwif, u16 msix_index,
			u8 *pending_limit, u8 *coalesc_timer,
			u8 *lli_timer, u8 *lli_credit_limit,
			u8 *resend_timer)
{
	u32 addr, val;

	if (!VALID_MSIX_IDX(&hwif->attr, msix_index))
		return -EINVAL;

	addr = HINIC_CSR_MSIX_CTRL_ADDR(msix_index);
	val  = hinic_hwif_read_reg(hwif, addr);

	*pending_limit    = HINIC_MSIX_ATTR_GET(val, PENDING_LIMIT);
	*coalesc_timer    = HINIC_MSIX_ATTR_GET(val, COALESC_TIMER);
	*lli_timer        = HINIC_MSIX_ATTR_GET(val, LLI_TIMER);
	*lli_credit_limit = HINIC_MSIX_ATTR_GET(val, LLI_CREDIT);
	*resend_timer     = HINIC_MSIX_ATTR_GET(val, RESEND_TIMER);
	return 0;
}

/**
 * hinic_msix_attr_cnt_clear - clear message attribute counters for msix entry
 * @hwif: the HW interface of a pci function device
 * @msix_index: msix_index
 *
 * Return 0 - Success, negative - Failure
 **/
int hinic_msix_attr_cnt_clear(struct hinic_hwif *hwif, u16 msix_index)
{
	u32 msix_ctrl, addr;

	if (!VALID_MSIX_IDX(&hwif->attr, msix_index))
		return -EINVAL;

	msix_ctrl = HINIC_MSIX_CNT_SET(1, RESEND_TIMER);
	addr = HINIC_CSR_MSIX_CNT_ADDR(msix_index);

        //pr_info("*********before write,  msic_ctrl val %8x", hinic_hwif_read_reg(hwif, addr));
	hinic_hwif_write_reg(hwif, addr, msix_ctrl);
        //pr_info("*********after write,  msic_ctrl val %8x", hinic_hwif_read_reg(hwif, addr));
	return 0;
}

/**
 * hinic_set_pf_action - set action on pf channel
 * @hwif: the HW interface of a pci function device
 * @action: action on pf channel
 *
 * Return 0 - Success, negative - Failure
 **/
void hinic_set_pf_action(struct hinic_hwif *hwif, enum hinic_pf_action action)
{
	u32 attr5 = hinic_hwif_read_reg(hwif, HINIC_CSR_FUNC_ATTR5_ADDR);

	attr5 = HINIC_FA5_CLEAR(attr5, PF_ACTION);
	attr5 |= HINIC_FA5_SET(action, PF_ACTION);

	hinic_hwif_write_reg(hwif, HINIC_CSR_FUNC_ATTR5_ADDR, attr5);
}

enum hinic_outbound_state hinic_outbound_state_get(struct hinic_hwif *hwif)
{
	u32 attr4 = hinic_hwif_read_reg(hwif, HINIC_CSR_FUNC_ATTR4_ADDR);

	return HINIC_FA4_GET(attr4, OUTBOUND_STATE);
}

void hinic_outbound_state_set(struct hinic_hwif *hwif,
			      enum hinic_outbound_state outbound_state)
{
	u32 attr4 = hinic_hwif_read_reg(hwif, HINIC_CSR_FUNC_ATTR4_ADDR);

	attr4 = HINIC_FA4_CLEAR(attr4, OUTBOUND_STATE);
	attr4 |= HINIC_FA4_SET(outbound_state, OUTBOUND_STATE);

	hinic_hwif_write_reg(hwif, HINIC_CSR_FUNC_ATTR4_ADDR, attr4);
}

enum hinic_db_state hinic_db_state_get(struct hinic_hwif *hwif)
{
	u32 attr4 = hinic_hwif_read_reg(hwif, HINIC_CSR_FUNC_ATTR4_ADDR);

	return HINIC_FA4_GET(attr4, DB_STATE);
}

void hinic_db_state_set(struct hinic_hwif *hwif,
			enum hinic_db_state db_state)
{
	u32 attr4 = hinic_hwif_read_reg(hwif, HINIC_CSR_FUNC_ATTR4_ADDR);

	attr4 = HINIC_FA4_CLEAR(attr4, DB_STATE);
	attr4 |= HINIC_FA4_SET(db_state, DB_STATE);

	hinic_hwif_write_reg(hwif, HINIC_CSR_FUNC_ATTR4_ADDR, attr4);
}

/**
 * hwif_ready - test if the HW is ready for use
 * @hwif: the HW interface of a pci function device
 *
 * Return 0 - Success, negative - Failure
 **/
static int hwif_ready(struct hinic_hwif *hwif)
{
	struct pci_dev *pdev = hwif->pdev;
	u32 addr, attr1;

	addr   = HINIC_CSR_FUNC_ATTR1_ADDR;
	attr1  = hinic_hwif_read_reg(hwif, addr);

	if (!HINIC_FA1_GET(attr1, INIT_STATUS)) {
		dev_err(&pdev->dev, "hwif status is not ready\n");
		return -EFAULT;
	}

	return 0;
}

/**
 * set_hwif_attr - set the attributes in the relevant members in hwif
 * @hwif: the HW interface of a pci function device
 * @attr0: the first attribute that was read from the hw
 * @attr1: the second attribute that was read from the hw
 **/
static void set_hwif_attr(struct hinic_hwif *hwif, u32 attr0, u32 attr1)
{
	hwif->attr.func_idx     = HINIC_FA0_GET(attr0, FUNC_IDX);
	hwif->attr.pf_idx       = HINIC_FA0_GET(attr0, PF_IDX);
	hwif->attr.pci_intf_idx = HINIC_FA0_GET(attr0, PCI_INTF_IDX);
	hwif->attr.func_type    = HINIC_FA0_GET(attr0, FUNC_TYPE);

	hwif->attr.num_aeqs = BIT(HINIC_FA1_GET(attr1, AEQS_PER_FUNC));
	hwif->attr.num_ceqs = BIT(HINIC_FA1_GET(attr1, CEQS_PER_FUNC));
	hwif->attr.num_irqs = BIT(HINIC_FA1_GET(attr1, IRQS_PER_FUNC));
	hwif->attr.num_dma_attr = BIT(HINIC_FA1_GET(attr1, DMA_ATTR_PER_FUNC));
}

/**
 * read_hwif_attr - read the attributes and set members in hwif
 * @hwif: the HW interface of a pci function device
 **/
static void read_hwif_attr(struct hinic_hwif *hwif)
{
	u32 addr, attr0, attr1;

	addr   = HINIC_CSR_FUNC_ATTR0_ADDR;
	attr0  = hinic_hwif_read_reg(hwif, addr);

	addr   = HINIC_CSR_FUNC_ATTR1_ADDR;
	attr1  = hinic_hwif_read_reg(hwif, addr);

	set_hwif_attr(hwif, attr0, attr1);
}

/**
 * set_ppf - try to set hwif as ppf and set the type of hwif in this case
 * @hwif: the HW interface of a pci function device
 **/
static void set_ppf(struct hinic_hwif *hwif)
{
	struct hinic_func_attr *attr = &hwif->attr;
	u32 addr, val, ppf_election;

	/* Read Modify Write */
	addr = HINIC_CSR_PPF_ELECTION_ADDR(HINIC_HWIF_PCI_INTF(hwif));

	val = hinic_hwif_read_reg(hwif, addr);
	val = HINIC_PPF_ELECTION_CLEAR(val, IDX);

	ppf_election = HINIC_PPF_ELECTION_SET(HINIC_HWIF_FUNC_IDX(hwif), IDX);

	val |= ppf_election;
	hinic_hwif_write_reg(hwif, addr, val);

	/* check PPF */
	val = hinic_hwif_read_reg(hwif, addr);

	attr->ppf_idx = HINIC_PPF_ELECTION_GET(val, IDX);
	if (attr->ppf_idx == HINIC_HWIF_FUNC_IDX(hwif))
		attr->func_type = HINIC_PPF;
}

/**
 * set_dma_attr - set the dma attributes in the HW
 * @hwif: the HW interface of a pci function device
 * @entry_idx: the entry index in the dma table
 * @st: PCIE TLP steering tag
 * @at: PCIE TLP AT field
 * @ph: PCIE TLP Processing Hint field
 * @no_snooping: PCIE TLP No snooping
 * @tph_en: PCIE TLP Processing Hint Enable
 **/
static void set_dma_attr(struct hinic_hwif *hwif, u32 entry_idx,
			 u8 st, u8 at, u8 ph,
			 enum hinic_pcie_nosnoop no_snooping,
			 enum hinic_pcie_tph tph_en)
{
	u32 addr, val, dma_attr_entry;

	/* Read Modify Write */
	addr = HINIC_CSR_DMA_ATTR_ADDR(entry_idx);

	val = hinic_hwif_read_reg(hwif, addr);
	val = HINIC_DMA_ATTR_CLEAR(val, ST)             &
	      HINIC_DMA_ATTR_CLEAR(val, AT)             &
	      HINIC_DMA_ATTR_CLEAR(val, PH)             &
	      HINIC_DMA_ATTR_CLEAR(val, NO_SNOOPING)    &
	      HINIC_DMA_ATTR_CLEAR(val, TPH_EN);

	dma_attr_entry = HINIC_DMA_ATTR_SET(st, ST)                     |
			 HINIC_DMA_ATTR_SET(at, AT)                     |
			 HINIC_DMA_ATTR_SET(ph, PH)                     |
			 HINIC_DMA_ATTR_SET(no_snooping, NO_SNOOPING)   |
			 HINIC_DMA_ATTR_SET(tph_en, TPH_EN);

	val |= dma_attr_entry;
	hinic_hwif_write_reg(hwif, addr, val);
}

/**
 * dma_attr_table_init - initialize the the default dma attributes
 * @hwif: the HW interface of a pci function device
 **/
static void dma_attr_init(struct hinic_hwif *hwif)
{
	set_dma_attr(hwif, PCIE_ATTR_ENTRY, HINIC_PCIE_ST_DISABLE,
		     HINIC_PCIE_AT_DISABLE, HINIC_PCIE_PH_DISABLE,
		     HINIC_PCIE_SNOOP, HINIC_PCIE_TPH_DISABLE);
}
static int
print_bars(struct pci_dev *pdev)
{
        int i, iom, iop;
        unsigned long flags;
        unsigned long addr, len;
        static const char *bar_names[PCI_STD_RESOURCE_END + 1]  = {
                "BAR0",
                "BAR1",
                "BAR2",
                "BAR3",
                "BAR4",
                "BAR5",
        };

        iom = 0;
        iop = 0;

        for (i = 0; i < ARRAY_SIZE(bar_names); i++) {
             pr_err("******bar name : %s", bar_names[i]);
                addr = pci_resource_start(pdev, i);
                len = pci_resource_len(pdev, i);
                if (len != 0 && addr != 0) {
                        flags = pci_resource_flags(pdev, i);
                        if (flags & IORESOURCE_MEM) {
                                iom++;
                        } else if (flags & IORESOURCE_IO) {
                                iop++;
                        }
                        pr_info("flags  %lx,  and addr %lx, and len %lx \n", flags & IORESOURCE_MEM,  addr, len);
                }
        }
        return 0;
}
static void print_hwif_attr(struct hinic_hwif *hwif)
{
        u32 addr, attr0, attr1;

        addr   = HINIC_CSR_FUNC_ATTR0_ADDR;
        attr0  = hinic_hwif_read_reg(hwif, addr);

        addr   = HINIC_CSR_FUNC_ATTR1_ADDR;
        attr1  = hinic_hwif_read_reg(hwif, addr);
        pr_info("in hinic driver attr0 %8x, and attr1 %8x \n",  attr0, attr1);

}
/**
 * hinic_init_hwif - initialize the hw interface
 * @hwif: the HW interface of a pci function device
 * @pdev: the pci device for acessing PCI resources
 *
 * Return 0 - Success, negative - Failure
 **/
int hinic_init_hwif(struct hinic_hwif *hwif, struct pci_dev *pdev)
{
	int err;

	hwif->pdev = pdev;

        pr_err("in %s before call pci_ioremap_bar",__func__);
        print_bars(pdev);
	hwif->cfg_regs_bar = pci_ioremap_bar(pdev, HINIC_PCI_CFG_REGS_BAR);
        pr_info("map addr %p", hwif->cfg_regs_bar);
        pr_err("in %s after call pci_ioremap_bar",__func__);
        print_bars(pdev);
	if (!hwif->cfg_regs_bar) {
		dev_err(&pdev->dev, "Failed to map configuration regs\n");
		return -ENOMEM;
	}

	err = hwif_ready(hwif);
	if (err) {
		dev_err(&pdev->dev, "HW interface is not ready\n");
		goto err_hwif_ready;
	}

	read_hwif_attr(hwif);

        print_hwif_attr(hwif);
	if (HINIC_IS_PF(hwif))
		set_ppf(hwif);
	/* No transactionss before DMA is initialized */
	dma_attr_init(hwif);
	return 0;

err_hwif_ready:
	iounmap(hwif->cfg_regs_bar);
	return err;
}

/**
 * hinic_free_hwif - free the HW interface
 * @hwif: the HW interface of a pci function device
 **/
void hinic_free_hwif(struct hinic_hwif *hwif)
{
	iounmap(hwif->cfg_regs_bar);
}
