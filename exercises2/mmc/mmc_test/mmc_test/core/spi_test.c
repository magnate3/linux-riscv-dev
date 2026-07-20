// SPDX-License-Identifier: GPL-2.0-or-later
/*
 *  Copyright 2007-2008 Pierre Ossman
 */

//#include <linux/mmc/core.h>
//#include <linux/mmc/card.h>
//#include <linux/mmc/host.h>
//#include <linux/mmc/mmc.h>
#include <linux/spi/spi.h>
#include <linux/idr.h>
#include <linux/slab.h>

#include <linux/scatterlist.h>
#include <linux/swap.h>		/* For nr_free_buffer_pages() */
#include <linux/list.h>

#include <linux/debugfs.h>
#include <linux/uaccess.h>
#include <linux/seq_file.h>
#include <linux/module.h>

#include "spi-dw.h"

#define SPI_DEVICE_NAME "spi0.0"
typedef int (*TRANSFER_ONE_FUNC)(struct spi_controller *ctlr, struct spi_device *spi, struct spi_transfer *transfer);
static TRANSFER_ONE_FUNC old=NULL;
static struct spi_device *find_spi_device_by_name(const char * name)
{
	struct device *dev;
	dev = bus_find_device_by_name(&spi_bus_type, NULL, name);
	return dev ? to_spi_device(dev) : NULL;
}

/* Return the max entries we can fill into tx fifo */
static inline u32 tx_max(struct dw_spi *dws)
{
	u32 tx_room, rxtx_gap;

	tx_room = dws->fifo_len - dw_readl(dws, DW_SPI_TXFLR);

	/*
	 * Another concern is about the tx/rx mismatch, we
	 * though to use (dws->fifo_len - rxflr - txflr) as
	 * one maximum value for tx, but it doesn't cover the
	 * data which is out of tx/rx fifo and inside the
	 * shift registers. So a control from sw point of
	 * view is taken.
	 */
	rxtx_gap = dws->fifo_len - (dws->rx_len - dws->tx_len);

	return min3((u32)dws->tx_len, tx_room, rxtx_gap);
}

/* Return the max entries we should read out of rx fifo */
static inline u32 rx_max(struct dw_spi *dws)
{
	return min_t(u32, dws->rx_len, dw_readl(dws, DW_SPI_RXFLR));
}
static void dw_writer(struct dw_spi *dws)
{
	u32 max = tx_max(dws);
	u32 txw = 0;

	while (max--) {
		if (dws->tx) {
			if (dws->n_bytes == 1)
				txw = *(u8 *)(dws->tx);
			else if (dws->n_bytes == 2)
				txw = *(u16 *)(dws->tx);
			else
				txw = *(u32 *)(dws->tx);

			dws->tx += dws->n_bytes;
		}
		dw_write_io_reg(dws, DW_SPI_DR, txw);
		--dws->tx_len;
	}
}

static void dw_reader(struct dw_spi *dws)
{
	u32 max = rx_max(dws);
	u32 rxw;

	while (max--) {
		rxw = dw_read_io_reg(dws, DW_SPI_DR);
		if (dws->rx) {
			if (dws->n_bytes == 1)
				*(u8 *)(dws->rx) = rxw;
			else if (dws->n_bytes == 2)
				*(u16 *)(dws->rx) = rxw;
			else
				*(u32 *)(dws->rx) = rxw;

			dws->rx += dws->n_bytes;
		}
		--dws->rx_len;
	}
}

static int dw_spi_poll_transfer(struct dw_spi *dws,
				struct spi_transfer *transfer)
{
	struct spi_delay delay;
	u16 nbits;
	int ret;

	delay.unit = SPI_DELAY_UNIT_SCK;
	nbits = dws->n_bytes * BITS_PER_BYTE;

	do {
		dw_writer(dws);

		delay.value = nbits * (dws->rx_len - dws->tx_len);
		spi_delay_exec(&delay, transfer);

		dw_reader(dws);

		ret = dw_spi_check_status(dws, true);
		if (ret)
			return ret;
	} while (dws->rx_len);

	return 0;
}
int transfer_one_test(struct spi_controller *master, struct spi_device *spi, struct spi_transfer *transfer)
{
#if 0
     pr_info(" %s  \n", __func__);
     return old(master, spi, transfer);
#else
	struct dw_spi *dws = spi_controller_get_devdata(master);
	struct dw_spi_cfg cfg = {
		.tmode = SPI_TMOD_TR,
		.dfs = transfer->bits_per_word,
		.freq = transfer->speed_hz,
	};
	int ret = 0;

	//pr_err("******************SPI transfer return: %d, cur msg status %d \n", ret,  dws->master->cur_msg->status);
	dws->dma_mapped = 0;
	dws->n_bytes = DIV_ROUND_UP(transfer->bits_per_word, BITS_PER_BYTE);
	dws->tx = (void *)transfer->tx_buf;
	dws->tx_len = transfer->len / dws->n_bytes;
	dws->rx = transfer->rx_buf;
	dws->rx_len = dws->tx_len;

	/* Ensure the data above is visible for all CPUs */
	smp_mb();

	spi_enable_chip(dws, 0);

	dw_spi_update_config(dws, spi, &cfg);

	transfer->effective_speed_hz = dws->current_freq;
#if 0
	/* Check if current transfer is a DMA transaction */
	if (master->can_dma && master->can_dma(master, spi, transfer))
		dws->dma_mapped = master->cur_msg_mapped;

	/* For poll mode just disable all interrupts */
	spi_mask_intr(dws, 0xff);

	if (dws->dma_mapped) {
		ret = dws->dma_ops->dma_setup(dws, transfer);
		if (ret)
			return ret;
	}

	spi_enable_chip(dws, 1);

	if (dws->dma_mapped)
		return dws->dma_ops->dma_transfer(dws, transfer);
	else if (dws->irq == IRQ_NOTCONNECTED)
		return dw_spi_poll_transfer(dws, transfer);

	dw_spi_irq_setup(dws);
#else
	/* For poll mode just disable all interrupts */
	spi_mask_intr(dws, 0xff);
	spi_enable_chip(dws, 1);
	ret = dw_spi_poll_transfer(dws, transfer);
	//pr_err("SPI transfer return: %d, cur msg status %d \n", ret,  dws->master->cur_msg->status);
	// refer to  spi_transfer_one_message
	if (ret < 0) {
	        pr_err("SPI transfer failed: %d\n", ret);
	} 
	else
        {
		//dws->master->cur_msg->status = 0;
	}
	return ret;
#endif
	return 1;
#endif
}
static int __init spi_test_init(void)
{
	struct spi_controller *ctrl = NULL;
	int  id =0;
	struct spi_device * spi = find_spi_device_by_name(SPI_DEVICE_NAME);
	old = NULL;
	if(!spi)
	{
	     pr_info("spi device %s not found \n", SPI_DEVICE_NAME);
	}
	else
	{
	     
	     pr_info("spi device %s  found \n", SPI_DEVICE_NAME);
	     ctrl = spi->controller;
	}
	//found = idr_find(&spi_master_idr, id);
	if(!ctrl)
	{
	     pr_info("spi controller id %d not found \n", id);

	}
        else
	{
	     mutex_lock(&ctrl->bus_lock_mutex);
	     old = ctrl->transfer_one;
	     ctrl->transfer_one =  transfer_one_test;
#if 0
	     struct dw_spi *dws = spi_controller_get_devdata(ctrl);
#endif
	     mutex_unlock(&ctrl->bus_lock_mutex);
        }
	return 0;
}

static void __exit spi_test_exit(void)
{

	struct spi_controller *ctrl = NULL;
	int  id =0;
	struct spi_device * spi = find_spi_device_by_name(SPI_DEVICE_NAME);
	if(!spi)
	{
	     pr_info("spi device %s not found \n", SPI_DEVICE_NAME);
	}
	else
	{
	     
	     pr_info("spi device %s  found \n", SPI_DEVICE_NAME);
	     ctrl = spi->controller;
	}
	if(!ctrl)
	{
	     pr_info("spi controller id %d not found \n", id);

	}
        else
	{
	     mutex_lock(&ctrl->bus_lock_mutex);
	     if(old)
	     {
	         ctrl->transfer_one =  old;
	     }
	     mutex_unlock(&ctrl->bus_lock_mutex);
        }
}

module_init(spi_test_init);
module_exit(spi_test_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Multimedia Card (MMC) host test driver");
MODULE_AUTHOR("Pierre Ossman");
