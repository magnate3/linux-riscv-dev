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

#define SPI_DEVICE_NAME "spi0.0"
typedef int (*TRANSFER_ONE_FUNC)(struct spi_controller *ctlr, struct spi_device *spi, struct spi_transfer *transfer);
TRANSFER_ONE_FUNC old;
static struct spi_device *find_spi_device_by_name(const char * name)
{
	struct device *dev;
	dev = bus_find_device_by_name(&spi_bus_type, NULL, name);
	return dev ? to_spi_device(dev) : NULL;
}
int transfer_one_test(struct spi_controller *ctlr, struct spi_device *spi, struct spi_transfer *transfer)
{
     pr_info(" %s  \n", __func__);
     return old(ctlr, spi, transfer);
}
static int __init spi_test_init(void)
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
