/* SPDX-License-Identifier: GPL-2.0-or-later */

/* DMA Buffer Exporter Kernel Mode Driver. 
 * Copyright (C) 2021 Intel Corporation.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
 
#ifndef _UAPI_LINUX_DMA_BUF_EXPORTER__H
#define _UAPI_LINUX_DMA_BUF_EXPORTER__H
#include <linux/ioctl.h>
#include <linux/types.h>

/* Character device node name */
#define DMA_BUF_EXPORTER_DEV_NAME "dma_buf_exporter"

/* Character device node full path */
#define DMA_BUF_EXPORTER_DEV_PATH "/dev/dma_buf_exporter"

/* dma_buf input/output parameters passed to IOCTL calls
 * made by the application to allocate/free dma_buf
 */
struct dma_exporter_buf_alloc_data {
	__u32 fd; /* dma_buf_fd */
	__u64 size; /* size in bytes */
	__u32 reserved [3];
};

#define DMA_BUF_EXPORTER_MAGIC		'D'

/* IOCTL call to allocate dma_buf and return the fd to the application 
 * This cal takes struct dma_exporter_buf_alloc_data as parameter with
 * size being the input parameter fd is the output parameter.
 */
#define DMA_BUF_EXPORTER_ALLOC		_IOWR(DMA_BUF_EXPORTER_MAGIC, 0, \
				      struct dma_exporter_buf_alloc_data)

/* IOCTL call to free the dma_buf 
 * This cal takes struct dma_exporter_buf_alloc_data as parameter with
 * size and fd being the input parameters for this call.
 */
#define DMA_BUF_EXPORTER_FREE     _IOWR(DMA_BUF_EXPORTER_MAGIC, 8, \
					struct dma_exporter_buf_alloc_data)
                    
#endif /* _UAPI_LINUX_DMA_BUF_EXPORTER__H */