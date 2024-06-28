//////////////////////////////////////////////////////////////////////
//                             PMC-Sierra, Inc.
//
//
//
//                             Copyright 2014
//
////////////////////////////////////////////////////////////////////////
//
// This program is free software; you can redistribute it and/or modify it
// under the terms and conditions of the GNU General Public License,
// version 2, as published by the Free Software Foundation.
//
// This program is distributed in the hope it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.
//
////////////////////////////////////////////////////////////////////////
//
//   Author:  Logan Gunthorpe
//
//   Description:
//     Manage Nvidia Pin Buf Memory
//
////////////////////////////////////////////////////////////////////////


#ifndef _UAPI_LINUX_DONARD_NV_PINBUF_H
#define _UAPI_LINUX_DONARD_NV_PINBUF_H

#include <linux/types.h>


struct nv_gpu_mem {
    __u64 address;
    __u64 size;
    unsigned long long p2pToken;
    unsigned int vaSpaceToken;
    void *handle;
};

#define NV_PIN_GPU_MEMORY     _IOWR('N', 0x43, struct nv_gpu_mem)
#define NV_UNPIN_GPU_MEMORY   _IOWR('N', 0x44, struct nv_gpu_mem)
#define NV_SELECT_MMAP_MEMORY _IOWR('N', 0x45, void *)
#define NV_NVMED_WAKEUP _IOWR('N', 0x48, void *)
#define NV_NVMED_ACCQUIRE _IOWR('N', 0x49, void *)
#define NV_NVMED_RESET _IOWR('N', 0x47, void *)
#define NV_NVMED_EXCLUDE _IOWR('N', 0x50, void *)
#define NV_NVMED_RELEASE _IOWR('N', 0x51, void *)


#endif
