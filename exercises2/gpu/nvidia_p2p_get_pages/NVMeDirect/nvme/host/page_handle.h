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
//     Shared Page Handle Structure
//
////////////////////////////////////////////////////////////////////////

#ifndef __DONARD_PAGE_HANDLE_H__
#define __DONARD_PAGE_HANDLE_H__

#include "nv-p2p.h"

#define PAGE_HANDLE_ID 0xA859B77C

struct page_handle {
    unsigned long id;
    struct nvidia_p2p_page_table *page_table;
};



#endif
