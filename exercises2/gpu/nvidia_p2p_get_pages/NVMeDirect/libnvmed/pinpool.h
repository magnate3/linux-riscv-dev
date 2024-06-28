////////////////////////////////////////////////////////////////////////
//
// Copyright 2014 PMC-Sierra, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you
// may not use this file except in compliance with the License. You may
// obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0 Unless required by
// applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for
// the specific language governing permissions and limitations under the
// License.
//
////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////
//
//   Author: Logan Gunthorpe
//
//   Date:   Oct 23 2014
//
//   Description:
//     Pinned Memory Pool
//
////////////////////////////////////////////////////////////////////////


#ifndef __LIBDONARD_PINPOOL_H__
#define __LIBDONARD_PINPOOL_H__

#include <stdlib.h>

struct pin_buf {
    void *address;
    void *handle;
    size_t bufsize;
    void *mmap;
};

#define PINPOOL_MAX_BUFS -1

int pinpool_init(int count, size_t bufsize);
void pinpool_deinit(void);

struct pin_buf *pinpool_alloc(void);
void pinpool_free(struct pin_buf *p);
void *pinpool_mmap(struct pin_buf *p);
int pinpool_munmap(struct pin_buf *p);

#endif
