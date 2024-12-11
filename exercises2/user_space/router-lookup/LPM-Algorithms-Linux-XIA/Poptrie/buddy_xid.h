/*_
 * Copyright (c) 2014-2015 Hirochika Asai <asai@jar.jp>
 * All rights reserved.
 */

#include "poptrie_xid.h"
#include <stdlib.h>

#ifndef _POPTRIE_BUDDY_XID_H
#define _POPTRIE_BUDDY_XID_H

/*
 * Buddy system
 */
struct buddy {
	/* Size of buddy system (# of blocks) */
	int sz;
	/* Size of each block */
	int bsz;
	/* Bitmap */
	u8 *b;
	/* Memory blocks */
	void *blocks;
	/* Level */
	int level;
	/* Heads */
	u32 *buddy;
};

/* buddy.c */
int buddy_init(struct buddy *, int, int, int);
void buddy_release(struct buddy *);
void * buddy_alloc(struct buddy *, int);
int buddy_alloc2(struct buddy *, int);
void buddy_free(struct buddy *, void *);
void buddy_free2(struct buddy *, int);

#endif
