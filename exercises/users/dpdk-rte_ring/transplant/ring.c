/*-
 *   BSD LICENSE
 *
 *   Copyright(c) 2010-2015 Intel Corporation. All rights reserved.
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in
 *       the documentation and/or other materials provided with the
 *       distribution.
 *     * Neither the name of Intel Corporation nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *   OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * Derived from FreeBSD's bufring.c
 *
 **************************************************************************
 *
 * Copyright (c) 2007,2008 Kip Macy kmacy@freebsd.org
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. The name of Kip Macy nor the names of other
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
#include <sys/queue.h>

#include "ring.h"

__thread int ring_debug_stats_num;
/* true if x is a power of 2 */
#define POWEROF2(x) ((((x)-1) & (x)) == 0)

/* return the size of memory occupied by a ring */
ssize_t ring_get_memsize(unsigned count)
{
	ssize_t sz;

	/* count must be a power of 2 */
	if ((!POWEROF2(count)) || (count > RING_SZ_MASK )) {
		fprintf(stderr, "RING:"
			"Requested size is invalid, must be power of 2, and "
			"do not exceed the size limit %u\n", RING_SZ_MASK);
		return -EINVAL;
	}

	sz = sizeof(struct ring) + count * sizeof(void *);
	sz = (size_t) ((sz + (size_t) (CONFIG_CACHE_LINE_MASK)) & ~((size_t)(CONFIG_CACHE_LINE_MASK)));
	return sz;
}

int ring_init(struct ring *r, const char *name, unsigned count,
	unsigned flags)
{
	int ret;

	/* compilation-time checks */
	BUILD_BUG_ON((sizeof(struct ring) &
			  CONFIG_CACHE_LINE_MASK) != 0);
	BUILD_BUG_ON((offsetof(struct ring, cons) &
			  CONFIG_CACHE_LINE_MASK) != 0);
	BUILD_BUG_ON((offsetof(struct ring, prod) &
			  CONFIG_CACHE_LINE_MASK) != 0);
#ifdef CONFIG_RING_DEBUG
	BUILD_BUG_ON((sizeof(struct ring_debug_stats) &
			  CONFIG_CACHE_LINE_MASK) != 0);
	BUILD_BUG_ON((offsetof(struct ring, stats) &
			  CONFIG_CACHE_LINE_MASK) != 0);
#endif

	/* init the ring structure */
	memset(r, 0, sizeof(*r));
	ret = snprintf(r->name, sizeof(r->name), "%s", name);
	if (ret < 0 || ret >= (int)sizeof(r->name))
		return -ENAMETOOLONG;
	r->flags = flags;
	r->prod.watermark = count;
	r->prod.sp_enqueue = !!(flags & RING_F_SP_ENQ);
	r->cons.sc_dequeue = !!(flags & RING_F_SC_DEQ);
	r->prod.size = r->cons.size = count;
	r->prod.mask = r->cons.mask = count-1;
	r->prod.head = r->cons.head = 0;
	r->prod.tail = r->cons.tail = 0;

	return 0;
}

/* create the ring */
struct ring * ring_create(const char *name, unsigned count, unsigned flags)
{
	struct ring *r;
	const struct rte_memzone *mz;
	ssize_t ring_size;
	int mz_flags = 0;
	int ret;

	ring_size = ring_get_memsize(count);
	if (ring_size < 0) {
		return NULL;
	}

	//rte_rwlock_write_lock(RTE_EAL_TAILQ_RWLOCK);

	r = calloc(1, ring_size);
	if (!r) {
		return NULL;
	}
	/* no need to check return value here, we already checked the
	 * arguments above */
	ring_init(r, name, count, flags);

//	rte_rwlock_write_unlock(RTE_EAL_TAILQ_RWLOCK);

	return r;
}

/* free the ring */
void
ring_free(struct ring *r)
{
	if (r == NULL)
		return;

	free(r);
	//rte_rwlock_write_lock(RTE_EAL_TAILQ_RWLOCK);


	//rte_rwlock_write_unlock(RTE_EAL_TAILQ_RWLOCK);
}

/*
 * change the high water mark. If *count* is 0, water marking is
 * disabled
 */
int
ring_set_water_mark(struct ring *r, unsigned count)
{
	if (count >= r->prod.size)
		return -EINVAL;

	/* if count is 0, disable the watermarking */
	if (count == 0)
		count = r->prod.size;

	r->prod.watermark = count;
	return 0;
}

/* dump the status of the ring on the console */
void
ring_dump(FILE *f, const struct ring *r)
{
#ifdef CONFIG_RING_DEBUG
	struct ring_debug_stats sum;
	unsigned lcore_id;
#endif

	fprintf(f, "ring <%s>@%p\n", r->name, r);
	fprintf(f, "  flags=%x\n", r->flags);
	fprintf(f, "  size=%"PRIu32"\n", r->prod.size);
	fprintf(f, "  ct=%"PRIu32"\n", r->cons.tail);
	fprintf(f, "  ch=%"PRIu32"\n", r->cons.head);
	fprintf(f, "  pt=%"PRIu32"\n", r->prod.tail);
	fprintf(f, "  ph=%"PRIu32"\n", r->prod.head);
	fprintf(f, "  used=%u\n", ring_count(r));
	fprintf(f, "  avail=%u\n", ring_free_count(r));
	if (r->prod.watermark == r->prod.size)
		fprintf(f, "  watermark=0\n");
	else
		fprintf(f, "  watermark=%"PRIu32"\n", r->prod.watermark);

	/* sum and dump statistics */
#ifdef CONFIG_RING_DEBUG
	memset(&sum, 0, sizeof(sum));
	for (lcore_id = 0; lcore_id < 128; lcore_id++) {
		sum.enq_success_bulk += r->stats[lcore_id].enq_success_bulk;
		sum.enq_success_objs += r->stats[lcore_id].enq_success_objs;
		sum.enq_quota_bulk += r->stats[lcore_id].enq_quota_bulk;
		sum.enq_quota_objs += r->stats[lcore_id].enq_quota_objs;
		sum.enq_fail_bulk += r->stats[lcore_id].enq_fail_bulk;
		sum.enq_fail_objs += r->stats[lcore_id].enq_fail_objs;
		sum.deq_success_bulk += r->stats[lcore_id].deq_success_bulk;
		sum.deq_success_objs += r->stats[lcore_id].deq_success_objs;
		sum.deq_fail_bulk += r->stats[lcore_id].deq_fail_bulk;
		sum.deq_fail_objs += r->stats[lcore_id].deq_fail_objs;
	}
	fprintf(f, "  size=%"PRIu32"\n", r->prod.size);
	fprintf(f, "  enq_success_bulk=%"PRIu64"\n", sum.enq_success_bulk);
	fprintf(f, "  enq_success_objs=%"PRIu64"\n", sum.enq_success_objs);
	fprintf(f, "  enq_quota_bulk=%"PRIu64"\n", sum.enq_quota_bulk);
	fprintf(f, "  enq_quota_objs=%"PRIu64"\n", sum.enq_quota_objs);
	fprintf(f, "  enq_fail_bulk=%"PRIu64"\n", sum.enq_fail_bulk);
	fprintf(f, "  enq_fail_objs=%"PRIu64"\n", sum.enq_fail_objs);
	fprintf(f, "  deq_success_bulk=%"PRIu64"\n", sum.deq_success_bulk);
	fprintf(f, "  deq_success_objs=%"PRIu64"\n", sum.deq_success_objs);
	fprintf(f, "  deq_fail_bulk=%"PRIu64"\n", sum.deq_fail_bulk);
	fprintf(f, "  deq_fail_objs=%"PRIu64"\n", sum.deq_fail_objs);
#else
	fprintf(f, "  no statistics available\n");
#endif
}

/* dump the status of all rings on the console */
void
ring_list_dump(FILE *f)
{
#if 0
	const struct rte_tailq_entry *te;
	struct ring_list *ring_list;

	ring_list = RTE_TAILQ_CAST(ring_tailq.head, ring_list);

	rte_rwlock_read_lock(RTE_EAL_TAILQ_RWLOCK);

	TAILQ_FOREACH(te, ring_list, next) {
		ring_dump(f, (struct ring *) te->data);
	}

	rte_rwlock_read_unlock(RTE_EAL_TAILQ_RWLOCK);
#endif
}

/* search a ring from its name */
struct ring *
ring_lookup(const char *name)
{
#if 0
	struct rte_tailq_entry *te;
	struct ring *r = NULL;
	struct ring_list *ring_list;

	ring_list = RTE_TAILQ_CAST(ring_tailq.head, ring_list);

	rte_rwlock_read_lock(RTE_EAL_TAILQ_RWLOCK);

	TAILQ_FOREACH(te, ring_list, next) {
		r = (struct ring *) te->data;
		if (strncmp(name, r->name, ring_NAMESIZE) == 0)
			break;
	}

	rte_rwlock_read_unlock(RTE_EAL_TAILQ_RWLOCK);

	if (te == NULL) {
		rte_errno = ENOENT;
		return NULL;
	}

	return r;
#endif
}