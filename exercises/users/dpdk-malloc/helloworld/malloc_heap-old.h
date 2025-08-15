/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2010-2014 Intel Corporation
 */

#ifndef MALLOC_HEAP_H_
#define MALLOC_HEAP_H_

#include <stdbool.h>
#include <sys/queue.h>

#include <rte_malloc.h>
#include <rte_spinlock.h>

/* Number of free lists per heap, grouped by size. */
#define RTE_HEAP_NUM_FREELISTS  13
#define RTE_HEAP_NAME_MAX_LEN 32

/* dummy definition, for pointers */
struct malloc_elem;

/**
 * Structure to hold malloc heap
 */
struct malloc_heap {
	rte_spinlock_t lock;
	LIST_HEAD(, malloc_elem) free_head[RTE_HEAP_NUM_FREELISTS];
	struct malloc_elem *volatile first;
	struct malloc_elem *volatile last;

	unsigned int alloc_count;
	unsigned int socket_id;
	size_t total_size;
	char name[RTE_HEAP_NAME_MAX_LEN];
} __rte_cache_aligned;

#endif /* MALLOC_HEAP_H_ */
