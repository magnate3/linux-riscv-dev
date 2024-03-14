#pragma once
#include <sys/queue.h>//LIST_***
#include "hugepage_memory.h"
#include "cus_spinlock.h"

#define ELEM_HEADER_SIZE ( sizeof(hugepage_malloc_elem) )
#define MIN_ELEM_SIZE ( MIN_DATA_SIZE + ELEM_HEADER_SIZE )

struct hugepage_malloc_heap;//dummy definition of hugepage_heap struct to use it in hugepage_malloc_elem

enum elem_state{
	ELEM_FREE = 0,
	ELEM_BUSY,
	ELEM_PAD
};

struct hugepage_malloc_elem{
	struct hugepage_malloc_heap *heap;
	struct hugepage_malloc_elem *volatile prev;
	LIST_ENTRY(hugepage_malloc_elem) free_list;
	const struct hugepage_memseg *ms;
	size_t size;
	uint32_t pad; 
	enum elem_state state;
}__attribute__((__aligned__(64)));

typedef struct hugepage_malloc_elem hugepage_malloc_elem;

struct hugepage_malloc_heap{
	LIST_HEAD(	,hugepage_malloc_elem) free_head[MAX_FREE_LIST_NB];
	size_t total_size;
	size_t using_size;
	unsigned alloc_counter;
	cus_spinlock_t heap_lock;
}__attribute__((__aligned__(64)));

typedef struct hugepage_malloc_heap hugepage_malloc_heap;

//API
uint32_t global_heap_init();
void show_heaps_state();
void * mem_malloc(size_t size, unsigned align);//align must be larger than 0 and power of 2
void mem_free(void * data);

//global vars
extern hugepage_malloc_heap global_malloc_heap[MAX_SOCKET_NB];
