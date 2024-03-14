#include <stdio.h>
#include "hugepage_malloc.h"
#include "string.h"
#include "common.h"
#include "runtime_info.h"

hugepage_malloc_heap global_malloc_heap[MAX_SOCKET_NB];

//init a elem
void malloc_elem_init(hugepage_malloc_elem *elem, hugepage_malloc_heap *heap, const hugepage_memseg *ms, size_t size)
{
	elem->heap = heap;
	elem->ms = ms;
	elem->prev = NULL;
	memset(&elem->free_list, 0, sizeof(elem->free_list));
	elem->size = size;
	elem->state = ELEM_FREE;		
}

//create end elem 
void malloc_elem_mkend(hugepage_malloc_elem *elem, hugepage_malloc_elem *prev)
{
	malloc_elem_init(elem, prev->heap, prev->ms, 0);
	elem->prev = prev;
	elem->state = ELEM_BUSY;
}

//find suitable free list id
//    idx      |     size   
//free_list[0]---(0,    2^ 8]
//free_list[1]---(2^ 8, 2^10]
//free_list[2]---(2^10, 2^12]
//free_list[3]---(2^12, 2^14]
//free_list[4]---(2^14, 2^16]
//...
size_t find_free_list_idx(size_t size)
{
	#define MALLOC_MINSIZE_LOG2 8
	#define MALLOC_LOG2_INCREMENT 2
	
	size_t log2;
	size_t index;
	
	//if size <= 2^8, return 0
	if(size <= (1UL << MALLOC_MINSIZE_LOG2))
		return 0;
	
	//find the smallest 2 ^ (log2) >= size
	//__builtin_clzl(long var) : In binary bits array of var ,return the number of 0 before the first "1" from left to right.
	//"log2" is the largest number that is power of 2 and smaller than "size"
	//for example, when size is 4096
	//the largest number, that is smaller than 4097 and is power of 2, is 4096
	//so the "log2" is log2(4096) = 12
	//unsigned long var consist of 64 bits
	//how to calculate?(size = 4096)
	//4096          = 00000000 00000000 00000000 00000000 00000000 00000000 00010000 00000000
	//4096-1 = 4095 = 00000000 00000000 00000000 00000000 00000000 00000000 00001111 11111111
	//log2 = 64 - 52 = 12 = log2(4096)  
	log2 = sizeof(size) * 8 - __builtin_clzl(size-1);

	//why "(log2-min_log2)+(increment - 1)/increment"?
	//for example:
	//we want to "7/2 = 3.5 = 4", but system will get 3 instead of 4
	//so we make 7 add "increment-1" to get result we want 	
	index = (log2 - MALLOC_MINSIZE_LOG2 + (MALLOC_LOG2_INCREMENT - 1)) / MALLOC_LOG2_INCREMENT;

	return index <= MAX_FREE_LIST_NB - 1 ? index: MAX_FREE_LIST_NB - 1; 	
}

//insert elem into free list
void malloc_elem_insert_freelist(hugepage_malloc_elem *elem)
{
	size_t idx;

    idx = find_free_list_idx(elem->size);	
	elem->state = ELEM_FREE;
	LIST_INSERT_HEAD(&elem->heap->free_head[idx], elem, free_list);	
}

//add a memseg to the heap of specified socket_id
uint32_t memsegs_to_heaps(hugepage_memseg *ms)
{
	int socket_id = ms->socket_id;
	hugepage_malloc_elem *start_elem = (hugepage_malloc_elem *)ms->addr;
	hugepage_malloc_elem *end_elem = (void *)((uintptr_t)start_elem 
									+ (ms->len - ELEM_HEADER_SIZE)); 	
 	end_elem = ALIGN_PTR_FLOOR(end_elem, ALIGN_SIZE);
	const size_t elem_size = (uintptr_t)end_elem - (uintptr_t)start_elem;	
	
	malloc_elem_init(start_elem, &global_malloc_heap[socket_id], ms, elem_size);
	malloc_elem_mkend(end_elem, start_elem);
	malloc_elem_insert_freelist(start_elem);
	
	global_malloc_heap[socket_id].total_size += elem_size;				
}

//global malloc hugepage heap initialization
uint32_t global_heap_init()
{
	uint32_t ms_cnt, i;
	hugepage_memseg *ms;
	
	memset(global_malloc_heap, 0, sizeof(hugepage_malloc_heap) * MAX_SOCKET_NB);
	
	//init heap lock
	for(i=0; i<MAX_SOCKET_NB; i++){
		cus_spinlock_init(&global_malloc_heap[i].heap_lock);
	}	

	for(ms = &global_memseg[0], ms_cnt = 0; 
			(ms_cnt < nb_memsegs) && ms->len > 0;
				ms++, ms_cnt++)
	{
		//add a memseg into heap
	    memsegs_to_heaps(ms);		
	}
	if(ms_cnt < nb_memsegs)
		return 1;
	else
		return 0;
}

//print heaps' free_list state
void show_heap_free_list(const hugepage_malloc_heap * const heap, int idx)
{
	if(idx < 0 || idx >= MAX_FREE_LIST_NB)
		return;
	hugepage_malloc_elem *elem;
	int is_1th_elem = 0;
	int th=0;	

//	for(elem = LIST_FIRST(&heap->free_head[idx]);
////						  elem != NULL; 
//						  !!elem;
//						  elem = LIST_NEXT(elem, free_list))
	LIST_FOREACH(elem, &heap->free_head[idx], free_list)
	{
		if(is_1th_elem == 0){
			printf("Free_list_%d:",idx);
			printf("[elem%d-size:%lu]", th++, elem->size);
			is_1th_elem = 1;
		}
		else{
			printf("->[elem%d-size:%lu]", th++, elem->size);
		}
	}
	if(th != 0)
		printf("\n");
}

//print current heaps' state
void show_heaps_state()
{
	uint32_t i,j;
	for(i=0; i<MAX_SOCKET_NB; i++){
		if(global_malloc_heap[i].total_size > 0){
			printf("heap_%u total_size:%lu alloc_counter:%u\n", i,
												global_malloc_heap[i].total_size,
												global_malloc_heap[i].alloc_counter);
			for(j=0; j<MAX_FREE_LIST_NB; j++){
				show_heap_free_list(&global_malloc_heap[i], j);	
			}
		}
	}
	return;
}

//calculate new elem start addr
void * get_new_malloc_elem_start(hugepage_malloc_elem *elem, size_t size, unsigned align)
{
	uintptr_t end_pt = (uintptr_t)elem + elem->size;
	uintptr_t new_data_start = ALIGN_FLOOR((end_pt - size), align);
	uintptr_t new_elem_start;

	//malloc_elem layout:
	//|-------------------------------elem-------------------------------|
	//|<--struct hugepage_malloc_elem-->|<-------------data------------->|
	new_elem_start = new_data_start - ELEM_HEADER_SIZE;
	if(new_elem_start < (uintptr_t)elem)
		return NULL;
	return (void *)new_elem_start;
}

//checkt if elem fit given size
int elem_fit_size(hugepage_malloc_elem *elem, size_t size, size_t align)
{
	return get_new_malloc_elem_start(elem, size, align) != NULL;	
}

//find suitable free elem
hugepage_malloc_elem* find_suitable_elem(hugepage_malloc_heap *heap, size_t size, 
									 size_t align)
{
	size_t idx;
	hugepage_malloc_elem *elem=NULL, *tmp_elem=NULL;

	//if fail try  free list of bigger elem	
	for(idx = find_free_list_idx(size);
			idx<MAX_FREE_LIST_NB; idx++)
	{
	//	for(elem = LIST_FIRST(&heap->free_head[idx]);//start from first elem of current free list 
	//	//	elem != NULL; 
	//		!!elem;
	//		elem = LIST_NEXT(elem, free_list))
		LIST_FOREACH(elem, &heap->free_head[idx], free_list)
		{
			if(elem_fit_size(elem, size, align) == 1){
				return elem;
			}
			else
				tmp_elem = elem;
		}	
	}

	if( tmp_elem != NULL)
		return tmp_elem;
	return NULL;
}

//splite new_elem out from elem
void* split_elem(hugepage_malloc_elem *elem, hugepage_malloc_elem *new_elem)
{
	hugepage_malloc_elem *next_elem = PTR_ADD(elem, elem->size);  
	const size_t old_elem_size = (uintptr_t)new_elem - (uintptr_t)elem;
	const size_t new_elem_size = elem->size - old_elem_size;

	malloc_elem_init(new_elem, elem->heap, elem->ms, new_elem_size);
	new_elem->prev = elem;
	if( next_elem != NULL )
		next_elem->prev = new_elem;
	elem->size = old_elem_size;
}

//remove the elem from current freelist
void malloc_elem_remove(hugepage_malloc_elem *elem)
{
	LIST_REMOVE(elem,free_list);	
}

//malloc on one elem
hugepage_malloc_elem* malloc_on_elem(hugepage_malloc_elem *elem, size_t size, 
									unsigned align)
{
	hugepage_malloc_elem * new_elem = get_new_malloc_elem_start(elem, size, align);
	const size_t old_elem_size = (uintptr_t)new_elem - (uintptr_t)elem;
	const size_t trailer_size = elem->size - old_elem_size - size - ELEM_HEADER_SIZE;
		

	//elem layout:
	//|-----------------------------------elem------------------------------------|
	//|----------------old_elem-------------|-------------new_elem----------------|
	//|---HEADER---|--------data------------|---HEADER---|---data----|---trailer--|
	//|<----------------------------------elem_size------------------------------>|
	//|<--------------old_size------------->|            |<---size-->|
	//according to the layout
	//some addr space will be left because of align_floor
	//if the align value is large enough, the trailer space may be larger than HEADER+MIN_DATA_SIZE(64)
	//if it is the case, split it and re-insert trailer into free_list
	malloc_elem_remove(elem);
	if( trailer_size >= MIN_ELEM_SIZE )
	{
		struct hugepage_malloc_elem *new_free_elem = PTR_ADD(new_elem, size + ELEM_HEADER_SIZE);
		
		split_elem(new_elem, new_free_elem);
		malloc_elem_insert_freelist(new_free_elem);			
	}
	
	//if old_size is smaller than HEADER_SIZE + MIN_DATA_SIZE, do not split it, just set it state to unusable
	//and set the new elem's state is pad
	//when free the new_elem back to free_list, the pad will be use to find the original elem start 
	//and free the whole elem into free list, that consists of new_elem and old_elem
	if(old_elem_size < MIN_ELEM_SIZE )
	{
		elem->state = ELEM_BUSY;
		elem->pad = old_elem_size;
		if(old_elem_size>0){
			new_elem->pad = elem->pad;
			new_elem->state = ELEM_PAD;
			new_elem->size = elem->size - elem->pad;
		}
		return new_elem;	 
	}	
	
	//if the old_elem is larger enough, re-insert it into free_list
	split_elem(elem, new_elem);
	new_elem->state = ELEM_BUSY;
	malloc_elem_insert_freelist(elem);
	
	return new_elem;			
}

//malloc on specified heap
void * malloc_on_heap(hugepage_malloc_heap *heap, size_t size, size_t align)
{
	hugepage_malloc_elem *elem = NULL;
	
	//align size and align
	size = ALIGN_SIZE_ROUNDUP(size);
	align = ALIGN_SIZE_ROUNDUP(align);

	//try entry critical area and lock
	cus_spinlock_lock(&heap->heap_lock);
	
	elem = find_suitable_elem(heap, size, align);
	if(elem != NULL){//if find a suitable elem in current heap, try alloc from it
		elem = malloc_on_elem(elem, size, align);
		heap->alloc_counter += 1;//alloc counter++
	}
		
	//go out critical area and unlock
	cus_spinlock_unlock(&heap->heap_lock);

	if(elem != NULL)
		return (void *)(&elem[1]);//elem[0]  storing a struct hugapage_malloc_elem	as a header
	else
		return NULL;
}

//malloc on specified socket
void * malloc_on_socket(size_t size, unsigned align, int socket_id)
{
	int socket, i;
	void *ret;

	//check if size is not 0 and align is power of 2
	if( size == 0 || (align <=0 && !is_power_of_two(align)) )
		return NULL;

	ret = malloc_on_heap(&global_malloc_heap[socket_id], size, align);
	
	if(ret != NULL)
		return ret;
		
	//try other socket's heap
	for(i=0 ; i < MAX_SOCKET_NB; i++ )
	{
		//tried this socket already before
		if( i == socket_id )
			continue;	
		ret = malloc_on_heap(&global_malloc_heap[i], size, align);
		if( ret != NULL)
			return ret;
	}	

	return NULL;
}

//malloc API
void *mem_malloc(size_t size, unsigned align)
{
	int socket_id = get_cur_socket_id();
	
	//check if socket_id is valid
	if(socket_id < 0 || socket_id >= MAX_SOCKET_NB)
		return NULL;
	
	return malloc_on_socket(size, align, socket_id);
		
}

//join two elem together
//put elem2 following elem1
void malloc_elem_join(hugepage_malloc_elem *elem1, hugepage_malloc_elem *elem2)
{
	hugepage_malloc_elem *next_elem = PTR_ADD(elem2,elem2->size);
	elem1->size += elem2->size;
	if(next_elem!=NULL)
		next_elem->prev = elem1;
}

//free elem into free list
//if the next elem or the previous elem is free
//join them together and re-insert into free_list
int free_elem(hugepage_malloc_elem *elem)
{
	if(elem->state != ELEM_BUSY)
		return -1;
	cus_spinlock_lock(&elem->heap->heap_lock);
	size_t data_sz = elem->size - ELEM_HEADER_SIZE;	
	uint8_t *ptr = (uint8_t *)&elem[1];
	hugepage_malloc_elem *next = PTR_ADD(elem, elem->size);
	
	//check if next elem is free
	if(next->state == ELEM_FREE){
		malloc_elem_remove(next);
		malloc_elem_join(elem, next);
		data_sz += next->size;
	}	

	//check if previous elem is free
	if(elem->prev != NULL && elem->prev->state == ELEM_FREE){
		malloc_elem_remove(elem->prev);
		malloc_elem_join(elem->prev, elem);
		data_sz += ELEM_HEADER_SIZE;
		ptr -= ELEM_HEADER_SIZE;
		elem = elem->prev;	
	}
	malloc_elem_insert_freelist(elem);

	//decrease heap's alloc counter
	elem->heap->alloc_counter -= 1;
	memset(ptr,0,data_sz);
	
	cus_spinlock_unlock(&elem->heap->heap_lock);
	return 0;
}

//get elem from given data addr
hugepage_malloc_elem* get_elem_from_data(const void * data)
{
	if(data == NULL)
		return NULL;
	hugepage_malloc_elem *elem = PTR_SUB(data, ELEM_HEADER_SIZE);
	if(elem->state == ELEM_PAD)//what pad means decribed in malloc_on_elem()
		return PTR_SUB(elem, elem->pad);
	return elem;
}

//free API
void mem_free(void *data)
{
	if(data == NULL)
		return;
	if(free_elem( get_elem_from_data(data) ) != 0)
	{
		printf("Free invalid mem error...\n");
		return;	
	}	
}


