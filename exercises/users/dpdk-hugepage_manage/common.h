#pragma once
#include <emmintrin.h>//_mm_pause()

#define ALIGN_SIZE 64
#define MIN_DATA_SIZE 64

//align to upper boundary
//page layout :|---page_1---|---page_2---|---....---|---page_n---|
//before align:    |addr-------|
//after align :             |addr--------|
#define ALIGN_ADDR(addr,size) ((addr + (size-1)) & (~(size-1)))

//align to down boundart
//align       :|---align---|---align---|...|---align---|
//before align:		|----val----|
//after align :|----val----|
#define ALIGN_FLOOR(val, align) \
	(typeof(val))( (val) & (~(typeof(val))((align) - 1)) )

#define ALIGN_PTR_FLOOR(ptr, align) \
	((typeof(ptr))ALIGN_FLOOR((uintptr_t)ptr, align))

//set size aligned to next round based on ALGN_SIZE, that is 64 by default 
#define ALIGN_SIZE_ROUNDUP(size) \
	(ALIGN_SIZE * ((size + ALIGN_SIZE - 1) / (ALIGN_SIZE))) 

//ptr add
#define PTR_ADD(ptr,x) ( (void *)( (uintptr_t)(ptr) + (x) ) )

//ptr sub
#define PTR_SUB(ptr,x) ( (void *)( (uintptr_t)(ptr) - (x) ) )

static inline int
is_power_of_two(int var)
{
	return var && !(var & (var-1));
}

static inline void
cus_pause()
{
	_mm_pause();
	return;	
}
