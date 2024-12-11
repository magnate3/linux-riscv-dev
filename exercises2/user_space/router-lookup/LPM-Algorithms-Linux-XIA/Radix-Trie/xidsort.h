/*
   xidsort.h
*/
#ifndef _XIDSORT_H_
#define _XIDSORT_H_

#include "radix.h"

void xidsort(list *base, size_t narray, size_t size,
		int (*compare)(const void *, const void *));
void xidentrysort(struct entryrec *entry[], size_t narray, size_t size,
		int (*compare)(const void *, const void *));

#endif
