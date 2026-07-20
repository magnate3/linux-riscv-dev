/*
   xidsort.c

   Using the standard sorting funtion defined in stdlib.h
   This function uses a polymorphic sorting algorithm and
   the comparison function passed to qsort uses memcmp()
   as defined in build_table.c
*/
#include <stdlib.h>
#include "lc_trie.h"

/* Sort an array of XIDs pointed by "base" of size "narray" */
void xidsort(list *base, size_t narray, size_t size, int (*compare)(const void *, const void *))
{
	qsort(&base[0], narray, size, compare);
}

/* Sort an array of entry structures pointed by "entry" of size "narray" */
void xidentrysort(entry_t entry[], size_t narray, size_t size, int (*compare)(const void *, const void *))
{
	qsort(&entry[0], narray, size, compare);
}
