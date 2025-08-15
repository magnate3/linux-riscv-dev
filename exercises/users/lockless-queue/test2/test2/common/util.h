#ifndef __H_UTIL__
#define __H_UTIL__
#include <unistd.h>
inline void* multi_proc_lfqueue_alloc(void *pl, size_t sz);
inline void multi_proc_lfqueue_free(void *pl, void *ptr);
#endif
