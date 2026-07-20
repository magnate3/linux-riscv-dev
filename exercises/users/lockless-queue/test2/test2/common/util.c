#include "util.h"
#include "ngx_config.h"
#include "ngx_core.h"
void* multi_proc_lfqueue_alloc(void *pl, size_t sz) {
     return ngx_slab_alloc_locked( pl, sz);
}
void multi_proc_lfqueue_free(void *pl, void *ptr) {
			                ngx_slab_free_locked( pl, ptr);
}
