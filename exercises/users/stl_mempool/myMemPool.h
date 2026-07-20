#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>


/**
 *  this is a mempool implement
 *  free-list -|     --------------------
               |->   |     |             |
               |-    --------------------
               |     --------------------
               |     |                   |  -> for client use
               |     --------------------
               |     --------------------
               |->   |     |             |
                     --------------------
 */

enum { __ALIGN = 8 };
enum { __MAX_BYTES = 128};
enum { __NFREELISTS = __MAX_BYTES/__ALIGN };

typedef union obj_t{
    union obj_t *free_list_link;
    char client_data[1];
}obj;
//-------------------------------------

typedef struct mem_pool_manager_t{
    obj *volatile free_list[__NFREELISTS]; //free list for mem
    char *start_free;       //mempool start position
    char *end_free;         //mempoll end position
    size_t heap_size;
}mem_pool_manager;
//-------------------------------------------------
// for use outside
// 
bool mem_pool_init(mem_pool_manager *pmanager);
void *allocate(mem_pool_manager *pmanager, size_t n);
void deallocate(mem_pool_manager *pmanager, void *p, size_t n);
