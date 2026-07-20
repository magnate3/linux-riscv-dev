/* $Id: hashtable.h,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_HASHTABLE_H
#define WAN_HASHTABLE_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

#include "pool.h"

#define DEFAULT_SIZE 64

#include "general.h"

EXTERN_TYPE(hash_entry);
EXTERN_TYPE(hash_table);

hash_table       hash_new(int size);
void             hash_destroy(hash_table *ht);
hash_entry       hash_make_entry(void *key, void *value);
void             hash_insert(hash_table ht, hash_entry he);
void *           hash_lookup(hash_table ht, void *key);
pool             hash_get_pool();

#endif // WAN_HASHTABLE_H
