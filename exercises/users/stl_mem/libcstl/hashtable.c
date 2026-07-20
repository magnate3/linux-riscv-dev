/* $Id: hashtable.c,v 1.3 2003/09/15 12:47:46 jfwan Exp $ */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hashtable.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef struct hash_entry_struct
{
    void *key;
    void *value;
} hash_entry_;

typedef struct hash_table_struct
{
    hash_entry  *vec;
    int          size;
    int          used;
} hash_table_;

static pool hash_pool   = NULL;
static int  hash_number = 0; /* objects' counter */

static void check_hash_pool();

hash_table hash_new(int size)
{
    hash_table ht;
    check_hash_pool();
    ht = (hash_table) pmalloc(hash_pool, sizeof(hash_table_));

    if (size <= 0)
        size = DEFAULT_SIZE;
    else
        size = ALIGN(size);

    ht->vec  = (hash_entry*) pmalloc(hash_pool, size * sizeof(hash_entry));
    ht->size = size;
    ht->used = 0;

    /* increase the counter */
    ++hash_number;

    return ht;
}

void hash_destroy(hash_table *pht)
{
    if (pht == NULL || *pht == NULL)
        return;

    if (--hash_number <= 0) {
        pool_free(hash_pool);
        hash_pool = NULL;
    }

    *pht = NULL;
}

hash_entry hash_make_entry(void *key, void *value)
{
    hash_entry he;
    check_hash_pool();
    he = (hash_entry) pmalloc(hash_pool, sizeof(hash_entry_));
    he->key   = key;
    he->value = value;
    return he;
}

/* XXX: I don't think a program should has more than 64 command line options,
 * if so, I think it should be re-designed.
 * jfwan@vip.sina.com 07/07/2003
 */
void hash_insert(hash_table ht, hash_entry he)
{
    if (ht->used >= ht->size) {
        fprintf(stderr, "Error: too many options to contain");
        exit(1);
    }
    ht->vec[ht->used++] = he;
}

void * hash_lookup(hash_table ht, void *key)
{
    int i;

    for (i = 0; i < ht->used; ++i)
        if (!strcmp((char*) key, (char*) ht->vec[i]->key))
            return ht->vec[i]->value;

    return NULL;
}

pool hash_get_pool()
{
    return hash_pool;
}

static void check_hash_pool()
{
    if (hash_pool == NULL)
        hash_pool = pool_new(0);
}
