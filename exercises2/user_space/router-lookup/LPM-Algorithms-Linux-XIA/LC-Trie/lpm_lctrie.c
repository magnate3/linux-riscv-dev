/*
   Garnaik Sumeet, Michel Machado 2015
   LPM Algorithms in Linux-XIA
*/

#include "lpm_lctrie.h"
#include <assert.h>

static int fib_format(entry_t entry[], struct nextcreate *table,
		unsigned long size)
{
	int i;

	for (i = 0; i < size; i++) {
		memcpy(&(entry[i]->data), table[i].prefix, HEXXID);
		entry[i]->len = table[i].len;
		entry[i]->nexthop = table[i].nexthop;
	}

	return 0;
}

struct routtablerec *lctrie_create_fib(struct nextcreate *table,
		unsigned long size)
{
	entry_t *entry = malloc(size * sizeof(entry_t));
	int nentries;
	int i;
	struct routtablerec *fib;

	struct entryrec *tmp_entry = calloc(size, sizeof(struct entryrec));
	for (i = 0; i < size; i++)
		entry[i] = (tmp_entry + i);
	assert(0 == fib_format(entry, table, size));

	fib = buildrouttable(entry, size);
	free(entry);
	return fib;
}

int lctrie_destroy_fib(struct routtablerec *rtable)
{
	free(rtable->trie);
	free(rtable->base);
	free(rtable->nexthop);
	free(rtable);

	return 0;
}

unsigned int lctrie_lookup(const xid *id, routtable_t table)
{
	return find(id, table);
}
