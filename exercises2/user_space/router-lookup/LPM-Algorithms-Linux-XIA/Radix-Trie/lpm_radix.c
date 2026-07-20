/*
 * Garnaik Sumeet, Michel Machado 2015
 * LPM Algorithms for Linux-XIA
 */

#include "lpm_radix.h"
#include <assert.h>

static int fib_format(struct entryrec *entry[], struct nextcreate *table,
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

struct routtablerec *radix_create_fib(struct nextcreate *table,
		unsigned long size)
{
	struct entryrec **entry = malloc(size * sizeof(struct entryrec *));
	int nentries;
	int i;
	struct routtablerec *fib = NULL;

	struct entryrec *tmp_entry = calloc(size, sizeof(struct entryrec));
	for (i = 0; i < size; i++)
		entry[i] = (tmp_entry + i);
	assert(0 == fib_format(entry, table, size));

	fib = buildrouttable(entry, size);
	free(entry);
	return fib;
}

int radix_destroy_fib(struct routtablerec *rtable)
{
	free_trie(rtable->root);
	free(rtable->base);
	free(rtable->nexthop);
	free(rtable);

	return 0;
}

unsigned int lookup_radix(const xid *id, struct routtablerec *table, int opt)
{
	return find(*id, table, opt);
}
