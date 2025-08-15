#include "lpm_bloom.h"

static int comparekeys(void *key1, void *key2)
{
	unsigned char *k1 = (unsigned char *) key1;
	unsigned char *k2 = (unsigned char *) key2;

	return memcmp(k1, k2, HEXXID);
}

static int sortbylength(const void *e1, const void *e2)
{
	int res;
	struct nextcreate **entry1 = (struct nextcreate **) e1;
	struct nextcreate **entry2 = (struct nextcreate **) e2;
	unsigned int len1 = (*entry1)->len;
	unsigned int len2 = (*entry2)->len;

	if (len1 < len2)
		return -1;
	else if (len1 > len2)
		return 1;
	else
		return 0;
}

static int bloom_proportion(struct bloom_structure *filter,
		struct nextcreate **tmp_table, unsigned long size)
{
	int j = 0;
	int i;

	for (i = MINLENGTH; i <= MAXLENGTH; i++) {
		if (i != tmp_table[j]->len)
			continue;
		filter->low[i - MINLENGTH] = j;
		while (i == tmp_table[j]->len) {
			j++;
			if (size == j)
				break;
		}
		filter->high[i - MINLENGTH] = (j - 1);
		filter->length[i - MINLENGTH] = j - filter->low[i - MINLENGTH];
		filter->bloom[i - MINLENGTH] = NULL;
		filter->hashtable[i - MINLENGTH] = NULL;
		filter->flag[i - MINLENGTH] = 1;
		if (size == j)
			break;
	}
}

int bloom_destroy_fib(struct bloom_structure *filter)
{
	int i;

	for (i = 0; i < WDIST; i++) {
		if (filter->flag[i]) {
			free_counting_bloom(filter->bloom[i]);
			hashit_destroy(filter->hashtable[i]);
		}
	}
	free(filter);

	return 0;
}

struct bloom_structure *bloom_create_fib(struct nextcreate *table,
		unsigned long size, double error_rate)
{
	int i, j, k, m, *next = alloca(sizeof(int));
	unsigned char tmp[HEXXID] = {0};
	struct nextcreate **tmp_table = NULL;
	struct nextcreate **dup_table = NULL;
	struct bloom_structure *filter = NULL;
	hash_t tmp_hashmap = NULL;

	filter = calloc(1, sizeof(struct bloom_structure));
	assert(filter);
	memset(filter->low, -1, WDIST * sizeof(int));
	memset(filter->high, -1, WDIST * sizeof(int));

	tmp_table = malloc(size * sizeof(struct nextcreate *));
	dup_table = malloc(DUPS * sizeof(struct nextcreate *));
	assert(tmp_table);
	for (i = 0; i < size; i++)
		tmp_table[i] = &table[i];

	qsort(tmp_table, size, sizeof(struct nextcreate *), sortbylength);
	bloom_proportion(filter, tmp_table, size);

	for (i = 0; i < WDIST; i++) {
		if (!filter->flag[i])
			continue;
		if (!(filter->bloom[i] =
		new_counting_bloom(filter->length[i] * NTIMES, error_rate))) {
			printf("ERROR: Could not create bloom filter\n");
			return NULL;
		}
		tmp_hashmap = hashit_create(filter->length[i], HEXXID, NULL, comparekeys, CHAIN_H);
		for (j = filter->low[i]; j <= filter->high[i]; j++) {
			memcpy(&tmp, &(tmp_table[j]->prefix), HEXXID);
			assert(0 == counting_bloom_add(filter->bloom[i], tmp_table[j]->prefix, HEXXID));
			assert(0 == hashit_insert(tmp_hashmap, tmp_table[j]->prefix, &(tmp_table[j]->nexthop)));
		}
		filter->hashtable[i] = tmp_hashmap;
	}
	free(tmp_table);
	return filter;
}

unsigned int lookup_bloom(unsigned char (*id)[HEXXID], unsigned int len,
		void *bf)
{
	int i, min_len;
	struct bloom_structure *filter = (struct bloom_structure *) bf;
	unsigned int *nexthop = NULL;
	// The returned values of counting_bloom_check() are 0 if found else 1
	unsigned char matchvec[WDIST] = {0};
	unsigned char tmp1[HEXXID + 1] = {0};
	unsigned char tmp2[HEXXID] = {0};

	memcpy(tmp1, id, HEXXID);
	memcpy(tmp2, tmp1, HEXXID);
	for (i = 0; i < WDIST; i++)
		matchvec[i] = 1;
	// Although the paper suggests to perform parallel membership queries
	for (i = MAXLENGTH; i >= MINLENGTH; i--) {
		tmp1[i / BYTE] = (tmp1[i / BYTE] >> (BYTE - i % BYTE)) <<
							(BYTE - i % BYTE);
		if (!filter->flag[i - MINLENGTH])
			continue;
		matchvec[i - MINLENGTH] =
		counting_bloom_check(filter->bloom[i - MINLENGTH], tmp1,
								HEXXID);
	}
	// Parse the matchvec from longest to shortest to perform table search
	for (i = MAXLENGTH; i >= MINLENGTH; i--) {
		tmp2[i / BYTE] = (tmp2[i / BYTE] >> (BYTE - i % BYTE)) <<
							(BYTE - i % BYTE);
		if (!matchvec[i - MINLENGTH] || !filter->flag[i - MINLENGTH])
			continue;
		nexthop = hashit_lookup(filter->hashtable[i - MINLENGTH],
					tmp2);
		if (nexthop)
			return *nexthop;
	}
	return 0;
}
