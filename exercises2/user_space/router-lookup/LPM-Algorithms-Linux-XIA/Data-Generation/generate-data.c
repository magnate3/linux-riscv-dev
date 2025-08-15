/* This file is concerned with the generation of synthetic data for evaluation
 * of LPM algorithms in Linux-XIA. By synthetic data, we refer to the FIB for
 * LPM on XIDs.
 *
 * This is a free software and licensed under GNU General Public License Version
 * 2.
 *
 * Garnaik Sumeet, Michel Machado 2015
 */

#include "generate_fibs.h"

/*
 * Print the table
 */
int print_table(struct nextcreate *table, int size, int prexp)
{
	unsigned int i, j;
	unsigned char tmp_prefix[HEXXID];
	char file_name[MAXFILENAME];
	FILE *tmp_file;

	memset(file_name, 0, MAXFILENAME);
	sprintf(file_name, "%d.table", prexp);
	tmp_file = fopen(file_name, "w+");

	for (i = 0; i < size; i++) {
		memcpy(tmp_prefix, table[i].prefix, HEXXID);
		for (j = 0; j < HEXXID; j++)
			fprintf(tmp_file, "%02x", tmp_prefix[j]);
		fprintf(tmp_file, " %d %d\n", table[i].len, table[i].nexthop);
	}
	fclose(tmp_file);
}

/*
 * Comparison function to be passed to qsort for sorting of entries
 */
static int sortentries(const void *e1, const void *e2)
{
	int res;
	struct nextcreate **entry1 = (struct nextcreate **) e1;
	struct nextcreate **entry2 = (struct nextcreate **) e2;
	unsigned char *pre1 = (*entry1)->prefix;
	unsigned char *pre2 = (*entry2)->prefix;

	res = memcmp(pre1, pre2, 20);

	if (res < 0)
		return -1;
	else if (res > 0)
		return 1;
	else
		return 0;
}

/*
 * This routine generates an unsigned char containing `8 - bit` random bits using
 * the random variable `r` from MSB to LSB.
 *
 * Inputs:
 * r: Random variable used by GSL
 * bit: Number of random bits = 8 - bit
 */
static unsigned char uniform_char(gsl_rng *r, int bit)
{
	int i;
	unsigned char tmp_char = 0;
	unsigned char res = 0;
	unsigned long int tmp_bit;

	for (i = (BYTE - 1); i >= bit; i--) {
		tmp_bit = gsl_rng_uniform_int(r, BITRAND);
		tmp_char = (1 == tmp_bit) ? 1 << i : 0;
		res = res | tmp_char;
	}

	return res;
}

/* 
 * This routine is concerned with the generation of nexthops for the prefix
 * entries in the table. The total number of unique nexthops in each table is
 * passed as parameter to the function.
 *
 * Inputs 
 * table: An array consisting of entry structs
 * size: number of entries in the table
 * r: random number generator
 * nnexthops: number of unique nexthops
 */
static int nexthop_dist(struct nextcreate *table, int size, gsl_rng *r,
		int nnexthops)
{
	int i, j;
	unsigned long int tmp;

	for (i = 0; i < size; i++) {
		tmp = gsl_rng_uniform_int(r, nnexthops) + 1;
		table[i].nexthop = (unsigned int) tmp;
	}

	return 0;
}

static struct nextcreate *lduplicates(struct nextcreate **key,
				struct nextcreate **base, int num)
{
	struct nextcreate *ret = NULL;
	int i;

	for (i = 0; i < num; i++) {
		if (0 == sortentries(key, (base + i))) {
			ret = *(base + i);
			break;
		}
	}

	return ret;
}

/*
 * This routine is concerned with the replacement of duplicates in the FIBs
 * generated such that all the entries in the FIB are distinct.
 *
 * Inputs:
 * dup_table: array consisting of indexes of duplicate structs
 * dsize: size of dup_table
 * sortedtable: array consisting of sorted entries
 * ssize: size of sortedtable
 * r: random number generator
 */
static int remove_duplicates(struct nextcreate **dup_table, int dsize,
		struct nextcreate **sortedtable, int ssize, gsl_rng *r)
{
	int i, j;
	int bytes, bits;
	int flag;
	unsigned char tmp_prefix[HEXXID] = {0};
	struct nextcreate **match_sorted = NULL;
	struct nextcreate *match_dup = NULL;
	struct nextcreate *tmp_entry = malloc(sizeof(struct nextcreate));

	for (i = 0; i < dsize; i++) {
		flag = TRUE;
		while (flag) {
			memset(tmp_prefix, 0, HEXXID);
			memset(tmp_entry->prefix, 0, HEXXID);
			bytes = dup_table[i]->len / BYTE;
			bits = dup_table[i]->len % BYTE;
			for (j = 0; j < bytes; j++)
				tmp_prefix[j] = uniform_char(r, 0);
			tmp_prefix[j] = uniform_char(r, (BYTE - bits));
			memcpy(tmp_entry->prefix, tmp_prefix, HEXXID);
			tmp_entry->len = dup_table[i]->len;
			match_sorted = bsearch(&tmp_entry,
				&sortedtable[0], ssize,
				sizeof(struct nextcreate *), sortentries);
			match_dup = lduplicates(&tmp_entry, &dup_table[0],
									dsize);
			if ((NULL == match_sorted) && (NULL == match_dup)) {
				flag = FALSE;
				memcpy(dup_table[i]->prefix,
					tmp_entry->prefix, HEXXID);
			}
		}
	}

	free(tmp_entry);
	return 0;
}

/*
 * This routine finds and replaces duplicates in the FIB table.
 *
 * Inputs:
 * table: An array consisting of entry structs
 * size: number of entries in the table
 * r: random number generator
 */
static int deduplication(struct nextcreate *table, int size, gsl_rng *r)
{
	int i, j, k;
	struct nextcreate **tmp_table = NULL;
	struct nextcreate **dup_table = NULL;

	// To actually stop moving the structs around
	tmp_table = malloc(size * sizeof(struct nextcreate *));
	assert(tmp_table);
	for (i = 0; i < size; i++)
		tmp_table[i] = &table[i];

	qsort(tmp_table, size, sizeof(struct nextcreate *), sortentries);
	dup_table = malloc(DUPS * sizeof(struct nextcreate *));
	assert(dup_table);
	j = 0;
	k = 0; // Index of number of distinct entries
	for (i = 1; i < size; i++) {
		if (0 == sortentries(&tmp_table[i - 1], &tmp_table[i])) {
			dup_table[j] = tmp_table[i - 1];
			assert(++j < DUPS);
		} else {
			tmp_table[k] = tmp_table[i - 1];
			k++;
		}
	}
	remove_duplicates(dup_table, j, tmp_table, k, r);

	free(tmp_table);
	free(dup_table);
}

/* 
 * This routine is concerned with the generation of uniformly distributed
 * prefixes of certain prefix length.
 *
 * Inputs:
 * table: An array consisting of entry structs
 * size: number of entries in the table
 * r: random number generator
 */
static int prefix_dist(struct nextcreate *table, int size, gsl_rng *r)
{
	int i, j, k;
	int bytes, bits;
	unsigned char tmp_prefix[HEXXID] = {0};

	for (i = 0; i < size; i++) {
		memset(table[i].prefix, 0, HEXXID);
		memset(tmp_prefix, 0, HEXXID);
		bytes = table[i].len / BYTE;
		bits = table[i].len % BYTE;
		for (j = 0; j < bytes; j++)
			tmp_prefix[j] = uniform_char(r, 0);
		tmp_prefix[j] = uniform_char(r, (BYTE - bits));
		memcpy(table[i].prefix, tmp_prefix, HEXXID);
	}

	return 0;
}

/* 
 * This routine is concerned with the generation of uniformly distributed prefix
 * lengths. We are aware of this strong assumption and will soon replace this
 * with a more suitable distribution.
 *
 * table: An array consisting of entry structs
 * size: number of entries in the table
 * r: random number generator
 */
static int prelength_dist(struct nextcreate *table, int size, uint32_t *seeds,
						int low, double alpha)
{
	int i;
	unsigned long int tmp;
	struct zipf_cache zcache;

	init_zipf_cache(&zcache, MAXRAND * 1000, alpha, MAXRAND,
					(seeds + low), SEED_UINT32_N);
	for (i = 0; i < size; i++) {
		tmp = sample_zipf_cache(&zcache) + OFFSET;
		assert((tmp >= 20) && (tmp <= 159));
		table[i].len = (unsigned int) tmp;
	}
	end_zipf_cache(&zcache);

	return 0;
}

/* 
 * This is the primary function that facilitates creation of multiple FIBs of
 * table sizes from the range [2^LEXPFIB, 2^HEXPFIB] where the table size grows
 * exponentially.
 *
 * Inputs:
 * tablexp: log2(length of pretable i.e number of entries)
 * seeds: array containing seeds
 * low: lower index in the array for using seeds
 * table: FIB
 * size_seeds: number of seeds in seeds
 * nnexthops: number of unique nexthops in the FIB
 */
int table_dist(int tablexp, uint32_t *seeds, int low, struct nextcreate *table,
		int size_seeds, int nnexthops, double alpha)
{
	int i;
	int size;
	int tmp_low = low;
	gsl_rng *r[2];

	low = low + SEED_UINT32_N;
	for (i = 0; i < 2; i++) {
		r[i] = gsl_rng_alloc(gsl_rng_ranlux);
		assert((low + 2) <= size_seeds);
		gsl_rng_set(r[i], seeds[low + i]);
	}
	size = 1 << tablexp;
	prelength_dist(table, size, seeds, tmp_low, alpha);
	prefix_dist(table, size, r[0]);
	deduplication(table, size, r[0]);
	nexthop_dist(table, size, r[1], nnexthops);

	for (i = 0; i < 2; i++)
		gsl_rng_free(r[i]);

	return 0;
}
