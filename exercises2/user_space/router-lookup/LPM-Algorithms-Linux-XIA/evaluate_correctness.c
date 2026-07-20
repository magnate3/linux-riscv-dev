#include "generate_fibs.h"
#include "lpm_bloom.h"
#include "lpm_radix.h"
#include <fcntl.h>

#define LEXPFIB 4
#define HEXPFIB 20
#define BLOOMERRORRATE 0.05

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

static int correctness_experiment(int exp, uint32_t *seeds, int low,
							int seedsize)
{
	int i;
	unsigned long size = 1 << exp;
	struct nextcreate *table = NULL;
	struct nextcreate *tmp_table = NULL;
	int nnexthops = 16;
	unsigned int len;
	int nexthops[3] = {0};
	xid *id1 = alloca(sizeof(xid));
	unsigned char (*id2)[HEXXID] = calloc(HEXXID, sizeof(unsigned char));

	table = malloc(sizeof(struct nextcreate) * size);
	tmp_table = malloc(sizeof(struct nextcreate) * size);
	assert(0 == table_dist(exp, seeds, low, table, seedsize, nnexthops, 1.0));
	memcpy(tmp_table, table, sizeof(struct nextcreate) * size);
	// Create bloom
	struct bloom_structure *filter = bloom_create_fib(table, size, BLOOMERRORRATE);
	// Create radix
	// Radix changes original data hence at the end
	struct routtablerec *fib = radix_create_fib(table, size);
	for (i = 0; i < size; i++) {
		memcpy(id1, tmp_table[i].prefix, HEXXID);
		memcpy(id2, tmp_table[i].prefix, HEXXID);
		len = tmp_table[i].len;
		nexthops[0] = tmp_table[i].nexthop;
		nexthops[1] = lookup_radix(id1, fib, 0);	// radix lookup
		nexthops[2] = lookup_bloom(&id2[0], len, filter);// bloom lookup
		assert((nexthops[0] == nexthops[1]) && (nexthops[1] == nexthops[2]));
	}
	free(id2);
	free(tmp_table);
	free(table);

	return 0;
}

int main(int argc, char *argv[])
{
	int i;
	unsigned long size;
	FILE *seed_file = NULL;
	uint32_t *seeds = NULL;
	int seedsize = 0;
	int low = 0;
	int val;

	assert(0 == access(SEEDFILE, R_OK | F_OK));
	seeds = malloc(SEEDS * sizeof(uint32_t));
	seed_file = fopen(SEEDFILE, "r");
	assert(seed_file);
	i = 0;
	while (fscanf(seed_file, "%8x\n", &seeds[i]) != EOF)
		assert(i++ < SEEDS);
	seedsize = i;
	assert(0 == fclose(seed_file));

	for (i = LEXPFIB; i <= HEXPFIB; i++) {
		size = 1 << i;
		// This is taken as a constant for the number of 
		assert(0 == correctness_experiment(i, seeds, low, seedsize));
		printf("Done 2^%d\n", i);
		low = low + NEXTSEED;
		assert(low < seedsize);
	}

	return 0;
}
