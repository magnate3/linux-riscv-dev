#include "generate_fibs.h"
#include "lpm_bloom.h"
#include "lpm_radix.h"
#include <fcntl.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "rdist.h"

#define LEXPFIB 4
#define HEXPFIB 20
#define NLOOKUPS 1000000
#define RUNS 20
#define BLOOMERRORRATE 0.05
#define MINNEXTHOPS 16
#define MAXNEXTHOPS 256
#define NEXTHOPJUMP 16
#define LOOPSEED ((RUNS) * ((NEXTSEED) + SEED_UINT32_N))
#define LOOKUPFILEBLOOM "bloom_lookup_measurements"
#define LOOKUPFILERADIX "radix_lookup_measurements"
#define NEXTHOPSFILERADIX "radix_nexthops_measurements"
#define NEXTHOPSFILEBLOOM "bloom_nexthops_measurements"


static unsigned long sampleindex(struct zipf_cache *zcache)
{
	return sample_zipf_cache(zcache);
}

static inline unsigned long gettime(const struct timespec *x,
		const struct timespec *y)
{
	unsigned long tmp;
	assert((tmp =
	(y->tv_sec - x->tv_sec) * 1000000000L + (y->tv_nsec - x->tv_nsec)) > 0);
	return tmp;
}

static int time_measure(struct timespec *ntime)
{
	if(clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ntime) == -1) {
		perror("clock gettime");
		exit(EXIT_FAILURE);
	}

	return 0;
}

static int evaluate_nexthops_radix(const void *t, const void *ts,
		const void *nh, const void *s, const void *al)
{
	struct timespec start, stop;
	struct nextcreate *table = (struct nextcreate *) t;
	unsigned long *size = (unsigned long *) ts;
	struct nextcreate *tmp_table = malloc(sizeof(struct nextcreate) * *size);
	memcpy(tmp_table, table, sizeof(struct nextcreate) * *size);
	uint32_t *seed = (uint32_t *) s;
	int *nnexthops = (int *) nh;
	double *alpha = (double *) al;
	unsigned long tmp;
	FILE *fp = NULL;
	unsigned long accum = 0;
	int i;
	xid *id = NULL;
	struct zipf_cache zcache;

	init_zipf_cache(&zcache, *size * 30, *alpha, *size, seed, SEED_UINT32_N);
	setpriority(PRIO_PROCESS, 0, -20);
	// Create the data structure
	struct routtablerec *fib = radix_create_fib(table, *size);
	for (i = 0; i < NLOOKUPS; i++) {
		tmp = sampleindex(&zcache);
		id = (xid *) &(tmp_table[tmp].prefix);
		// Sample from table and set the XID in appropriate form
		time_measure(&start);
		lookup_radix(id, fib, 0);
		time_measure(&stop);
		accum += gettime(&start, &stop);
	}
	end_zipf_cache(&zcache);
	fp = fopen(NEXTHOPSFILERADIX, "a");
	fprintf(fp, "%d\t%lu\n", *nnexthops, accum);
	fclose(fp);
	radix_destroy_fib(fib);
	free(tmp_table);
	return 0;
}

static int evaluate_nexthops_bloom(const void *t, const void *ts,
		const void *nh, const void *s, const void *al)
{
	struct timespec start, stop;
	struct nextcreate *table = (struct nextcreate *) t;
	unsigned long *size = (unsigned long *) ts;
	uint32_t *seed = (uint32_t *) s;
	int *nnexthops = (int *) nh;
	double *alpha = (double *) al;
	int i;
	unsigned long tmp;
	FILE *fp = NULL;
	double error_rate = BLOOMERRORRATE;
	unsigned char (*id)[HEXXID] = calloc(HEXXID, sizeof(unsigned char));
	unsigned int len;
	unsigned long accum = 0;
	int val;
	struct zipf_cache zcache;

	init_zipf_cache(&zcache, *size * 30, *alpha, *size, seed, SEED_UINT32_N);
	setpriority(PRIO_PROCESS, 0, -20);
	// Create the data structure
	struct bloom_structure *filter = bloom_create_fib(table, *size, error_rate);
	for (i = 0; i < NLOOKUPS; i++) {
		tmp = sampleindex(&zcache);
		memcpy(id, table[tmp].prefix, HEXXID);
		len = table[tmp].len;
		time_measure(&start);
		lookup_bloom(&id[0], len, filter);
		time_measure(&stop);
		accum += gettime(&start, &stop);
	}
	free(id);
	end_zipf_cache(&zcache);
	fp = fopen(NEXTHOPSFILEBLOOM, "a");
	fprintf(fp, "%d\t%lu\n", *nnexthops, accum);
	fclose(fp);
	bloom_destroy_fib(filter);
	return 0;
}

static int nexthops_experiments(int exp, uint32_t *seeds, int low,
		int seedsize, double alpha)
{
	int i, j, k;
	pid_t id;
	unsigned long size = 1 << exp;
	struct nextcreate *table = NULL;
	int nnexthops;
	int o_seed = low;
	int (*experiments[])(const void *, const void *, const void *,
					const void *, const void *) = {
		evaluate_nexthops_bloom,
		evaluate_nexthops_radix,
		NULL,
	};

	for (i = 0; experiments[i] != NULL;  i++) {
		low = o_seed;
		for (j = MINNEXTHOPS; j <= MAXNEXTHOPS; j += NEXTHOPJUMP) {
			nnexthops = j;
			for (k = 0; k < RUNS; k++) {
				id = fork();
				assert(id >= 0);
				if (0 == id) {
					table = malloc(sizeof(struct nextcreate)
							* size);
					assert(0 == table_dist(exp, seeds, low, table, seedsize, nnexthops, alpha));
					low = low + NEXTSEED;
					assert(low < seedsize);
					assert(0 == (experiments[i])
					(table, &size, &nnexthops, &seeds[low], &alpha));
					free(table);
					exit(EXIT_SUCCESS);
				} else {
					assert(wait(NULL) >= 0);
					low = low + NEXTSEED + SEED_UINT32_N;
					assert(low < seedsize);
					if (0 == i)
						printf("Done lookup experiments bloom 2^%d with nexthops: %d run: %d\n", exp, j, k);
					else
						printf("Done lookup experiments radix 2^%d with nexthops: %d run: %d\n", exp, j, k);
				}
			}
		}
	}
	return 0;
}

static int evaluate_lookups_radix(const void *t, const void *ts,
		const void *s, const void *al)
{
	struct timespec start, stop;
	struct nextcreate *table = (struct nextcreate *) t;
	unsigned long *size = (unsigned long *) ts;
	struct nextcreate *tmp_table = malloc(sizeof(struct nextcreate) * *size);
	memcpy(tmp_table, table, sizeof(struct nextcreate) * *size);
	uint32_t *seed = (uint32_t *) s;
	double *alpha = (double *) al;
	FILE *fp = NULL;
	unsigned long tmp;
	unsigned long accum = 0;
	int i;
	xid *id = NULL;
	struct zipf_cache zcache;
	int wrong = 0;

	init_zipf_cache(&zcache, *size * 30, *alpha, *size, seed, SEED_UINT32_N);
	setpriority(PRIO_PROCESS, 0, -20);
	// Create the data structure
	struct routtablerec *fib = radix_create_fib(table, *size);
	for (i = 0; i < NLOOKUPS; i++) {
		tmp = sampleindex(&zcache) % *size;
		id = (xid *) &(tmp_table[tmp].prefix);
		// Sample from table and set the XID in appropriate form
		time_measure(&start);
		lookup_radix(id, fib, 0);
		time_measure(&stop);
		accum += gettime(&start, &stop);
	}
	end_zipf_cache(&zcache);
	fp = fopen(LOOKUPFILERADIX, "a");
	fprintf(fp, "%lu\t%lu\n", *size, accum);
	fclose(fp);
	radix_destroy_fib(fib);
	free(tmp_table);
	return 0;
}

static int evaluate_lookups_bloom(const void *t, const void *ts,
		const void *s, const void *al)
{
	struct timespec start, stop;
	struct nextcreate *table = (struct nextcreate *) t;
	unsigned long *size = (unsigned long *) ts;
	uint32_t *seed = (uint32_t *) s;
	double *alpha = (double *) al;
	int i;
	unsigned long tmp;
	FILE *fp = NULL;
	double error_rate = BLOOMERRORRATE;
	unsigned char (*id)[HEXXID] = calloc(HEXXID, sizeof(unsigned char));
	unsigned int len;
	unsigned long accum = 0;
	struct zipf_cache zcache;
	int wrong = 0;

	init_zipf_cache(&zcache, *size * 30, *alpha, *size, seed, SEED_UINT32_N);
	setpriority(PRIO_PROCESS, 0, -20);
	// Create the data structure
	struct bloom_structure *filter = bloom_create_fib(table, *size, error_rate);
	for (i = 0; i < NLOOKUPS; i++) {
		tmp = sampleindex(&zcache) % *size;
		memcpy(id, table[tmp].prefix, HEXXID);
		len = table[tmp].len;
		time_measure(&start);
		lookup_bloom(&id[0], len, filter);
		time_measure(&stop);
		accum += gettime(&start, &stop);
	}
	free(id);
	end_zipf_cache(&zcache);
	fp = fopen(LOOKUPFILEBLOOM, "a");
	fprintf(fp, "%lu\t%lu\n", *size, accum);
	fclose(fp);
	bloom_destroy_fib(filter);
	return 0;
}

static int lookup_experiments(int exp, uint32_t *seeds, int low, int seedsize,
		int nnexthops, double alpha)
{
	int i, j;
	pid_t id;
	unsigned long size = 1 << exp;
	struct nextcreate *table = NULL;
	int o_seed = low;
	int (*experiments[])(const void *, const void *, const void *,
						const void *) = {
		evaluate_lookups_bloom,
		evaluate_lookups_radix,
		NULL,
	};

	for (i = 0; experiments[i] != NULL;  i++) {
		low = o_seed;
		for (j = 0; j < RUNS; j++) {
			id = fork();
			assert(id >= 0);
			if (0 == id) {
				table = malloc(sizeof(struct nextcreate) *
						size);
				assert(0 == table_dist(exp, seeds, low, table, seedsize, nnexthops, alpha));
				low = low + NEXTSEED;
				assert(low < seedsize);
				assert(0 == (experiments[i])
					(table, &size, &seeds[low], &alpha));
				free(table);
				exit(EXIT_SUCCESS);
			} else {
				assert(wait(NULL) >= 0);
				low = low + NEXTSEED + SEED_UINT32_N;
				assert(low < seedsize);
				if (0 == i)
					printf("Done lookup experiments bloom 2^%d with run: %d\n", exp, j);
				else
					printf("Done lookup experiments radix 2^%d with run: %d\n", exp, j);
			}
		}
	}
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
	int nnexthops = 0;
	double alpha = 0.0;
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
		nnexthops = 16;
		alpha = 1.0;
		assert(0 == lookup_experiments(i, seeds, low, seedsize,
					nnexthops, alpha));
		printf("Done lookup experiment 2^%d\n", i);
		assert(0 ==
			nexthops_experiments(i, seeds, low, seedsize, alpha));
		printf("Done nexthop experiment 2^%d\n", i);
		low = low + LOOPSEED;
		assert(low < seedsize);
	}

	return 0;
}
