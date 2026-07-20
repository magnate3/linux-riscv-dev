#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "rdist.h"

/* Reference: http://en.wikipedia.org/wiki/Zipf%27s_law */
struct zipf_state {
	double s;
	long n;
	double *cdf;
};

static void init_zipf_cdf(struct zipf_state *state)
{
	long i;
	double hns = 0.0; /* H_n^s */

	/* Add terms from smallest to largest to preserve precision. */
	i = state->n - 1;
	while (i >= 0) {
		hns += 1.0 / pow(i + 1.0, state->s);
		state->cdf[i] = hns;
		i--;
	}

	i = state->n - 1;
	while (i >= 0) {
		state->cdf[i] = 1.0 - (state->cdf[i] / hns);
		i--;
	}

	assert(state->cdf[0] == 0.0);
}

/* For debuging. */
/*
static void print_zipf_cdf(struct zipf_state *state)
{
	long i;
	for (i = 0; i < state->n; i++)
		printf("CDF_%li = %f\n", i + 1, state->cdf[i]);
	printf("\n");
}
*/

/* When s == 0, it's just the uniform distribution. */
static void init_zipf(struct zipf_state *state, double s, long n)
{
	assert(s >= 0.0);
	assert(n >= 1);

	state->s = s;
	state->n = n;

	state->cdf = malloc(sizeof(state->cdf[0]) * n);
	assert(state->cdf);
	init_zipf_cdf(state);
}

static void end_zipf(struct zipf_state *state)
{
	void *p = state->cdf;
	state->cdf = NULL;
	free(p);
}

/* IMPORTANT: z must be in [0, 1) OR [0, 1]. */
static long sample_zipf(struct zipf_state *state, double z)
{
	long a = 0;
	long b = state->n - 1;

	while (b - a >= 2) {
		long c = (a + b) / 2;
		double cdf_c = state->cdf[c];
		/*
		printf("%s: (a, c, b) = (%li, %li, %li)\n", __func__, a, c, b);
		*/
		if (cdf_c <= z)
			a = c;
		else
			b = c;
	}

	/* Deal with (b - a) == 1. */
	if (a < b && state->cdf[b] <= z)
		a = b;

	return a + 1;
}

void init_zipf_cache(struct zipf_cache *cache, long sample_size,
	double s, long n, uint32_t *seeds, int len)
{
	dsfmt_t mt;
	struct zipf_state zipf;
	long i;

	cache->s = s;
	cache->n = n;

	cache->index = 0;
	cache->sample_size = sample_size;

	cache->samples = malloc(sizeof(cache->samples[0]) * sample_size);
	assert(cache->samples);

	dsfmt_init_by_array(&mt, seeds, len);
	init_zipf(&zipf, s, n);
	for (i = 0; i < sample_size; i++) {
		double z = dsfmt_genrand_close_open(&mt);
		cache->samples[i] = sample_zipf(&zipf, z);
	}
	end_zipf(&zipf);
}

void end_zipf_cache(struct zipf_cache *cache)
{
	void *p = cache->samples;
	cache->samples = NULL;
	free(p);
}

long sample_zipf_cache(struct zipf_cache *cache)
{
	long i = cache->index;
	long n = cache->sample_size;
	long r;

	assert(i >= 0);
	assert(i < n);
	r = cache->samples[cache->index++];
	if (cache->index == n)
		cache->index = 0;

	return r;
}

void print_zipf_cache(struct zipf_cache *cache)
{
	long i;

	printf("Zipf cache (parameters s=%f, n=%li)\n", cache->s, cache->n);
	printf("Next index: %li\t entries in the cache: %li\n",
		cache->index, cache->sample_size);

	for (i = 0; i < cache->sample_size; i++) {
		printf("%li\n", cache->samples[i]);
	}
}
