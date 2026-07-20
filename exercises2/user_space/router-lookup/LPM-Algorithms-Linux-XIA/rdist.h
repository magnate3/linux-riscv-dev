#ifndef _RDIST_H
#define _RDIST_H

#include <stdint.h>

#include "./dSFMT-src-2.2.1/dSFMT.h"

struct unif_state {
	dsfmt_t state;
};

static inline void init_unif(struct unif_state *unif, uint32_t *seeds, int len)
{
	dsfmt_init_by_array(&unif->state, seeds, len);
}

static inline void end_unif(struct unif_state *unif)
{
	/* Empty. */
}

/* Return a random aumber in [0..(n - 1)]. */
static inline long sample_unif_0_n1(struct unif_state *unif, long n)
{
	return (long)(dsfmt_genrand_close_open(&unif->state) * (double)n);
}

/* Return a random aumber in [1..n]. */
static inline long sample_unif_1_n(struct unif_state *unif, long n)
{
	return (long)(dsfmt_genrand_close_open(&unif->state) * (double)n) + 1;
}

/* Return a random aumber in [0..n]. */
static inline long sample_unif_0_n(struct unif_state *unif, long n)
{
	return (long)(dsfmt_genrand_close_open(&unif->state) * (n + 1.0));
}

struct zipf_cache {
	double s;
	long n;
	long index;
	long sample_size;
	long *samples;
};

void init_zipf_cache(struct zipf_cache *cache, long sample_size,
	double s, long n, uint32_t *seeds, int len);
void end_zipf_cache(struct zipf_cache *cache);
long sample_zipf_cache(struct zipf_cache *cache);
void print_zipf_cache(struct zipf_cache *cache);

#endif	/* _RDIST_H */
