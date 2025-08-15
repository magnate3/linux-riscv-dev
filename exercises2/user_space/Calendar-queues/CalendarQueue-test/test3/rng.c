#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <stdbool.h>
#include <inttypes.h>
#include <time.h>

#include "rng.h"
#ifdef TARGET_X86

#define CLOCK_READ() ({ \
			unsigned int lo; \
			unsigned int hi; \
			__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi)); \
			((unsigned long long)hi) << 32 | lo; \
			})

void Srand(long *idum) {
	*idum = (long)CLOCK_READ();
}
#elif TARGET_ARM64
/** Read generic counter */
static inline uint64_t arm64_cntvct(void) {
    uint64_t tsc;

    asm volatile("mrs %0, cntvct_el0" : "=r"(tsc));
    return tsc;
}
static inline uint64_t rte_rdtsc_pmccntr(void){
	unsigned tsc;
	uint64_t final_tsc;

	/* Read PMCCNTR */
	asm volatile("mrc p15, 0, %0, c9, c13, 0" : "=r"(tsc));
	/* 1 tick = 64 clocks */
	final_tsc = ((uint64_t)tsc) << 6;

	return (uint64_t)final_tsc;
}

/** Read PMU cycle counter */
static inline uint64_t __arm64_pmccntr(void) {
    uint64_t tsc;

    asm volatile("mrs %0, pmccntr_el0" : "=r"(tsc));
    return tsc;
}

static inline uint64_t rdtsc(void) {
    return __arm64_pmccntr();
}

void Srand(long *idum) {
	*idum = (long)arm64_cntvct();
	//*idum = (long)rdtsc();
}

#endif
float Random(long *idum) {
	long k;
	float ans;

	*idum ^= MASK;
	k = (*idum) / IQ;
	*idum = IA * (*idum - k * IQ) - IR * k;
	if (*idum < 0) *idum += IM;
	ans = AM * (*idum);
	*idum ^= MASK;

	return ans;
}


double Expent(long *idum, double mean) {

	if(mean < 0) {
		fprintf(stderr, "Error in call to Expent(): passed a negative mean value\n");
		abort();
	}

	return (-mean * log(1 - Random(idum)));
}
