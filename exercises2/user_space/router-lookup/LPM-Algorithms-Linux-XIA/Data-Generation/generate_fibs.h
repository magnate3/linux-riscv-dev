#ifndef _GENERATE_FIBS_H_
#define _GENERATE_FIBS_H_

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <search.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "../rdist.h"

#undef TRUE
#define TRUE 1

#undef FALSE
#define FALSE 0

#define SEED_UINT32_N 10
#define NEXTSEED ((SEED_UINT32_N) + 2)
#define MAXFILENAME 10 // Maximum lenght of file name
#define OFFSET 19 // Offset for the random length generated
#define MAXRAND 140 // Bound for random length generation
#define HEXXID 20 // The number of chars that comprise of a prefix
#define SEEDS 150000 // Size of array containing seeds
#define SEEDFILE "seeds" // Name of the file containing seeds
#define BITRAND 2 // Bound for random bit generation
#define BYTE 8
// This is simply an observation of the actual tables created although this
// number is huge considering uniform distribution
#define DUPS 500000

/* 
 * This is a structure that consists of one entry in the table generated.
 * The name of the structure is a misnomer.
 * Initially it was planned to be a structure to generate nexthop entry
 */
struct nextcreate {
	unsigned char prefix[HEXXID];
	unsigned int len;
	unsigned int nexthop;
};

int table_dist(int tablexp, uint32_t *seeds, int low, struct nextcreate *table,
		int size_seeds, int nnexthops, double alpha);
int print_table(struct nextcreate *table, int size, int prexp);

#endif
