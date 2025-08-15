#ifndef _POPTRIE_XIA_H
#define _POPTRIE_XIA_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <assert.h>

#define PASSED 0
#define FAILED -1

#define POPTRIE_DP 18
#define POPTRIE_INIT_FIB_SIZE 1024
#define M_ARY 6

#define SET_BIT_DP(dp) (*(dp) = *(dp) | (1 << 31))
#define popcnt(v) __builtin_popcountll(v)

typedef struct xid {
	unsigned char w[20];
} xid;

struct poptrie_fib {
	uint16_t *entries;
	int n;
	int size;
};

struct poptrie_node {
	// Leafvec to be added in further revisions
	uint64_t vector;
	uint32_t base1;
	uint32_t base0;
};

struct poptrie {
	// Direct Pointer
	uint32_t *direct_pointer;
	// Will add alternate direct pointer and routines for lockfree update in
	// further revisions

	// Nodes and Leaves
	struct poptrie_node *nodes;
	uint16_t *leaves;

	// FIB
	struct poptrie_fib fib;
};

/*
 * In poptrie_xia.c
 */
//int poptrie_xia_create();
int poptrie_xia_add(struct poptrie *poptrie, xid prefix, unsigned int len,
		unsigned int nexthop);
int poptrie_xia_lookup(struct poptrie *poptrie, xid prefix, uint16_t *nexthop);
// Currently keeping the structure to be a constant with 18-bit direct pointing
struct poptrie *poptrie_xia_init(struct poptrie *poptrie, int nodes,
		int leaves);
int poptrie_xia_destroy(struct poptrie *poptrie);

/*
 * In xid_operation.c
 */
xid extractXID(int pos, int length, xid data);
xid removeXID(int bits, xid data);
xid shift_left(xid id, int shift);
xid shift_right(xid id, int shift);
uint32_t XIDtounsigned(xid *bitpat);

#endif
