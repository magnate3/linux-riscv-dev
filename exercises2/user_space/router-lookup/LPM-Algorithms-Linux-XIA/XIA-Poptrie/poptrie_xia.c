#include "poptrie_xia.h"

/*
 * Converts an xid to corresponding integer value. Assumes that the number of
 * bits and positions of initial bit are all valid. Any non-valid arguments will
 * result in arbitrary behaviour
 *
 * NOTE: Non-Optimized. Feel free to optimize the way you want
 */
static uint32_t _xid_to_u32(xid addr, int start, int bits)
{
	xid tmp = extractXID(start, bits, addr);
	uint32_t utmp = XIDtounsigned(&tmp);
	return utmp;
}

/*
 * Looks up the longest prefix matching entry in Poptrie and returns the
 * corresponding nexthop address
 */
int poptrie_xia_lookup(struct poptrie *poptrie, xid prefix, uint16_t *nexthop)
{
	uint32_t index;
	int base;
	int pos;
	int inode;

	index = _xid_to_u32(prefix, 0, POPTRIE_DP);
	pos = POPTRIE_DP;

	if (poptrie->direct_pointer[index] & ((uint32_t) 1 << 31)) {
		*nexthop = (poptrie->fib).entries[(~((uint32_t) 1 << 31) & (poptrie->direct_pointer[index]))];
		return PASSED;
	} else {
		base = poptrie->direct_pointer[index];
		index = _xid_to_u32(prefix, pos, M_ARY);
		pos += M_ARY;
	}
	// Needs to traverse through Poptrie

	// Never to be reached here since we return the control while looping
	// through Poptrie
	return FAILED;
}

/*
 * Adds a new entry into the Poptrie data structure. This is the interface to
 * add new routes for application programs
 */
int poptrie_xia_add(struct poptrie *poptrie, xid prefix, unsigned int len,
		unsigned int nexthop)
{
	int i;
	uint16_t next = 0;
	uint32_t val = 0;

	for (i = 0; i < (poptrie->fib).n; i++) {
		// Found the nexthop in the fib
		if (nexthop == (poptrie->fib).entries[i]) {
			next = i;
			break;
		}
	}

	if (i == (poptrie->fib).n) {
		(poptrie->fib).entries[(poptrie->fib).n] = nexthop;
		next = (poptrie->fib).n;
		assert(++((poptrie->fib).n) < (poptrie->fib).size);
	}

	if (len < POPTRIE_DP) {
		// Insert into direct pointer without need to set the proper
		// bits for accessing nodes below the trie
		val = _xid_to_u32(prefix, 0, POPTRIE_DP);
		*(poptrie->direct_pointer + val) = next;
		SET_BIT_DP((poptrie->direct_pointer + val));
	} else if (POPTRIE_DP == len) {
		// Insert into direct pointer and set the right bits for
		// fib entries. Setting them different for alternate direct
		// pointer in further revisions
		val = _xid_to_u32(prefix, 0, len);
		*(poptrie->direct_pointer + val) = next;
		SET_BIT_DP((poptrie->direct_pointer + val));
	} else {
		// Insert into direct pointer the address of internal nodes
		// The case of leafvec shall be added in further revisions
	}
	return PASSED;
}

/*
 * Destroy the entire poptrie data structure
 */
int poptrie_xia_destroy(struct poptrie *poptrie)
{
	// Return Failed when no such data structure is passed
	if (NULL == poptrie) {
		return FAILED;
	} else {
		// Check for existence of required memory allocations and free
		if (NULL != poptrie->direct_pointer)
			free(poptrie->direct_pointer);
		if (NULL != poptrie->nodes)
			free(poptrie->nodes);
		if (NULL != poptrie->leaves)
			free(poptrie->leaves);
		if (NULL != (poptrie->fib).entries)
			free((poptrie->fib).entries);
		free(poptrie);
	}

	return PASSED;
}

/*
 * Initialize all the required memory for building the data structure
 */
struct poptrie *poptrie_xia_init(struct poptrie *poptrie, int nodes,
		int leaves)
{
	// Check if new allocation
	if (NULL == poptrie) {
		assert(NULL != (poptrie = calloc(1, sizeof(struct poptrie))));
	} else {
		// Resetting of old poptrie data structure and checking for
		// freeing all the memory
		if (NULL != poptrie->direct_pointer)
			free(poptrie->direct_pointer);
		if (NULL != poptrie->nodes)
			free(poptrie->nodes);
		if (NULL != poptrie->leaves)
			free(poptrie->leaves);
		if (NULL != (poptrie->fib).entries)
			free((poptrie->fib).entries);
		memset(poptrie, 0, sizeof(struct poptrie));
	}

	// Allocating the required memory
	assert(NULL != (poptrie->direct_pointer = calloc(1 << POPTRIE_DP, sizeof(uint32_t))));
	assert(NULL != ((poptrie->fib).entries = calloc(POPTRIE_INIT_FIB_SIZE, sizeof(uint16_t))));
	(poptrie->fib).size = POPTRIE_INIT_FIB_SIZE;
	(poptrie->fib).n++;
	assert(NULL != (poptrie->nodes = calloc(1 << nodes, sizeof(struct poptrie_node))));
	assert(NULL != (poptrie->leaves = calloc(1 << leaves, sizeof(uint16_t))));
	return poptrie;
}
