/*
 * Garnaik Sumeet, Michel Machado 2015
 * LPM Algorithm for Linux-XIA
*/
#ifndef _RADIX_H_
#define _RADIX_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "../Data-Generation/generate_fibs.h"

#undef BYTE
#define BYTE 8

#define ADRSIZE 160 /* the number of bits in an address */

typedef struct xid {unsigned char w[20];} xid;
typedef xid *list;

#define NOPRE -1          /* an empty prefix pointer */
#define NOBASE -1

struct tmp_prerec
{
	xid str;
	int pre;
	int len;
	unsigned int nexthop;
};

/* The routing table entries are initially stored in
   a simple array */
struct entryrec
{
	xid data;          /* the routing entry */
	int len;            /* and its length */
	unsigned int nexthop;  /* the corresponding next-hop */
	int pre;            /* this auxiliary variable is used in the */
};                    /* construction of the final data structure */

/* base vector */
struct baserec
{
	xid str;    /* the routing entry */
	int len;     /* and its length */
	int pre;     /* pointer to prefix table, -1 if no prefix */
	int nexthop;
};

/* prefix vector */
struct prerec
{
	int len;     /* the length of the prefix */
	int pre;     /* pointer to prefix, -1 if no prefix */
	int nexthop; /* pointer to nexthop table */
};

/*
 * The complete routing table data structure consists of a trie, a base vector,
 * a prefix vector, and a next-hop table.
 */
struct routtablerec
{
	struct node_patric *root;
	struct baserec *base;    /* the base vector */
	int basesize;
	struct prerec *pre;      /* the prefix vector */
	int presize;
	unsigned int *nexthop;   /* the next-hop table */
	int nexthopsize;
};

/*
 * The structure of each node for building the path compressed trie from XIDs
 */
struct node_patric
{
	int skip;
	struct node_patric *left;
	struct node_patric *right;
	int base;
};

/* Extract operation for xids*/
xid extract(int pos, int length, xid data);

/* Remove bits from xids*/
xid removexid(int bits, xid data);

/* Increment xids*/
int incrementxid(xid *pxid);

/* Bitwise shifting operations on xids*/
xid shift_left(xid id, int shift);
xid shift_right(xid id, int shift);

/* Build the routing table */
struct routtablerec *buildrouttable(struct entryrec *entry[], int nentries);

struct routtablerec *radixtrie_create_fib(struct nextcreate *table,
		unsigned long size);

int radixtrie_destroy_fib(struct routtablerec *rtable);

/* 
 * Extract the 32 LSBs and cast to uint32_t
 */
uint32_t xidtounsigned(xid *bitpat);

/* 
 * Compare XIDs and return 0, 1, -1 depending on the values pointed by
 * dereferencing id1 and id2 twice.
 */
int comparexid(const void *id1, const void *id2);

/* Perform a lookup. */
unsigned int find(xid s, struct routtablerec *t, int opt);

#endif
