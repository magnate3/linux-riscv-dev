/*
   lc_trie.h

   A routing table for wordsized (32bits) bitstrings implemented as a
   static level- and pathcompressed trie. For details please consult

      Stefan Nilsson and Gunnar Karlsson. Fast Address Look-Up
      for Internet Routers. International Conference of Broadband
      Communications (BC'97).

      http://www.hut.fi/~sni/papers/router/router.html

   The code presented in this file has been tested with care but is
   not guaranteed for any purpose. The writer does not offer any
   warranties nor does he accept any liabilities with respect to
   the code.

   Stefan Nilsson, 4 nov 1997.

   Laboratory of Information Processing Science
   Helsinki University of Technology
   Stefan.Nilsson@hut.fi

   Garnaik Sumeet, Michel Machado 2015
   Modified for LPM Algorithms Testing in Linux-XIA
*/
#ifndef _LC_TRIE_H_
#define _LC_TRIE_H_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include "generate_fibs.h"


#define ADRSIZE 160       /* the number of bits in an address */

typedef struct xid { unsigned char w[20]; } xid;
typedef xid *list;

typedef uint64_t word;
typedef word node_t;

#define NOPRE -1          /* an empty prefix pointer */
#define NOBASE -1

#define GETBRANCH(node)		((node) << 16 >> 56)
#define GETSKIP(node)		((node) << 24 >> 56)
#define GETADR(node)		((node) << 32 >> 32)

/* The routing table entries are initially stored in
   a simple array */

typedef struct entryrec *entry_t;
struct entryrec
{
	xid data;          /* the routing entry */
	int len;            /* and its length */
	unsigned int nexthop;  /* the corresponding next-hop */
	int pre;            /* this auxiliary variable is used in the */
};                    /* construction of the final data structure */

/* base vector */

typedef struct baserec *base_t;
struct baserec
{
	xid str;    /* the routing entry */
	int len;     /* and its length */
	int pre;     /* pointer to prefix table, -1 if no prefix */
	int nexthop; /* pointer to next-hop table */
};

/* compact version of above */
typedef struct
{
	xid str;
	int len;
	int pre;
	int nexthop;
} comp_base_t;

/* prefix vector */

typedef struct prerec *pre_t;
struct prerec
{
	int len;     /* the length of the prefix */
	int pre;     /* pointer to prefix, -1 if no prefix */
	int nexthop; /* pointer to nexthop table */
};

/* compact version of above */
typedef struct
{
	int len;
	int pre;
	int nexthop;
} comp_pre_t;

/* The complete routing table data structure consists of
   a trie, a base vector, a prefix vector, and a next-hop table. */

typedef struct routtablerec *routtable_t;
struct routtablerec
{
	node_t *trie;         /* the main trie search structure */
	int triesize;
	comp_base_t *base;    /* the base vector */
	int basesize;
	comp_pre_t *pre;      /* the prefix vector */
	int presize;
	unsigned int *nexthop;   /* the next-hop table */
	int nexthopsize;
};

/*
 * The structure of each node for building the path compressed trie from XIDs
 */
typedef struct node_patric node_patric;
struct node_patric
{
	int skip;	//The number of bits skipped
	node_patric *left;	//Pointer to left node in Patricia Trie
	node_patric *right;	//Pointer to right node in Patricia Trie
	int base;	//The address of the XID in base vector
};

/* The structure of each node for building the level compressed trie */
typedef struct node_lc node_lc;
struct node_lc
{
	int skip;	//The number of bits skipped
	node_lc **subtrie;	//Array of pointers to nodes in the trie
	int base;	//The address of the XID in base vector
	int branch;	//Branching factor i.e number of children = 2^branch
	int addr;	//Address of the node in array representation
	int child;	//Address of the left most child node in array representation
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
routtable_t buildrouttable(entry_t entry[], int nentries);

struct routtablerec *lctrie_create_fib(struct nextcreate *table,
		unsigned long size);

int lctrie_destroy_fib(struct routtablerec *rtable);

unsigned int lctrie_lookup(const xid *id, routtable_t table);

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
unsigned int find(const xid *s, routtable_t t);

#endif // _LC_TRIE_H_
