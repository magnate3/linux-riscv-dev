/*

   Original Code:
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
#include "lc_trie.h"


/* Return a nexthop or 0 if not found */
unsigned int find(const xid *s, routtable_t t)
{
	int i,val;
	node_t node;
	unsigned char pos, branch;
	uint32_t adr, next_jump;
	xid *bitmask = malloc(sizeof(xid));
	xid *tmp = malloc(sizeof(xid));
	xid tmp_node;
	int preadr;

	/* Traverse the trie */
	node = t->trie[0];
	pos = (unsigned char) GETSKIP(node);
	branch = (unsigned char) GETBRANCH(node);
	adr = (uint32_t) GETADR(node);
	while (branch != 0) {
		tmp_node = extract(pos, branch, *s);
		next_jump = xidtounsigned(&tmp_node);
		node = t->trie[adr + next_jump];
		pos += branch + GETSKIP(node);
		branch = (unsigned char) GETBRANCH(node);
		adr = (unsigned char) GETADR(node);
	}

	/* Was this a hit? */
	for (i = 0; i < 20; i++)
		bitmask->w[i] = (t->base[adr].str).w[i] ^ s->w[i];
	*bitmask = extract(0, t->base[adr].len, *bitmask);
	memset(tmp, 0, 20);
	val = comparexid(&bitmask, &tmp);
	if (0 == val) {
		free(bitmask);
		free(tmp);
		return t->nexthop[t->base[adr].nexthop];
	}

	/* If not, look in the prefix tree */
	preadr = t->base[adr].pre;
	while (preadr != NOPRE) {
		*bitmask = extract(0, t->pre[preadr].len, *bitmask);
		val = comparexid(&bitmask, &tmp);
		if (0 == val) {
			free(bitmask);
			free(tmp);
			return t->nexthop[t->pre[preadr].nexthop];
		}
		preadr = t->pre[preadr].pre;
	}

	return 0; //Not found
}
