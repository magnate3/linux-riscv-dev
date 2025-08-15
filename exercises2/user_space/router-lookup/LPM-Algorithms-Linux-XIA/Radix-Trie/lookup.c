/*
 * Garnaik Sumeet, Michel Machado 2015
 * LPM Algorithms for Linux-XIA
 */
#include "radix.h"
#include <inttypes.h>

/* 
 * Return a nexthop or 0 if not found
 */
unsigned int find(xid s, struct routtablerec *t, int opt)
{
	struct node_patric *cur_node = t->root;
	assert(cur_node);
	xid *bitmask = alloca(sizeof(xid));
	int pos = 0;
	int i, j;

	pos = cur_node->skip;
	while (NOBASE == cur_node->base) {
		*bitmask = extract(pos, 1, s);
		pos++;
		if (bitmask->w[HEXXID - 1])
			cur_node = cur_node->right;
		else
			cur_node = cur_node->left;
		pos += cur_node->skip;
	}

	// Check if actually a hit
	int bidx = cur_node->base;
	memset(bitmask, 0, HEXXID);
	xid *tmp = alloca(sizeof(xid));
	for (i = 0; i < HEXXID; i++)
		tmp->w[i] = t->base[bidx].str.w[i] ^ s.w[i];
	int neq = comparexid(&bitmask, &tmp);
	if (!neq) {
		return t->base[bidx].nexthop;
	}

	// Check in prefix tree
	int pidx = t->base[bidx].pre;
	int len;
	if (opt) {
		for (i = 0; i < 20; i++)
			printf("%02x", t->base[bidx].str.w[i]);
	}
	while (NOPRE != pidx) {
		len = t->pre[pidx].len;
		memset(tmp, 0, HEXXID);
		for (i = 0; i <= len / BYTE; i++)
			tmp->w[i] = t->base[bidx].str.w[i] ^ s.w[i];
		tmp->w[i - 1] = (tmp->w[i - 1] >> (BYTE - len % BYTE)) << (BYTE - len % BYTE);
		for (j = i; j < 20; j++)
			tmp->w[j] = 0;
		neq = comparexid(&bitmask, &tmp);
		if (!neq) {
			if (opt)
				printf("\tLen: %d\tNexthop: %d\n", t->pre[pidx].len, t->pre[pidx].nexthop);
			return t->pre[pidx].nexthop;
		} else {
			pidx = t->pre[pidx].pre;
		}
	}

	// Not found
	return 0;
}
