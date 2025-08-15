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
#include <stddef.h>
#include "xidsort.h"
#include "lc_trie.h"

/*
 * Compare two routing table entries.
 */
int compareentries(const void *id1, const void *id2)
{
	int tmpresult;
	entry_t *tmp_entry1 = (entry_t *) id1;
	entry_t *tmp_entry2 = (entry_t *) id2;
	xid *tmp_id1 = &((*tmp_entry1)->data);
	xid *tmp_id2 = &((*tmp_entry2)->data);

	tmpresult = comparexid(&tmp_id1, &tmp_id2);

	if (tmpresult < 0)
		return -1;
	else if (tmpresult > 0)
		return 1;
	else
		return 0;
}
/*
static int count_pat_nodes(node_patric *n)
{
	if (NULL == n->left) {
		free(n);
		return 1;
	} else {
		int i = count_pat_nodes(n->left) + count_pat_nodes(n->right) + 1;
		free(n);
		return i;
	}
}
*/
/*
 * Compute the number of bits to be skipped for a range of XID addresses.
 *
 * base: base vector
 * prefix: prefix common to the XIDs starting at position first and having
 * total n such XIDs
 */
static int skipcompute(base_t base[], int prefix, int first, int n)
{
	int i, match;
	xid low, high, *plow, *phigh;

	// Compute the skip value for that node
	low = removexid(prefix, base[first]->str);
	high = removexid(prefix, base[first + n - 1]->str);
	i = prefix;
	plow = malloc(sizeof(xid));
	phigh = malloc(sizeof(xid));
	*plow = extract(i, 1, low);
	*phigh = extract(i, 1, high);
	match = comparexid(&plow, &phigh);
	while (0 == match) {
		i++;
		*plow = extract(i, 1, low);
		*phigh = extract(i, 1, high);
		match = comparexid(&plow, &phigh);
	}
	free(plow);
	free(phigh);
	return i;
}

/*
 * Computes the total number of base XIDs for a given prefix
 * 
 * base: base vector
 * prefix: number of bits that are common to the callee node
 * first: starting address of the base for callee node
 * n: number of addresses of the base for the callee node
 */
static int subtriecompute(base_t base[], int prefix, int first, int n)
{
	xid tmp;
	int nleft = 0;
	unsigned char left = (unsigned char) 0;

	// Find the number of prefixes in base that belong to left subtrie
	tmp = extract(prefix, 1, base[first]->str);
	while (left == tmp.w[HEXXID - 1]) {
		nleft++;
		tmp = extract(prefix, 1, base[first + nleft]->str);
	}
	return nleft;
}

/*
 * This routine builds the Patricia trie recursively for all nodes in the range
 * of address "first" to "first + n -1" in base vector: base passed as arg
 *
 * base: base vector
 * prefix: number of bits that are common to all the XIDs in the range
 * "first" to "first + n - 1"
 * first: first index of base vector
 * n: number of prefixes in base vector starting at index first 
 */
static node_patric *buildpatricia(base_t base[], int prefix, int first, int n)
{
	int i,newprefix, nleft;
	node_patric *node;

	// Reached the leaf node
	if (1 == n) {
		node = malloc(sizeof(node_patric));
		node->left = NULL;
		node->right = NULL;
		node->skip = base[first]->len - prefix;
		node->base = first;
		return node;
	}

	//Find the number of bits to be skipped for the new node being constructed
	//and range of XIDs for each of its child
	newprefix = skipcompute(base, prefix, first, n);
	node = malloc(sizeof(node_patric));
	node->skip = newprefix - prefix;
	node->base = NOBASE;
	nleft = subtriecompute(base, newprefix, first, n);
	node->left = buildpatricia(base, newprefix + 1, first, nleft);
	node->right = buildpatricia(base, newprefix + 1, first + nleft, n - nleft);
	return node;
}

/*
 * This routine gives information regarding level compression i.e can "node" and
 * its children that were to be removed in level compression be removed.
 *
 * node: node of lctrie
 * level: level of node wrt node that demands compression
 */
static int compressperm(node_lc *node, int level)
{
	if (1 == level) {
		if (node->skip)
			return 1;
		else
			return 0;
	}
	if (node->skip)
		return 1;
	return (compressperm(node->subtrie[0], level - 1) +
			compressperm(node->subtrie[1], level - 1));
}

/*
 * This routine updates the branching factor of the node depending on the option
 * of being able to skip or not.
 *
 * node: node of lctrie
 */
static int updatebranch(node_lc *node)
{
	if (0 == node->branch)
		return 0;
	updatebranch(node->subtrie[0]);
	updatebranch(node->subtrie[1]);
	if (1 == node->branch)
		return 0;
	if (compressperm(node->subtrie[0], node->branch - 1) +
			compressperm(node->subtrie[1], node->branch - 1))
		node->branch = 1;
	return 0;
}

/*
 * Compute the branching factor of each node i.e number of complete levels
 * below "node"
 *
 * node: a node in a Patricia Trie (in level compression supporting data
 * structure)
 */
static int branchcompute(node_lc *node)
{
	int b1, b2;

	//If leaf node
	if (NULL == node->subtrie) {
		node->branch = 0;
		return 0;
	}
	b1 = branchcompute(node->subtrie[0]);
	b2 = branchcompute(node->subtrie[1]);
	//Select the lesser of the two values returned by the two child nodes since
	//that would represent the total number of complete levels below the current
	//node
	if ((b1 == b2) || (b1 > b2))
		node->branch = 1 + b2;
	else
		node->branch = 1 + b1;
	return node->branch;
}

/*
 * Count the total number of nodes in each level (root has level=0)
 *
 * node: address of a node in the LC-Trie
 * level: array consisting of number of nodes in each level
 * clevel: the current level of "node"
 */
static int countlevels(node_lc *node, int *level, int clevel)
{
	unsigned long tmp, i;
	level[clevel]++;

	if (NULL == node->subtrie)
		return 0;
	tmp = 1 << (node->branch);
	for (i = 0; i < tmp; i++)
		countlevels(node->subtrie[i], level, clevel + 1);
}

/*
 * Compute the address the node shall occupy in the array of LC-Trie and the
 * address the left most child of this node shall occupy in the array
 *
 * node: address of the node under consideration
 * level: an array representing the total number of nodes in each level
 * fill: an array that consists of the total number of nodes that are to the
 * left of "node" in the same level, where level in the array "level" is
 * represented by the index.
 */
static int addrarray(node_lc *node, int *level, int *fill, int clevel)
{
	int tmp, i;
	int sum = 0;
	node_t k, l;

	for (i = 0; i < clevel; i++)
		sum += level[i];
	tmp = sum + fill[clevel];
	fill[clevel]++;
	node->addr = tmp;
	//When a leaf node is reached child shall contain the address of string
	//in base vector
	if (0 == node->branch) {
		node->child = node->base;
		return 0;
	}
	l = 1 << (node->branch);
	for (k = 0; k < l; k++)
		addrarray(node->subtrie[k], level, fill, clevel+1);
	//When internal node child shall contain the address (array) of the left
	//most node it points to
	node->child = (node->subtrie[0])->addr;
}

/*
 * Fill the array with the nodes of the LC-Trie and free the memory occupied
 * by the linked list representation of LC-Trie.
 *
 * table: the final array that contains the LC-Trie
 * node: the address of the node that is currently being filled in the trie 
 */
static int fillarray(node_t *table, node_lc *node)
{
	node_t k, l;
	node_t tmp = 0;

	tmp = ((node_t) node->branch) << 40 | ((node_t) node->skip) << 32 |
							(node_t) node->child;
	table[node->addr] = tmp;
	if (0 == node->branch)
		return 0;
	l = 1 << (node->branch);
	for (k = 0; k < l; k++)
		fillarray(table, node->subtrie[k]);
}

/*
 * Convert the linked list representation to array representation to enable
 * fast lookup using the algorithm that levereages the idea of using words
 * to minimize memory accesses.
 *
 * root: address of the root node of the LC-Trie
 * table: the address of the array where the LC-Trie is to be stored
 * n: total number of nodes in the LC-Trie i.e total number of nodes present in
 * table
 */
static int lltoarray(node_lc *root, node_t *table, int *n)
{
	int level[ADRSIZE] = {0};
	int i, sum = 0;

	//Count the total number of nodes in each level where root has ``level"=0
	//and increases linearly as we traverse in a depth first manner
	countlevels(root, level, 0);
	for (i = 0; i < ADRSIZE; i++) {
		if (0 == level[i])
			break;
		sum += level[i];
	}

	//Represents the current number of nodes that have been filled in that level
	//in a breadth first fashion with the index representing the level.
	int fill[i];
	memset(fill, 0, sizeof(int) * i);
	*n = sum;

	//Compute the address the node is to occupy in the array and the address of
	//the left most node it points to
	addrarray(root, level, fill, 0);

	//Fill the array with the nodes of the LC-Trie and free the memory occupied
	//by the linked list representation of LC-Trie
	fillarray(table, root);
	return 0;
}

/*
 * Compute the addresses of all the nodes that are "level"s below the node
 *
 * node: a node that is strictly internal since the recursive call begins from
 * internal node and terminating condition mandates terminating before reaching
 * a leaf node
 * level: the number of vertical levels below "node" whose addresses are
 * required to be passed to the caller.
 */
static node_lc **compressedaddr(node_lc *node, int level)
{
	unsigned long tmp_branch, i;
	node_lc **tmp = NULL;

	//If the node is parent of the nodes that will be the new children after
	//level compression of Patricia trie
	if (1 == level) {
		tmp = node->subtrie;
		free(node);
		return tmp;
	}
	//An intermediate node that asks both its children to give the addresses of
	//nodes present "level"s below this node
	node_lc **left = NULL, **right = NULL;
	left = compressedaddr(node->subtrie[0], level - 1);
	right = compressedaddr(node->subtrie[1], level - 1);
	free(node->subtrie);
	tmp = malloc((1 << level) * sizeof(node_lc *));
	tmp_branch = 1 << (level - 1);
	for (i = 0; i < tmp_branch; i++) {
		tmp[i] = left[i];
		tmp[i + tmp_branch] = right[i];
	}
	free(node);
	return tmp;
}

/*
 * Compresses the Patricia trie into LC-Trie with the internal nodes removed
 * from memory.
 *
 * node: Node of the Patricia trie in the new data structure that supports
 * level compression.
 */
static int compresspattolc(node_lc *node)
{
	unsigned long i, tmp;

	//If leaf node
	if (0 == node->branch) {
		return 0;
	} else if (1 == node->branch) {
		// This is when node has two children or there is information
		// loss due to nodes having skip values
		compresspattolc(node->subtrie[0]);
		compresspattolc(node->subtrie[1]);
		return 0;
	}
	//If branching factor is more than 1 i.e there is need for compression
	node_lc **left = NULL, **right = NULL;

	//Find the addresses of the nodes that will be the new children after the
	//compression is achieved in this node
	left = compressedaddr(node->subtrie[0], node->branch - 1);
	right = compressedaddr(node->subtrie[1], node->branch - 1);
	free(node->subtrie);
	node->subtrie = malloc((1 << node->branch) * sizeof(node_lc *));
	tmp = 1 << (node->branch - 1);
	for (i = 0; i < tmp; i++) {
		node->subtrie[i] = left[i];
		node->subtrie[i + tmp] = right[i];
	}
	//Recursively call this routine for all the new children after the
	//compression is achieved
	for (i = 0; i < (tmp << 1); i++)
		compresspattolc(node->subtrie[i]);
}

/*
 * Converts the Patricia trie from the original minimalist data structure
 * to a new data structure that supports the level compression to be performed
 * to generate the LC-Trie.
 *
 * node: the root address of the Patricia trie
 *
 * Returns the root address of the new level compression supporting data
 * structure.
 */
static node_lc *pattonewdata(node_patric *node)
{
	node_lc *tmp = malloc(sizeof(node_lc));
	if (NULL == node->left) {
		tmp->subtrie = NULL;
		tmp->skip = node->skip;
		tmp->base = node->base;
		tmp->branch = -1;
		tmp->addr = -1;
		tmp->child = -1;
		free(node);
		return tmp;
	}
	tmp->subtrie = malloc(2 * sizeof(node_lc *));
	tmp->subtrie[0] = pattonewdata(node->left);
	tmp->subtrie[1] = pattonewdata(node->right);
	tmp->skip = node->skip;
	tmp->base = node->base;
	tmp->branch = -1;
	tmp->addr = -1;
	tmp->child = -1;
	free(node);
	return tmp;
}

/*
 * Does the job of coverting a Patricia Trie to an LC-Trie in array
 * representation.
 *
 * root: the address of the root of the Patricia Trie
 * table: the address of the array where the LC-Trie is to be stored
 * n: total number of nodes in the LC-Trie i.e total number of nodes present in
 * table
 */
static int pattolc(node_patric *root, node_t *table, int *n)
{
	node_lc *newroot = NULL;
	newroot = pattonewdata(root);
	newroot->branch = branchcompute(newroot);
	updatebranch(newroot);
	compresspattolc(newroot);
	lltoarray(newroot, table, n);
}

/*
 * The input of this routine is two pointers to entry record structures and
 * the output is an integer that essentially answers the question:
 * Is the routing entry in "s" a prefix of the routing entry in "t"?
 */
int isprefix(entry_t s, entry_t t)
{
	int tmp = 0;
	int equal = -1;
	int length;	//Represents the length of the prefix in the routing entry "s"
	xid s_data, t_data;
	xid *ps_data = &s_data;
	xid *pt_data = &t_data;

	if (NULL == s)
		return tmp;

	length = s->len;
	s_data = extract(0, length, s->data);
	t_data = extract(0, length, t->data);
	equal = comparexid(&ps_data, &pt_data);
	tmp = ((0 == length) || ((length <= t->len) && (0 == equal)));
	return tmp;
}

int compare(const void * a, const void * b)
{
	return ( *(unsigned int*)a - *(unsigned int*)b );
}

/*
 * This routine builds the nexthop table that contains nexthop XIDs in a sorted
 * fashion. The duplicate addresses are removed and this table returns a
 * pointer to the first address of an array containing the nexthop xids.
 *
 * entry: array of entryrec pointers
 * nentries: number of total entries
 * nexthopsize: contains size of nexthop table
 */
static unsigned int *buildnexthoptable(entry_t entry[], int nentries, int *nexthopsize)
{
	unsigned int *nexthop = NULL;
	unsigned int *tmpnext = malloc(nentries * sizeof(unsigned int));
	int count, i;

	//Extract the nexthop addresses from the entry array
	for (i = 0; i < nentries; i++)
		tmpnext[i] = entry[i]->nexthop;

	qsort(tmpnext, nentries, sizeof(unsigned int), compare);

	//Remove duplicates
	count = nentries > 0 ? 1 : 0;
	for (i = 1; i < nentries; i++) {
		if (tmpnext[i - 1] != tmpnext[i])
			tmpnext[count++] = tmpnext[i];
	}

	//Move the elements to an array of proper size
	nexthop = malloc(count * sizeof(unsigned int));
	for (i = 0; i < count; i++)
		nexthop[i] = tmpnext[i];

	free(tmpnext);

	*nexthopsize = count;
	return nexthop;
}

/*
 * This routine builds the entire routing table
 */
routtable_t buildrouttable(entry_t entry[], int nentries)
{
	unsigned int *nexthop; // Nexthop table
	int nnexthops;
	
	base_t *b, btemp;
	pre_t *p, ptemp;

	node_t *trie;
	comp_base_t *base;
	comp_pre_t *pre;
	
	routtable_t table = NULL;  // The complete data structure

	// Auxiliary variables
	int i, j, nprefs = 0, nbases = 0, *tmp_val = NULL;

	nexthop = buildnexthoptable(entry, nentries, &nnexthops);

	xidentrysort(entry, nentries, sizeof(entry_t), compareentries);
	// Remove duplicates
/*	size = nentries > 0 ? 1 : 0;
	for (i = 1; i < nentries; i++) {
		if (compareentries(&entry[i - 1], &entry[i]) != 0)
			entry[size++] = entry[i];
	}*/

	// The number of internal nodes in the tree can't be larger
	// than the number of entries.
	b = (base_t *) malloc(nentries * sizeof(base_t));
	p = (pre_t *) malloc(nentries * sizeof(pre_t));

	//Initialize pre-pointers
	for (i = 0; i < nentries; i++)
		entry[i]->pre = NOPRE;

	//Go through the entries and put the prefixes in p
	//and the rest of the strings in b
	for (i = 0; i < nentries; i++) {
		if (i < nentries - 1 && isprefix(entry[i], entry[i + 1])) {
			ptemp = (pre_t) malloc(sizeof(struct prerec));
			ptemp->len = entry[i]->len;
			ptemp->pre = entry[i]->pre;
			//Update 'pre' for all entries that have this prefix
			for (j = i + 1; j < nentries && isprefix(entry[i], entry[j]); j++)
					entry[j]->pre = nprefs;
			tmp_val = bsearch(&entry[i]->nexthop, nexthop,
				nnexthops, sizeof(unsigned int), compare);
			ptemp->nexthop = *tmp_val;
			p[nprefs++] = ptemp;
		} else {
			btemp = (base_t) malloc(sizeof(struct baserec));
			btemp->len = entry[i]->len;
			btemp->str = entry[i]->data;
			btemp->pre = entry[i]->pre;
			tmp_val = bsearch(&entry[i]->nexthop, nexthop,
				nnexthops, sizeof(unsigned int), compare);
			btemp->nexthop = *tmp_val;
			b[nbases++] = btemp;
		}
	}

	node_patric *root;
	root = buildpatricia(b, 0, 0, nbases);
	node_t *lctable = malloc(sizeof(node_t) * (2 * nbases - 1));
	int nnode_lc;
	pattolc(root, lctable, &nnode_lc);

	//At this point we now how much memory to allocate
	trie = malloc(nnode_lc * sizeof(node_t));
	base = malloc(nbases * sizeof(comp_base_t));
	pre = malloc(nprefs * sizeof(comp_pre_t));

	for (i = 0; i < nnode_lc; i++)
		trie[i] = lctable[i];
	free(lctable);

	for (i = 0; i < nbases; i++) {
		base[i].str = b[i]->str;
		base[i].len = b[i]->len;
		base[i].pre = b[i]->pre;
		base[i].nexthop = b[i]->nexthop;
		free(b[i]);
	}
	free(b);

	for (i = 0; i < nprefs; i++) {
		pre[i].len = p[i]->len;
		pre[i].pre = p[i]->pre;
		pre[i].nexthop = p[i]->nexthop;
		free(p[i]);
	}
	free(p);

	table = (routtable_t) malloc(sizeof(struct routtablerec));
	table->trie = trie;
	table->triesize = nnode_lc;
	table->base = base;
	table->basesize = nbases;
	table->pre = pre;
	table->presize = nprefs;
	table->nexthop = nexthop;
	table->nexthopsize = nnexthops;
	return table;
}
