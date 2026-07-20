#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef xid nexthop_t;
typedef struct baserec *base_t;
struct baserec {
   xid str;
   int len;
   int pre;
   int nexthop;
};
typedef struct prerec *pre_t;
struct prerec {
   int len;
   int pre;
   int nexthop;
};
typedef struct entryrec *entry_t;
struct entryrec {
   xid data;
   int len;
   nexthop_t nexthop;
   int pre;
};
typedef struct node_patric node_patric;
struct node_patric
{
	int skip;
	node_patric *left;
	node_patric *right;
	int base;
};

#define NOBASE -1
#define NOPRE -1
#define MAXENTRIES 50000
#define ADRSIZE 160

static int readlist(char *file_name, entry_t entry[], int maxsize);
static int comparexid(const void *id1, const void *id2);
static unsigned char hextochar(char a, char b);
static xid extract(int pos, int length, xid data);
static xid shift_left(xid id, int shift);
static xid shift_right(xid id, int shift);
static xid removexid(int bits, xid data);
static int incrementxid(xid *pxid);
xid addxid(xid a, xid b);
word xidtounsigned(xid *bitpat);
static int isprefix(entry_t s, entry_t t);
static int compareentries(const void *id1, const void *id2);
static void xidentrysort(entry_t entry[], size_t narray, size_t size, int (*compare)(const void *, const void *));
static node_patric *buildpatricia(base_t base[], int prefix, int first, int n);
static int subtriecompute(base_t base[], int prefix, int first, int n);
static int skipcompute(base_t base[], int prefix, int first, int n);
static print_trie(node_patric *trie, int level);

int main(int argc, char *argv[])
{
	static entry_t entry[MAXENTRIES];
	int nentry,i,j,k;
	int nprefs = 0, nbases = 0;
	int nextfree = 1;
	base_t *b, btemp;
	pre_t *p, ptemp;

	if ((nentry = readlist(argv[1], entry, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}

	xidentrysort(entry, nentry, sizeof(entry_t), compareentries);
	b = (base_t *) malloc(nentry * sizeof(base_t));
	p = (pre_t *) malloc(nentry * sizeof(pre_t));
	for (i = 0; i < nentry; i++)
	{
		if (i < nentry-1 && isprefix(entry[i], entry[i+1]))
		{
			ptemp = (pre_t) malloc(sizeof(struct prerec));
			ptemp->len = entry[i]->len;
			ptemp->pre = entry[i]->pre;
			/* Update 'pre' for all entries that have this prefix */
			for (j = i + 1; j < nentry && isprefix(entry[i], entry[j]); j++)
				entry[j]->pre = nprefs;
			p[nprefs++] = ptemp;
		}
		else
		{
			btemp = (base_t) malloc(sizeof(struct baserec));
			btemp->len = entry[i]->len;
			btemp->str = entry[i]->data;
			btemp->pre = entry[i]->pre;
			b[nbases++] = btemp;
		}
	}
	node_patric *root;
	root = buildpatricia(b, 0, 0, nbases);
	print_trie(root, 0);
}

static node_patric *buildpatricia(base_t base[], int prefix, int first, int n)
{
	int i,newprefix,nleft;
	node_patric *node;

	// Reached the leaf node
	if (1 == n)
	{
		node = malloc(sizeof(node_patric));
		node->left = NULL;
		node->right = NULL;
		node->skip = base[first]->len - prefix;
		node->base = first;
		return node;
	}
	newprefix = skipcompute(base, prefix, first, n);
	node = malloc(sizeof(node_patric));
	node->skip = newprefix - prefix;
	node->base = NOBASE;
	nleft = subtriecompute(base, newprefix, first, n);
	node->left = buildpatricia(base, newprefix+1, first, nleft);
	node->right = buildpatricia(base, newprefix+1, first+nleft, n-nleft);
	return node;
}

static int subtriecompute(base_t base[], int prefix, int first, int n)
{
	xid tmp;
	int nleft = 0;
	unsigned char left = (unsigned char) 0;

	// Find the number of prefixes in base that belong to left subtrie
	tmp = extract(prefix, 1, base[first]->str);
	while (left == tmp.w[19])
	{
		nleft++;
		tmp = extract(prefix, 1, base[first+nleft]->str);
	}
	return nleft;
}

static int skipcompute(base_t base[], int prefix, int first, int n)
{
	int i,match;
	xid low,high,*plow,*phigh;

	// Compute the skip value for that node
	low = removexid(prefix, base[first]->str);
	high = removexid(prefix, base[first+n-1]->str);
	i = prefix;
	plow = malloc(sizeof(xid));
	phigh = malloc(sizeof(xid));
	*plow = extract(i, 1, low);
	*phigh = extract(i, 1, high);
	match = comparexid(&plow, &phigh);
	while (0 == match)
	{
		i++;
		*plow = extract(i, 1, low);
		*phigh = extract(i, 1, high);
		match = comparexid(&plow, &phigh);
	}
	return i;
}

static print_trie(node_patric *trie, int level)
{
	printf("Level:%d\n", level);
	printf("Skip:%d\n", trie->skip);
	printf("Base Pointer:%d\n", trie->base);
	printf("\n");
	if (NULL != trie->left)
	{
		print_trie(trie->left, level+1);
		print_trie(trie->right, level+1);
	}
}

static int readlist(char *file_name, entry_t entry[], int maxsize)
{
	int nentry = 0;
	xid data, nextid;
	FILE *in_file;

	// Auxiliary variables
	char tmp_xid[41] = {0};
	int i, len;

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}
	memset(&nextid, 0, 20);

	while (fscanf(in_file, "%s%d", &tmp_xid, &len) != EOF)
	{
		if (nentry >= maxsize) return -1;
		entry[nentry] = (entry_t) malloc(sizeof(struct entryrec));
		
		for (i=0;i<20;i++)
			data.w[i] = hextochar(tmp_xid[2*i], tmp_xid[2*i + 1]);
			
		for (i=0;i<(len/8);i++);
		data.w[i] = data.w[i] >> (8 - (len%8)) << (8 - (len%8));
		i++;
		for (;i<20;i++)
			data.w[i] = (char) 0;
		entry[nentry]->data = data;
		entry[nentry]->len = len;
		entry[nentry]->nexthop = nextid;
		entry[nentry]->pre = NOPRE;
		nentry++;
	}
	return nentry;
}

static void xidentrysort(entry_t entry[], size_t narray, size_t size, int (*compare)(const void *, const void *))
{
	qsort(&entry[0], narray, size, compare);
}

static int compareentries(const void *id1, const void *id2)
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

static int isprefix(entry_t s, entry_t t)
{
	int tmp = 0;
	int equal = -1;
	int length;
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

static int comparexid(const void *id1, const void *id2)
{
	int tmpresult;
	xid **tmp1 = (xid **)id1;
	xid **tmp2 = (xid **)id2;

	tmpresult = memcmp(*tmp1, *tmp2, 20);

	if (tmpresult < 0)
		return -1;
	else if (tmpresult > 0)
		return 1;
	else
		return 0;
}

static unsigned char hextochar(char a, char b)
{
	unsigned char hn, ln;

	hn = a > '9' ? a - 'a' + 10 : a - '0';
	ln = b > '9' ? b - 'a' + 10 : b - '0';
	return ((hn << 4 ) | ln);
}

static xid shift_left(xid id, int shift)
{
	int byte_s = 0;
	int bit_s = 0;
	int i,j;
	xid tmp;

	if (shift >= 160)
		memset(&tmp, 0, 20);
	else
	{
		byte_s = shift/8;
		bit_s = shift%8;
		if (0 == byte_s)
		{
			for (i=0;i<19;i++)
				id.w[i] = (id.w[i] << bit_s)|(id.w[i+1] >> (8-bit_s));
			id.w[i] = id.w[i] << bit_s;
			tmp = id;
			return tmp;
		}
		for (i=0;i<(20 - byte_s);i++)
			id.w[i] = id.w[i+byte_s];
		for (j=i;j<20;j++)
			id.w[j] = (unsigned char) 0;
		for (j=0;j<i;j++)
			id.w[j] = (id.w[j] << bit_s)|(id.w[j+1] >> (8-bit_s));
		tmp = id;
	}
	return tmp;
}

static xid shift_right(xid id, int shift)
{
	int byte_s = 0;
	int bit_s = 0;
	int i,j;
	xid tmp;

	if (shift >= 160)
		memset(&tmp, 0, 20);
	else
	{
		byte_s = shift/8;
		bit_s = shift%8;
		if (0 == byte_s)
		{
			for (i=19;i>0;i--)
				id.w[i] = (id.w[i] >> bit_s)|(id.w[i-1] << (8-bit_s));
			id.w[i] = id.w[i] >> bit_s;
			tmp = id;
			return tmp;
		}
		for (i=19;i>(byte_s - 1);i--)
			id.w[i] = id.w[i-byte_s];
		for (j=i;j>-1;j--)
			id.w[j] = (unsigned char) 0;
		for (j=19;j>i;j--)
			id.w[j] = (id.w[j] >> bit_s)|(id.w[j-1] << (8-bit_s));
		tmp = id;
	}
	return tmp;
}

static xid extract(int pos, int length, xid data)
{
	int i;
	xid tmp;

	// length = 0 is handled in the boolean expression and is short-circuit
	// even though length of zero works in case of xids
	tmp = shift_right(shift_left(data, pos), (160-length));
	return tmp;
}

static xid removexid(int bits, xid data)
{
	int i;
	xid tmp;

	tmp = shift_right(shift_left(data, bits), bits);
	return tmp;
}

static int incrementxid(xid *pxid)
{
	int i = 19;
	unsigned char tmp_max = (unsigned char) -1;

	while ((tmp_max == pxid->w[i]) && (i > -1))
	{
		pxid->w[i] = (unsigned char) 0;
		i--;
	}
	if (-1 != i)
		pxid->w[i] = pxid->w[i] + 1;
}
