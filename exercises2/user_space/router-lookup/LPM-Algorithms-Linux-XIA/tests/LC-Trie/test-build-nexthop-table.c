#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef xid nexthop_t;
typedef xid *list;
typedef struct entryrec *entry_t;
struct entryrec
{
	xid data;
	int len;
	nexthop_t nexthop;
	int pre;
};

#define MAXENTRIES 50000

static int readentries(char *file_name, entry_t entry[], int maxsize);
static int print_nexttable(nexthop_t *nexthoptable, int nexthoptablesize);
unsigned char hextochar(char a, char b);
int comparexid(const void *id1, const void *id2);
void xidsort(list *base, size_t narray, size_t size, int (*compare)(const void *, const void *));
static nexthop_t *buildnexthoptable(entry_t entry[], int nentries, int *nexthopsize);

int main(int argc, char *argv[])
{
	static entry_t entry[MAXENTRIES];
	int nentries, nexthoptablesize;
	nexthop_t *nexthoptable;

	if ((nentries = readentries(argv[1], entry, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}
	nexthoptable = buildnexthoptable(entry, nentries, &nexthoptablesize);
	print_nexttable(nexthoptable, nexthoptablesize);
}

unsigned char hextochar(char a, char b)
{
	unsigned char hn, ln;

	hn = a > '9' ? a - 'a' + 10 : a - '0';
	ln = b > '9' ? b - 'a' + 10 : b - '0';
	return ((hn << 4 ) | ln);
}

static int readentries(char *file_name, entry_t entry[], int maxsize)
{
	int nentries = 0;
	xid data, nexthop;
	int len;
	FILE *in_file;

	// Auxiliary variables
	char tmp_data[41] = {0};
	char tmp_nexthop[41] = {0};
	int loop;

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}

	while (fscanf(in_file, "%s%i%s", &tmp_data, &len, &tmp_nexthop) != EOF)
	{
		if (nentries >= maxsize) return -1;
		entry[nentries] = (entry_t) malloc(sizeof(struct entryrec));

		for (loop=0;loop<20;loop++)
		{
			data.w[loop] = hextochar(tmp_data[2*loop], tmp_data[2*loop + 1]);
			nexthop.w[loop] = hextochar(tmp_nexthop[2*loop], tmp_nexthop[2*loop + 1]);
		}
		// extract the prefix from the bit pattern
		for (loop=0;loop<(len/8);loop++);
		data.w[loop] = data.w[loop] >> (8 - (len%8)) << (8 - (len%8));
		loop++;
		for (;loop<20;loop++)
			data.w[loop] = (char) 0;

		entry[nentries]->data = data;
		entry[nentries]->len = len;
		entry[nentries]->nexthop = nexthop;
		nentries++;
	}
	return nentries;
}

static nexthop_t *buildnexthoptable(entry_t entry[], int nentries, int *nexthopsize)
{
	nexthop_t *nexthop, *nexttemp;
	nexthop_t *pnexttmp[nentries];
	int count, i;

	/* Extract the nexthop addresses from the entry array */
	nexttemp = (nexthop_t *) malloc(nentries * sizeof(nexthop_t));
	for (i=0;i<nentries;i++)
	{
		nexttemp[i] = entry[i]->nexthop;
		pnexttmp[i] = &nexttemp[i];
	}

	xidsort(pnexttmp, nentries, sizeof(nexthop_t *), comparexid);

	/* Remove duplicates */
	count = nentries > 0 ? 1 : 0;
	for (i = 1; i < nentries; i++)
	{
		if (comparexid(&pnexttmp[i-1], &pnexttmp[i]) != 0)
			pnexttmp[count++] = pnexttmp[i];
	}

     	/* Move the elements to an array of proper size */
	nexthop = (nexthop_t *) malloc(count * sizeof(nexthop_t));
	for (i = 0; i < count; i++)
		nexthop[i] = *pnexttmp[i];

	free(nexttemp);

	*nexthopsize = count;
	return nexthop;
}

int comparexid(const void *id1, const void *id2)
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

void xidsort(list *base, size_t narray, size_t size, int (*compare)(const void *, const void *))
{
	qsort(&base[0], narray, size, compare);
}

static int print_nexttable(nexthop_t *nexthoptable, int nexthoptablesize)
{
	int i,j;
	nexthop_t data;

	for (i=0;i<nexthoptablesize;i++)
	{
		data = nexthoptable[i];
		for (j=0;j<20;j++)
			printf("%02x", data.w[j]);
		printf("\n");
	}
}
