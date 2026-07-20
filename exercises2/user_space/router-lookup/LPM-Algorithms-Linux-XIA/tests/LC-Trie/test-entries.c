#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef xid *list;
typedef xid nexthop_t;
typedef struct entryrec *entry_t;
struct entryrec {
   xid data;
   int len;
   nexthop_t nexthop;
   int pre;
}; 

#define MAXENTRIES 50000

static int readentries(char *file_name, entry_t entry[], int maxsize);
static int print_table(entry_t entry[], int nentries);
static int comparexid(const void *id1, const void *id2);
static int compareentries(const void *id1, const void *id2);
void xidentrysort(entry_t entry[], size_t narray, size_t size, int (*compare)(const void *, const void *));
static unsigned char hextochar(char a, char b);


int main(int argc, char *argv[])
{
	static entry_t addr[MAXENTRIES];
	int nentries, i, j, size;
	xid tmp;

	if ((nentries = readentries(argv[1], addr, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}

	xidentrysort(&addr[0], nentries, sizeof(entry_t), compareentries);
	size = nentries > 0 ? 1 : 0;
	for (i = 1; i < nentries; i++)
	{
		if (compareentries(&addr[i-1], &addr[i]) != 0)
			addr[size++] = addr[i];
	}
	print_table(addr, size);
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

static int print_table(entry_t entry[], int nentries)
{
	int loop,i;
	xid data, nexthop;

	for (loop=0;loop<nentries;loop++)
	{
		data = entry[loop]->data;
		nexthop = entry[loop]->nexthop;
		for (i=0;i<20;i++)
			printf("%02x", data.w[i]);
		printf(" %d ", entry[loop]->len);
		for (i=0;i<20;i++)
			printf("%02x", nexthop.w[i]);
		printf("\n");
	}
	return 0;
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

void xidentrysort(entry_t entry[], size_t narray, size_t size, int (*compare)(const void *, const void *))
{
	qsort(&entry[0], narray, size, compare);
}

static unsigned char hextochar(char a, char b)
{
	unsigned char hn, ln;

	hn = a > '9' ? a - 'a' + 10 : a - '0';
	ln = b > '9' ? b - 'a' + 10 : b - '0';
	return ((hn << 4 ) | ln);
}
