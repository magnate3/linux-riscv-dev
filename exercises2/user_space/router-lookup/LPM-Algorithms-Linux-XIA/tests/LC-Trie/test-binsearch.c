#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef xid *list;

#define MAXENTRIES 50000

static int readlist(char *file_name1, list table, int maxsize);
static int comparexid(const void *id1, const void *id2);
static unsigned char hextochar(char a, char b);
static int binsearch(xid data, xid *table, int n);

int main(int argc, char *argv[])
{
	static xid table[MAXENTRIES];
	int nlist,i,j,val;
	char tmp_data[41] = {0};
	FILE *in_file;
	xid data;

	if ((nlist = readlist(argv[1], table, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}
	if (!(in_file = fopen(argv[2], "rb")))
	{
		perror(argv[2]);
		exit(-1);
	}
	while (fscanf(in_file, "%s", &tmp_data) != EOF)
	{
		for (i=0;i<20;i++)
			data.w[i] = hextochar(tmp_data[2*i], tmp_data[2*i + 1]);
		val = binsearch(data, table, nlist);
		printf("%d\n", val);
	}
}

static int binsearch(xid data, xid *table, int n)
{
	int low, mid, high, val;
	xid *pdata = &data;
	xid *pmid;

	low = 0;
	high = n-1;
	while (low <= high)
	{
		mid = (low + high)/2;
		pmid = &table[mid];
		val = comparexid(&pdata, &pmid);
		if (-1 == val)
			high = mid - 1;
		else if (1 == val)
			low = mid + 1;
		else
			return mid;
	}
	return -1;
}

static int readlist(char *file_name, list addr, int maxsize)
{
	int nlist = 0;
	int i = 0;
	xid data;
	FILE *in_file;

	// Auxiliary variables
	char tmp_data[41] = {0};

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}
	while (fscanf(in_file, "%s", &tmp_data) != EOF)
	{
		for (i=0;i<20;i++)
			data.w[i] = hextochar(tmp_data[2*i], tmp_data[2*i + 1]);
		addr[nlist] = data;
		nlist++;
	}
	return nlist;
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
