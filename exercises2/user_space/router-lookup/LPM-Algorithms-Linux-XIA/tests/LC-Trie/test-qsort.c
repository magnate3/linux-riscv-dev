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


static int readlist(char *file_name, list addr[], int maxsize);
static int comparexid(const void *id1, const void *id2);
static int xidsort(list *base, size_t narray, size_t size, int (*compare)(const void *, const void *));
static unsigned char hextochar(char a, char b);


int main(int argc, char *argv[])
{
	static list addr[MAXENTRIES];
	int nlist,i,j;
	xid tmp;

	if ((nlist = readlist(argv[1], addr, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}

	xidsort((list *)&addr[0], nlist, sizeof(list), comparexid);

	for (i=0;i<nlist;i++)
	{
		tmp = *addr[i];
		for (j=0;j<20;j++)
			printf("%02x", tmp.w[j]);
		printf("\n");
	}
}

static int readlist(char *file_name, list addr[], int maxsize)
{
	int nlist = 0;
	xid data;
	FILE *in_file;

	// Auxiliary variables
	char tmp_data[41] = {0};
	int loop;

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}

	while (fscanf(in_file, "%s", &tmp_data) != EOF)
	{
		if (nlist >= maxsize) return -1;
		addr[nlist] = (list) malloc(sizeof(struct xid));

		for (loop=0;loop<20;loop++)
		{
			data.w[loop] = hextochar(tmp_data[2*loop], tmp_data[2*loop + 1]);
		}
		*addr[nlist] = data;
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

static int xidsort(list *base, size_t narray, size_t size, int (*compare)(const void *, const void *))
{
	qsort(&base[0], narray, size, compare);
}

static unsigned char hextochar(char a, char b)
{
	unsigned char hn, ln;

	hn = a > '9' ? a - 'a' + 10 : a - '0';
	ln = b > '9' ? b - 'a' + 10 : b - '0';
	return ((hn << 4 ) | ln);
}
