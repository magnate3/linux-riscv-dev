#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef struct xidlist *list;
struct xidlist
{
	xid id1;
	xid id2;
	int result;
};

#define MAXENTRIES 50000

static int readlist(char *file_name, list addr[], int maxsize);
static int print_table(list addr[], int nlist);
static int comparexid(const void *id1, const void *id2);
static unsigned char hextochar(char a, char b);

int main(int argc, char *argv[])
{
	static list addr[MAXENTRIES];
	xid *tmp1,*tmp2;
	int nlist,i;

	if ((nlist = readlist(argv[1], addr, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}
	tmp1 = (xid *) malloc(sizeof(xid));
	tmp2 = (xid *) malloc(sizeof(xid));
	for (i=0;i<nlist;i++)
	{
		*tmp1 = addr[i]->id1;
		*tmp2 = addr[i]->id2;
		addr[i]->result = comparexid(&tmp1, &tmp2);
	}
	print_table(addr, nlist);
}

static int readlist(char *file_name, list addr[], int maxsize)
{
	int nlist = 0;
	xid id1, id2;
	FILE *in_file;

	// Auxiliary variables
	char tmp_xid1[41] = {0};
	char tmp_xid2[41] = {0};
	int loop;

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}

	while (fscanf(in_file, "%s%s", &tmp_xid1, &tmp_xid2) != EOF)
	{
		if (nlist >= maxsize) return -1;
		addr[nlist] = (list) malloc(sizeof(struct xidlist));
		
		for (loop=0;loop<20;loop++)
		{
			id1.w[loop] = hextochar(tmp_xid1[2*loop], tmp_xid1[2*loop + 1]);
			id2.w[loop] = hextochar(tmp_xid2[2*loop], tmp_xid2[2*loop + 1]);
		}
		addr[nlist]->id1 = id1;
		addr[nlist]->id2 = id2;
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

static int print_table(list addr[], int nlist)
{
	int loop,i;

	for (loop=0;loop<nlist;loop++)
	{
		if (addr[loop]->result < 0)
			printf("1\n");
		else if (addr[loop]->result > 0)
			printf("2\n");
		else
			printf("0\n");
	}
	return 0;
}
