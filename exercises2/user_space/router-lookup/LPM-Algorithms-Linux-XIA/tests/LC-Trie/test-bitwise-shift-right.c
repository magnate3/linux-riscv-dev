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


static int readlist(char *file_name, list addr[], int shift[], int maxsize);
static unsigned char hextochar(char a, char b);
static xid shift_right(xid id, int shift);

int main(int argc, char *argv[])
{
	static list addr[MAXENTRIES];
	static int shift[MAXENTRIES] = {0};
	int nlist,i,j;
	xid tmp;

	if ((nlist = readlist(argv[1], addr, shift, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}

	for (i=0;i<nlist;i++)
	{
		tmp = shift_right(*addr[i], shift[i]);
		for (j=0;j<20;j++)
			printf("%02x", tmp.w[j]);
		printf("\n");
	}
}

static int readlist(char *file_name, list addr[], int shift[], int maxsize)
{
	int nlist = 0;
	xid data;
	FILE *in_file;

	// Auxiliary variables
	char tmp_data[41] = {0};
	int loop, s_l;

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}

	while (fscanf(in_file, "%s%d", &tmp_data, &s_l) != EOF)
	{
		if (nlist >= maxsize) return -1;
		addr[nlist] = (list) malloc(sizeof(struct xid));

		for (loop=0;loop<20;loop++)
		{
			data.w[loop] = hextochar(tmp_data[2*loop], tmp_data[2*loop + 1]);
		}
		*addr[nlist] = data;
		shift[nlist] = s_l;
		nlist++;
	}
	return nlist;
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

static unsigned char hextochar(char a, char b)
{
	unsigned char hn, ln;

	hn = a > '9' ? a - 'a' + 10 : a - '0';
	ln = b > '9' ? b - 'a' + 10 : b - '0';
	return ((hn << 4 ) | ln);
}
