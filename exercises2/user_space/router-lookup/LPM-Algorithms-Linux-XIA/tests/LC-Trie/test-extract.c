#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef struct extractrec *extractrec_t;
struct extractrec
{
	xid data;
	int pos;
	int len;
};

#define MAXENTRIES 50000

static int readrecords(char *file_name, extractrec_t rec[], int maxsize);
static int print_substrings(extractrec_t rec[], int nrec);
unsigned char hextochar(char a, char b);
static xid shift_left(xid id, int shift);
static xid shift_right(xid id, int shift);
static xid extract(int pos, int length, xid data);

int main(int argc, char *argv[])
{
	static extractrec_t rec[MAXENTRIES];
	int nrec;

	if ((nrec = readrecords(argv[1], rec, MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}
	print_substrings(rec, nrec);
}

unsigned char hextochar(char a, char b)
{
	unsigned char hn, ln;

	hn = a > '9' ? a - 'a' + 10 : a - '0';
	ln = b > '9' ? b - 'a' + 10 : b - '0';
	return ((hn << 4 ) | ln);
}

static int readrecords(char *file_name, extractrec_t rec[], int maxsize)
{
	int nrec = 0;
	xid data;
	int len, pos;
	FILE *in_file;

	// Auxiliary variables
	char tmp_data[41] = {0};
	int loop;

	if (!(in_file = fopen(file_name, "rb")))
	{
		perror(file_name);
		exit(-1);
	}

	while (fscanf(in_file, "%s%i%i", &tmp_data, &len, &pos) != EOF)
	{
		if (nrec >= maxsize) return -1;
		rec[nrec] = (extractrec_t) malloc(sizeof(struct extractrec));

		for (loop=0;loop<20;loop++)
		{
			data.w[loop] = hextochar(tmp_data[2*loop], tmp_data[2*loop + 1]);
		}
		rec[nrec]->data = data;
		rec[nrec]->len = len;
		rec[nrec]->pos = pos;
		nrec++;
	}
	return nrec;
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

static int print_substrings(extractrec_t rec[], int nrec)
{
	int i,j;
	xid tmp;

	for (i=0;i<nrec;i++)
	{
		tmp = extract(rec[i]->pos, rec[i]->len, rec[i]->data);
		for (j=0;j<20;j++)
			printf("%02x", tmp.w[j]);
		printf("\n");
	}
	return 0;
}
