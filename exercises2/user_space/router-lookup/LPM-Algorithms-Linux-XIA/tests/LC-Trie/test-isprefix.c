#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <float.h>

typedef unsigned int word;
typedef struct xid { unsigned char w[20]; } xid;
typedef xid nexthop_t;
typedef struct entryrec *entry_t;
struct entryrec
{
	xid data;
	int len;
	nexthop_t nexthop;
	int pre;
};

#define MAXENTRIES 50000

static int readentries(char *file_name1, char *file_name2, entry_t entry1[],
		entry_t entry2[], int maxsize);
static unsigned char hextochar(char a, char b);
static int isprefix(entry_t s, entry_t t);
static int comparexid(const void *id1, const void *id2);
static xid extract(int pos, int length, xid data);
static xid shift_left(xid id, int shift);
static xid shift_right(xid id, int shift);

int main(int argc, char *argv[])
{
	static entry_t entry1[MAXENTRIES];
	static entry_t entry2[MAXENTRIES];
	int nentries,i;

	if ((nentries = readentries(argv[1], argv[2], entry1, entry2, 
					MAXENTRIES)) < 0)
	{
		fprintf(stderr, "Input file too large.\n");
		return 1;
	}
	for (i=0;i<nentries;i++)
	{
		printf("%d\n", isprefix(entry1[i], entry2[i]));
	}
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

static xid extract(int pos, int length, xid data)
{
	int i;
	xid tmp;
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

static int readentries(char *file_name1, char *file_name2, entry_t entry1[], 
		entry_t entry2[], int maxsize)
{
	int nentries1 = 0;
	int nentries2 = 0;
	xid data, nexthop;
	int len;
	FILE *in_file1, *in_file2;

	// Auxiliary variables
	char tmp_data[41] = {0};
	char tmp_nexthop[41] = {0};
	int i;

	if (!(in_file1 = fopen(file_name1, "rb")))
	{
		perror(file_name1);
		exit(-1);
	}
	
	if (!(in_file2 = fopen(file_name2, "rb")))
	{
		perror(file_name2);
		exit(-1);
	}

	while (fscanf(in_file1, "%s%i%s", &tmp_data, &len, &tmp_nexthop) != EOF)
	{
		if (nentries1 >= maxsize) return -1;
		entry1[nentries1] = (entry_t) malloc(sizeof(struct entryrec));
		for (i=0;i<20;i++)
		{
			data.w[i] = hextochar(tmp_data[2*i], tmp_data[2*i + 1]);
			nexthop.w[i] = hextochar(tmp_nexthop[2*i], tmp_nexthop[2*i + 1]);
		}
		for (i=0;i<(len/8);i++);
		data.w[i] = data.w[i] >> (8 - (len%8)) << (8 - (len%8));
		i++;
		for (;i<20;i++)
			data.w[i] = (char) 0;
		entry1[nentries1]->data = data;
		entry1[nentries1]->len = len;
		entry1[nentries1]->nexthop = nexthop;
		nentries1++;
	}
	while (fscanf(in_file2, "%s%i%s", &tmp_data, &len, &tmp_nexthop) != EOF)
	{
		if (nentries2 >= maxsize) return -1;
		entry2[nentries2] = (entry_t) malloc(sizeof(struct entryrec));
		for (i=0;i<20;i++)
		{
			data.w[i] = hextochar(tmp_data[2*i], tmp_data[2*i + 1]);
			nexthop.w[i] = hextochar(tmp_nexthop[2*i], tmp_nexthop[2*i + 1]);
		}
		for (i=0;i<(len/8);i++);
		data.w[i] = data.w[i] >> (8 - (len%8)) << (8 - (len%8));
		i++;
		for (;i<20;i++)
			data.w[i] = (char) 0;
		entry2[nentries2]->data = data;
		entry2[nentries2]->len = len;
		entry2[nentries2]->nexthop = nexthop;
		nentries2++;
	}
	if (nentries1 == nentries2)
		return nentries1;
}
