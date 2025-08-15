#include "poptrie_xid.h"
#include <string.h>

/*
 * Bitwise shift left an XID by shift bits and return that shifted XID
 */
XID shift_left(XID id, int shift)
{
	int byte_s = 0;
	int bit_s = 0;
	int i,j;
	XID tmp;

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

/*
 * Bitwise shift right an XID by shift bits and return that shifted XID
 */
XID shift_right(XID id, int shift)
{
	int byte_s = 0;
	int bit_s = 0;
	int i,j;
	XID tmp;

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

/*
 * Extract length bits starting at pos from data
 */
XID extractXID(int pos, int length, XID data)
{
	int i;
	XID tmp;

	// length = 0 is handled in the boolean expression and is short-circuit
	// even though length of zero works in case of xids
	tmp = shift_right(shift_left(data, pos), (160-length));
	return tmp;
}

/*
 * Remove "bits" bits from data
 */
XID removeXID(int bits, XID data)
{
	int i;
	XID tmp;

	tmp = shift_right(shift_left(data, bits), bits);
	return tmp;
}

/*
 * Compare two xids i.e 160-bit addresses and returns 1, 0, -1 
 * id1 > id2 : return 1
 * id1 < id2 : return -1
 * id1 = id2 : return 0
 */
int compareXID(const void *id1, const void *id2)
{
	int tmpresult;
	XID **tmp1 = (XID **)id1;
	XID **tmp2 = (XID **)id2;

	tmpresult = memcmp(*tmp1, *tmp2, 20);

	if (tmpresult < 0)
		return -1;
	else if (tmpresult > 0)
		return 1;
	else
		return 0;
}

/*
 * Convert and XID to uint32_t
 */
uint32_t XIDtounsigned(XID *bitpat)
{
	uint32_t tmp = bitpat->w[16] << 24 | bitpat->w[17] << 16 |
				bitpat->w[18] << 8 | bitpat->w[19];
	return tmp;
}
