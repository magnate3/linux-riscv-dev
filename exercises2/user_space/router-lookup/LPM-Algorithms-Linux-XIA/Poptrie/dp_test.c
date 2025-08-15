#include "poptrie_xid.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

#define UINT32XID (160 - 18)

static int _msb_int_to_xid(uint32_t num, XID *addr, int len)
{
	XID tmp;

	memset(&tmp, 0, 20);
	if (len <= 2) {
		// If value of XID lies in [0, 3]
		tmp.w[2] = ((unsigned char) num) << 6;
	} else if ((len <= 10) && (len > 2)) {
		// If value of XID lies in [4, 2^10 - 1]
		tmp.w[2] = ((unsigned char) num) << 6;
		tmp.w[1] = (unsigned char) (num >> 2);
	} else {
		// If value of XID lies in [2^10, 2^18 - 1]
		tmp.w[2] = ((unsigned char) num) << 6;
		tmp.w[1] = (unsigned char) (num >> 2);
		tmp.w[0] = (unsigned char) (num >> 10);
	}
	memcpy(addr, &tmp, 20);
}

static int _create_data_dp(int size, XID *tmp_xid)
{
	uint32_t i;
	int j;
	XID *tmp = tmp_xid;
	XID check_xid;

	for (i = 0; i < (1 << size); i++) {
		_msb_int_to_xid(i, tmp, size);
		check_xid = shift_right(*tmp, UINT32XID);
		assert(i == XIDtounsigned(&check_xid));
		tmp++;
	}
//	printf("Verified Data creation for upto 2^%d\n", size);

	return 0;
}

int main(int argc, const char *const argv[])
{
	int ret;
	int i, j;
	struct poptrie *poptrie = NULL;
	XID *tmp_xid = NULL;
	XID tmp;
	unsigned int nexthop = 1;

	for (i = 1; i < 19; i++) {
		assert(NULL != (poptrie = poptrie_init(NULL, 19, 22)));
		tmp_xid = malloc(sizeof(XID) * (1 << i));
		assert(0 == _create_data_dp(i, tmp_xid));
		for (j = 0; j < (1 << i); j++) {
//			for (k = 0; k < 20; k++)
//				printf("%x", *(tmp_xid + j).w[k]);
			tmp = shift_left(*(tmp_xid + j), (18 - i));
			assert(poptrie_route_add(poptrie, tmp, i, &nexthop) >= 0);
			nexthop = (nexthop + 1) % (i + 1);
			if (!nexthop)
				nexthop++;
		}
		for (j = 0; j < (1 << i); j++) {
			tmp = shift_left(*(tmp_xid + j), (18 - i));
			assert(poptrie_lookup(poptrie, tmp) == poptrie_rib_lookup(poptrie, tmp));
		}
		printf("Added and searched routes upto 2^%d\n", i);
		free(tmp_xid);
		poptrie_release(poptrie);
	}

	return 0;
}
