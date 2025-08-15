/* $Id: general.c,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#include <string.h>
#include "general.h"

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

typedef union
{
    int   integer;
    char  array[sizeof(int)];
} endian;

int primitive_compare(const void *x, const void *y, size_t sz)
{
    int ret;

    if (judge_endian() == 1) {
        const char *tmp1 = (const char*) x + sz - 1;
        const char *tmp2 = (const char*) y + sz - 1;

        while (--sz >= 0) {
            if ((ret = memcmp(tmp1, tmp2, 1)) != 0)
                break;
            --tmp1;
            --tmp2;
        }
    } else {
        ret = memcmp(x, y, sz);
    }

    return ret;
}

int judge_endian()
{
    endian e;

    e.integer = 1;
    if (1 == e.array[0]) return 1;
    return 0;
}
