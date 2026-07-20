/* $Id: general.h,v 1.4 2003/09/15 12:47:46 jfwan Exp $ */
#ifndef WAN_GENERAL_H
#define WAN_GENERAL_H

/**
 * @author jfwan@vip.sina.com
 * @date   09/15/2003
 */

/* alignment */
enum { NR_ALIGN_BASE = 3 }; /* 8 bytes alignment */
enum { NR_ALIGN_BYTES = 8 };
#define ALIGN(x) ((x + NR_ALIGN_BYTES - 1) & ~(NR_ALIGN_BYTES - 1))
#if 0
#define ALIGN(x) (((((x) - 1) >> ALIGN_BASE) + 1) << ALIGN_BASE)
#endif

/* hide internal data structures */
#define EXTERN_TYPE(type) typedef struct type##_struct *type
#define CONST_TYPE(type, const_type) \
    typedef struct type##_struct const * const const_type

/* for Multi-Threading */
#ifndef WAN_THREADS
#define VOLATILE
#else
#define VOLATILE volatile
#endif

#define DISTANCE(b,e) ((char*) (b) - (char*) (e))

typedef enum { FALSE = 0, TRUE } boolean;

#include <stdio.h>
#include <stdlib.h>
#define warning(x) fprintf(stderr, "%s\n", x)
#define error(x)   fprintf(stderr, "%s\n", x), exit(1)

typedef int (*comp_fun)(const void*, const void*);

#include <stddef.h> /* size_t */

/* XXX: just compare built-in types in little endian cpus */
int primitive_compare(const void *x, const void *y, size_t size);

/* returns 1 for little endian, 0 for big endian */
int judge_endian();

#endif // WAN_GENERAL_H
