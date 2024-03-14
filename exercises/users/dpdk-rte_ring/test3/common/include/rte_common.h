/* SPDX-License-Identifier: BSD-3-Clause
 *  * Copyright(c) 2010-2019 Intel Corporation
 *   */

#ifndef _RTE_COMMON_H_
#define _RTE_COMMON_H_

/**
 *  * @file
 *   *
 *    * Generic, commonly-used macro and inline function definitions
 *     * for DPDK.
 *      */

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#ifndef typeof
#define typeof __typeof__
#endif

#ifndef asm
#define asm __asm__
#endif

/** C extension macro for environments lacking C11 features. */
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 201112L
#define RTE_STD_C11 __extension__
#else
#define RTE_STD_C11
#endif
/**
 *  * Force alignment
 *   */
#define __rte_aligned(a) __attribute__((__aligned__(a)))
/*********** Macros for compile type checks ********/

/**
 *  * Triggers an error at compilation time if the condition is true.
 *   */
#define RTE_BUILD_BUG_ON(condition) ((void)sizeof(char[1 - 2*!!(condition)]))

/*********** Cache line related macros ********/

/** Cache line mask. */
#define RTE_CACHE_LINE_MASK (RTE_CACHE_LINE_SIZE-1)

/** Return the first cache-aligned value greater or equal to size. */
#define RTE_CACHE_LINE_ROUNDUP(size) \
        (RTE_CACHE_LINE_SIZE * ((size + RTE_CACHE_LINE_SIZE - 1) / \
        RTE_CACHE_LINE_SIZE))

/** Cache line size in terms of log2 */
#if RTE_CACHE_LINE_SIZE == 64
#define RTE_CACHE_LINE_SIZE_LOG2 6
#elif RTE_CACHE_LINE_SIZE == 128
#define RTE_CACHE_LINE_SIZE_LOG2 7
#else
//#error "Unsupported cache line size"
#endif
/** Minimum Cache line size. */
#define RTE_CACHE_LINE_MIN_SIZE 64

/** Force alignment to cache line. */
#define __rte_cache_aligned __rte_aligned(RTE_CACHE_LINE_SIZE)


/**
 *  * Force a function to be inlined
 *   */
#define __rte_always_inline inline __attribute__((always_inline))

/**
 *  * Force a function to be noinlined
 *   */
#define __rte_noinline  __attribute__((noinline))

/**
 *  * definition to mark a variable or function parameter as used so
 *   * as to avoid a compiler warning
 *    */
#define RTE_SET_USED(x) (void)(x)
/*********** Macros for pointer arithmetic ********/


/**
 *  * Combines 32b inputs most significant set bits into the least
 *   * significant bits to construct a value with the same MSBs as x
 *    * but all 1's under it.
 *     *
 *      * @param x
 *       *    The integer whose MSBs need to be combined with its LSBs
 *        * @return
 *         *    The combined value.
 *          */
static inline uint32_t
rte_combine32ms1b(register uint32_t x)
{
        x |= x >> 1;
        x |= x >> 2;
        x |= x >> 4;
        x |= x >> 8;
        x |= x >> 16;

        return x;
}

/**
 *  * Combines 64b inputs most significant set bits into the least
 *   * significant bits to construct a value with the same MSBs as x
 *    * but all 1's under it.
 *     *
 *      * @param v
 *       *    The integer whose MSBs need to be combined with its LSBs
 *        * @return
 *         *    The combined value.
 *          */
static inline uint64_t
rte_combine64ms1b(register uint64_t v)
{
        v |= v >> 1;
        v |= v >> 2;
        v |= v >> 4;
        v |= v >> 8;
        v |= v >> 16;
        v |= v >> 32;

        return v;
}

/*********** Macros to work with powers of 2 ********/

/**
 *  * Macro to return 1 if n is a power of 2, 0 otherwise
 *   */
#define RTE_IS_POWER_OF_2(n) ((n) && !(((n) - 1) & (n)))

/**
 *  * Returns true if n is a power of 2
 *   * @param n
 *    *     Number to check
 *     * @return 1 if true, 0 otherwise
 *      */
static inline int
rte_is_power_of_2(uint32_t n)
{
        return n && !(n & (n - 1));
}

/**
 *  * Aligns input parameter to the next power of 2
 *   *
 *    * @param x
 *     *   The integer value to align
 *      *
 *       * @return
 *        *   Input parameter aligned to the next power of 2
 *         */
static inline uint32_t
rte_align32pow2(uint32_t x)
{
        x--;
        x = rte_combine32ms1b(x);

        return x + 1;
}
#ifdef __cplusplus
}
#endif

#endif
