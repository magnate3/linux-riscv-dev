/**
 * Bitset data structure.
 * Adapted from https://c-faq.com/misc/bitsets.html
 */
#include <limits.h>

#define BITMASK(b) (1 << ((b) % CHAR_BIT))
#define BITSLOT(b) ((b) / CHAR_BIT)

#define BITSET_DECLARE(name, nb) volatile char name[BITSET_SLOTS (nb)]
#define BITSET_INIT(a, nb)                \
    do                                    \
    {                                     \
        memset ((char *)a, 0, BITSET_SLOTS (nb)); \
    } while (0)

/**
 * Set the bit at position b in the bitset a.
 * @param a the bitset
 * @param b the position of the bit to set. The position must be in the range [0, number of bits in the bitset)
 * @example BITSET_SET(bitset, 2); // set bit 2. BITSET_TEST(bitset, 2) == 1
 */
#define BITSET_SET(a, b) ((a)[BITSLOT (b)] |= BITMASK (b))

/**
 * Clear the bit at position b in the bitset a.
 * @param a the bitset
 * @param b the position of the bit to clear. The position must be in the range [0, number of bits in the bitset)
 * @example BITSET_CLEAR(bitset, 2); // clear bit 2. BITSET_TEST(bitset, 2) == 0
 */
#define BITSET_CLEAR(a, b) ((a)[BITSLOT (b)] &= ~BITMASK (b))

/**
 * Test if the bit at position b is set in the bitset a.
 * @param a the bitset
 * @param b the position of the bit to test. The position must be in the range [0, number of bits in the bitset)
 * @example if (BITSET_TEST(bitset, 2)) // if bit 2 is set...
 */
#define BITSET_TEST(a, b) ((a)[BITSLOT (b)] & BITMASK (b))

/**
 * Calculate the number of slots needed to store nb bits.
 * Used when creating a bitset.
 * @example char bitset[BITSET_SLOTS(100)]; // 100 bits
 */
#define BITSET_SLOTS(nb) ((nb + CHAR_BIT - 1) / CHAR_BIT)