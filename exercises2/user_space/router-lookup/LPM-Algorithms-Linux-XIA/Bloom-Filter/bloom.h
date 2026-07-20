/* Copyright @2012 by Justin Hines at Bitly under a very liberal license.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * 2015, LPM Algorithms for Linux XIA
 * Garnaik Sumeet, Michel Machado
 */

#ifndef __BLOOM_H__
#define __BLOOM_H__
#include <stdint.h>
#include <stdlib.h>

typedef struct {
	size_t bytes;
	char *array;
} bitmap_t;

bitmap_t *bitmap_resize(bitmap_t *bitmap, size_t old_size, size_t new_size);
bitmap_t *new_bitmap(size_t bytes);

int bitmap_increment(bitmap_t *bitmap, unsigned int index, long offset);
int bitmap_decrement(bitmap_t *bitmap, unsigned int index, long offset);
int bitmap_check(bitmap_t *bitmap, unsigned int index, long offset);

void free_bitmap(bitmap_t *bitmap);

typedef struct {
	uint64_t id;
	uint32_t count;
	uint32_t _pad;
} counting_bloom_header_t;

typedef struct {
	counting_bloom_header_t *header;
	unsigned int capacity;
	long offset;
	unsigned int counts_per_func;
	uint32_t *hashes;
	size_t nfuncs;
	size_t size;
	size_t num_bytes;
	double error_rate;
	bitmap_t *bitmap;
} counting_bloom_t;

int free_counting_bloom(counting_bloom_t *bloom);
counting_bloom_t *new_counting_bloom(unsigned int capacity, double error_rate);
int counting_bloom_add(counting_bloom_t *bloom, const char *s, size_t len);
int counting_bloom_remove(counting_bloom_t *bloom, const char *s, size_t len);
int counting_bloom_check(counting_bloom_t *bloom, const char *s, size_t len);

#endif
