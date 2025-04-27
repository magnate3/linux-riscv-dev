#pragma once

#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "common.h"

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) > (b) ? (b) : (a))

/**
 * Retrieve the current time in nanoseconds.
 * @return The current time in nanoseconds.
 */
uint64_t get_time_ns (void);

void pp_sleep (uint64_t ns);

/**
 * Print a hex dump of the given data.
 *
 * @param data the data to print
 * @param size the size of the data
 */
void hex_dump (const void *data, size_t size);