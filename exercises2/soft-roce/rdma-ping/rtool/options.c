/*
 *  Copyright (c) 2017, Mellanox Technologies. All rights reserved.
 *  Copyright (C) 2011-2015 Konstantin Khlebnikov <koct9i@gmail.com>
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <time.h>
#include <limits.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <unistd.h>
#include <hugetlbfs.h>
#include <sys/time.h>
#include <malloc.h>
#include <inttypes.h>

#define APP_PREFIX "rdma_mr_lat: "

#define NSEC_PER_SEC	1000000000ll

void err(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	fprintf(stderr, APP_PREFIX);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, ": %s\n", strerror(errno));
	va_end(ap);
	exit(eval);
}

void errx(int eval, const char *fmt, ...)
{
	va_list ap;

	va_start(ap, fmt);
	fprintf(stderr, APP_PREFIX);
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
	exit(eval);
}

#ifdef HAVE_CLOCK_GETTIME
long long current_time(void)
{
	struct timespec ts;

	if (clock_gettime(CLOCK_MONOTONIC, &ts))
		err(3, "clock_gettime failed");

	return ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

#else

long long current_time(void)
{
	struct timeval tv;

	if (gettimeofday(&tv, NULL))
		err(3, "gettimeofday failed");

	return tv.tv_sec * NSEC_PER_SEC + tv.tv_usec * 1000ll;
}

#endif /* HAVE_CLOCK_GETTIME */

struct suffix {
	const char	*txt;
	long long	mul;
};

static struct suffix int_suffix[] = {
	{ "T",		1000000000000ll },
	{ "G",		1000000000ll },
	{ "M",		1000000ll },
	{ "k",		1000ll },
	{ "",		1ll },
	{ "da",		10ll },
	{ "P",		1000000000000000ll },
	{ "E",		1000000000000000000ll },
	{ NULL,		0ll },
};

static struct suffix size_suffix[] = {
	/* These are first match for printing */
	{ "PiB",	1ll<<50 },
	{ "TiB",	1ll<<40 },
	{ "GiB",	1ll<<30 },
	{ "MiB",	1ll<<20 },
	{ "KiB",	1ll<<10 },
	{ "B",		1 },
	{ "",		1 },
	/* Should be decimal, keep binary for compatibility */
	{ "k",		1ll<<10 },
	{ "kb",		1ll<<10 },
	{ "m",		1ll<<20 },
	{ "mb",		1ll<<20 },
	{ "g",		1ll<<30 },
	{ "gb",		1ll<<30 },
	{ "t",		1ll<<40 },
	{ "tb",		1ll<<40 },
	{ "pb",		1ll<<50 },
	{ "eb",		1ll<<60 },
	{ "sector",	512 },
	{ "page",	4096 },
	{ NULL,		0ll },
};

static struct suffix time_suffix[] = {
	{ "hour",	NSEC_PER_SEC * 60 * 60 },
	{ "min",	NSEC_PER_SEC * 60 },
	{ "s",		NSEC_PER_SEC },
	{ "ms",		1000000ll },
	{ "us",		1000ll },
	{ "ns",		1ll },
	{ "nsec",	1ll },
	{ "usec",	1000ll },
	{ "msec",	1000000ll },
	{ "",		NSEC_PER_SEC },
	{ "sec",	NSEC_PER_SEC },
	{ "m",		NSEC_PER_SEC * 60 },
	{ "h",		NSEC_PER_SEC * 60 * 60 },
	{ NULL,		0ll },
};

long long parse_suffix(const char *str, struct suffix *sfx,
		       long long min, long long max)
{
	char *end;
	double val, den;

	val = strtod(str, &end);
	if (*end == '/') {
		if (end == str)
			val = 1;
		den = strtod(end + 1, &end);
		if (!den)
			errx(1, "division by zero in parsing argument: %s", str);
		val /= den;
	}
	for ( ; sfx->txt ; sfx++ ) {
		if (strcasecmp(end, sfx->txt))
			continue;
		val *= sfx->mul;
		if (val < min || val > max)
			errx(1, "integer overflow at parsing argument: %s", str);
		return val;
	}
	errx(1, "invalid suffix: \"%s\"", end);
	return 0;
}

int parse_int(const char *str)
{
	return parse_suffix(str, int_suffix, 0, INT_MAX);
}

ssize_t parse_size(const char *str)
{
	return parse_suffix(str, size_suffix, 0, LONG_MAX);
}

off_t parse_offset(const char *str)
{
	return parse_suffix(str, size_suffix, 0, LLONG_MAX);
}

long long parse_time(const char *str)
{
	return parse_suffix(str, time_suffix, 0, LLONG_MAX);
}

long long parse_time_seconds(const char *str)
{
	return parse_suffix(str, time_suffix, 0, LLONG_MAX) / NSEC_PER_SEC;
}

void print_suffix(long long val, struct suffix *sfx)
{
	int precision;

	while (val < sfx->mul && sfx->mul > 1)
		sfx++;

	if (val % sfx->mul == 0)
		precision = 0;
	else if (val >= sfx->mul * 10)
		precision = 1;
	else
		precision = 2;

	printf("%.*f", precision, val * 1.0 / sfx->mul);
	if (*sfx->txt)
		printf(" %s", sfx->txt);
}

void print_int(long long val)
{
	print_suffix(val, int_suffix);
}

void print_size(long long val)
{
	print_suffix(val, size_suffix);
}

void print_time(long long val)
{
	print_suffix(val, time_suffix);
}


