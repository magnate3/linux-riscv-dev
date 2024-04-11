/*
 * util.h -- set of various support routines.
 *
 * Copyright (c) 2001-2006, NLnet Labs.
 *
 * Modified Work Copyright (c) 2018 The TIGLabs Authors.
 *
 */

#ifndef _UTIL_H_
#define _UTIL_H_

#include <sys/time.h>
#include <stdarg.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <sys/socket.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <linux/if_ether.h>
#include "nat-conf.h"
struct rr;
struct buffer;

#define LOG_ERR 1
#define LOG_INFO 2

#define ALIGN_UP(n, alignment)  \
	(((n) + (alignment) - 1) & (~((alignment) - 1)))
#define PADDING(n, alignment)   \
	(ALIGN_UP((n), (alignment)) - (n))

#define ATTR_FORMAT(archetype, string_index, first_to_check) \
        __attribute__ ((format (archetype, string_index, first_to_check)))


typedef enum {
    FWD_MODE_TYPE_DISABLE,
    FWD_MODE_TYPE_DIRECT,
    FWD_MODE_TYPE_CACHE,
    FWD_MODE_TYPE_MAX,
} fwd_mode_type;

#if 0
static const char *fwd_mode_str_array[FWD_MODE_TYPE_MAX] = {
    "disable",
    "direct",
    "cache"
};
#endif
void log_open( char *ident);

void log_msg(int priority, const char *format, ...)
	ATTR_FORMAT(printf, 2, 3);

int log_file_reload(char *filename);

void *xalloc(size_t size);
void *xalloc_zero(size_t size);
void *xalloc_array_zero(size_t num, size_t size);
void *xrealloc(void *ptr, size_t size);

uint32_t strtoserial(const char *nptr, const char **endptr);
size_t strlcpy(char *dst, const char *src, size_t siz);


/*
 * Convert binary data to a string of hexadecimal characters.
 */
ssize_t hex_ntop(uint8_t const *src, size_t srclength, char *target,
		 size_t targsize);
ssize_t hex_pton(const char* src, uint8_t* target, size_t targsize);


/*
 * Convert a single (hexadecimal) digit to its integer value.
 */
int hexdigit_to_int(char ch);

int linux_set_if_mac(const char *ifname, const unsigned char mac[ETH_ALEN]);

int config_file_load(struct nat_config *cfg, const char *cfgfile_path, char *proc_name);
#endif /* _UTIL_H_ */
