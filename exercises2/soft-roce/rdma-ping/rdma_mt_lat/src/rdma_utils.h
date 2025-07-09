/*
 * Copyright (c) 2005 Mellanox Technologies. All rights reserved.
 *
 */

#ifndef VL_SOCK_H
#define VL_SOCK_H



#include <netinet/in.h>
#include <arpa/inet.h>
#include <byteswap.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined (__x86_64__) || defined(__i386__)
/* Note: only x86 CPUs which have rdtsc instruction are supported. */
typedef unsigned long long cycles_t;
static inline cycles_t get_cycles()
{
	unsigned low, high;
	unsigned long long val;
	asm volatile ("rdtsc" : "=a" (low), "=d" (high));
	val = high;
	val = (val << 32) | low;
	return val;
}
#elif defined(__PPC__) || defined(__PPC64__)
/* Note: only PPC CPUs which have mftb instruction are supported. */
/* PPC64 has mftb */
typedef unsigned long cycles_t;
static inline cycles_t get_cycles()
{
	cycles_t ret;

	asm volatile ("mftb %0" : "=r" (ret) : );
	return ret;
}
#elif defined(__ia64__)
/* Itanium2 and up has ar.itc (Itanium1 has errata) */
typedef unsigned long cycles_t;
static inline cycles_t get_cycles()
{
	cycles_t ret;

	asm volatile ("mov %0=ar.itc" : "=r" (ret));
	return ret;
}

#else
#warning get_cycles not implemented for this architecture: attempt asm/timex.h
#include <asm/timex.h>
#endif

extern double get_cpu_mhz(int);


#define IP_STR_LENGTH 15
#define SOCKET_RETRY  10

#if __BYTE_ORDER == __LITTLE_ENDIAN
#define ntohll(x) bswap_64(x)
#define htonll(x) bswap_64(x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#define ntohll(x) x
#define htonll(x) x
#else
#error __BYTE_ORDER is neither __LITTLE_ENDIAN nor __BIG_ENDIAN
#endif

struct sock_t{
    int            sock_fd;
    int            is_daemon;
    unsigned short port;
    char           ip[IP_STR_LENGTH+1];

};

struct sock_bind_t {
    unsigned int counter;
    int          socket_fd;
    int            is_daemon;
    unsigned short port;
    char           ip[IP_STR_LENGTH+1];
};

void bind_init(struct sock_bind_t *);
void sock_init(struct sock_t *);
int sock_connect(struct sock_bind_t *, struct sock_t *);
int sock_connect_multi(struct sock_bind_t *, struct sock_t *);
int sock_close(struct sock_t *);
int sock_close_multi(struct sock_t *, struct sock_bind_t *);
int sock_sync_data(struct sock_t *, unsigned int , void * , void *);
int sock_sync_ready(struct sock_t *);
int sock_d2c(struct sock_t *, unsigned int , void*);
int sock_c2d(struct sock_t *, unsigned int , void*);
int close_sock_fd(int);

#ifdef __cplusplus
}
#endif

#endif
