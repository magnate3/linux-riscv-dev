#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <linux/types.h>

/*
 * CoDel uses a 1024 nsec clock, encoded in u32
 * This gives a range of 2199 seconds, because of signed compares
 */
typedef u32 codel_time_t;
typedef s32 codel_tdiff_t;

/* Dealing with timer wrapping, according to RFC 1982, as desc in wikipedia:
 *  https://en.wikipedia.org/wiki/Serial_number_arithmetic#General_Solution
 * codel_time_after(a,b) returns true if the time a is after time b.
 */
#define codel_time_after(a, b)						\
	(typecheck(codel_time_t, a) &&					\
	 typecheck(codel_time_t, b) &&					\
	 ((s32)((a) - (b)) > 0))
#define codel_time_before(a, b) 	codel_time_after(b, a)

#define codel_time_after_eq(a, b)					\
	(typecheck(codel_time_t, a) &&					\
	 typecheck(codel_time_t, b) &&					\
	 ((s32)((a) - (b)) >= 0))
#define codel_time_before_eq(a, b)	codel_time_after_eq(b, a)


/* Our module has at most 8 queues */
#define wfq_max_queues 8
/* Our module supports at most 8 priorities */
#define wfq_max_prio 8

/*
 * 1538 = MTU (1500B) + Ethernet header(14B) + Frame check sequence (4B) +
 * Frame check sequence(8B) + Interpacket gap(12B)
 */
#define wfq_max_pkt_bytes 1538
/*
 * Ethernet packets with less than the minimum 64 bytes
 * (header (14B) + user data + FCS (4B)) are padded to 64 bytes.
 */
#define wfq_min_pkt_bytes 64
/* Maximum (per queue/per port shared) buffer size (2MB) */
#define wfq_max_buffer_bytes 2000000
/* Per port shared buffer management policy */
#define	wfq_shared_buffer 0
/* Per port static buffer management policy */
#define	wfq_static_buffer 1

/* Disable ECN marking */
#define	wfq_disable_ecn 0
/* Per queue ECN marking */
#define	wfq_queue_ecn 1
/* Per port ECN marking */
#define wfq_port_ecn 2
/* MQ-ECN. Note that MQ-ECN cannot support WFQ. It actually has no use here. */
#define wfq_mq_ecn 3
/* TCN */
#define wfq_tcn 4
/* CoDel */
#define wfq_codel 5

/* For CoDel timestamp */
#define wfq_codel_shift 10

#define wfq_disable 0
#define wfq_enable 1

/* The number of global (rather than 'per-queue') parameters */
#define wfq_global_params 10
/* The number of parameters for each queue */
#define wfq_queue_params 5
/* The total number of parameters (per-queue and global parameters) */
#define wfq_total_params (wfq_global_params + wfq_queue_params * wfq_max_queues)

/* Global parameters */
/* Enable debug mode or not */
extern int wfq_enable_debug;
/* Buffer management mode: shared (0) or static (1)*/
extern int wfq_buffer_mode;
/* Per port shared buffer (bytes) */
extern int wfq_shared_buffer_bytes;
/* Bucket size in bytes*/
extern int wfq_bucket_bytes;
/* Per port ECN marking threshold (bytes) */
extern int wfq_port_thresh_bytes;
/* ECN marking scheme */
extern int wfq_ecn_scheme;
/* Enable dequeue ECN marking or not */
extern int wfq_enable_dequeue_ecn;
/* TCN threshold (1024 nanoseconds) */
extern int wfq_tcn_thresh;
/* CoDel target (1024 nanoseconds) */
extern int wfq_codel_target;
/* CoDel interval (1024 nanoseconds) */
extern int wfq_codel_interval;

/* Per-queue parameters */
/* Per queue ECN marking threshold (bytes) */
extern int wfq_queue_thresh_bytes[wfq_max_queues];
/* DSCP value for different queues */
extern int wfq_queue_dscp[wfq_max_queues];
/* Weight for different queues*/
extern int wfq_queue_weight[wfq_max_queues];
/* Per queue static reserved buffer (bytes) */
extern int wfq_queue_buffer_bytes[wfq_max_queues];
/* Per queue priority (0 to wfq_max_prio - 1) */
extern int wfq_queue_prio[wfq_max_queues];

struct wfq_param
{
	char name[64];
	int *ptr;
};

extern struct wfq_param wfq_params[wfq_total_params + 1];

/* Intialize parameters and register sysctl */
bool wfq_params_init(void);
/* Unregister sysctl */
void wfq_params_exit(void);

#endif
