#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <linux/types.h>

/* Our module has 8 queues by default */
#define PRIO_QDISC_MAX_QUEUES 8
/* Ethernet packets with less than the minimum 64 bytes (header (14B) + user data + FCS (4B)) are padded to 64 bytes. */
#define PRIO_QDISC_MIN_PKT_BYTES 64
/* Maximum (per queue/per port shared) buffer size (2MB)*/
#define PRIO_QDISC_MAX_BUFFER_BYTES 2000000
/* Debug mode is off */
#define	PRIO_QDISC_DEBUG_OFF 0
/* Debug mode is on */
#define	PRIO_QDISC_DEBUG_ON 1

/* Shared buffer management policy. All the queues in a port share the same buffer region in first-in-fist-serve bias. */
#define	PRIO_QDISC_SHARED_BUFFER 0
/* Static buffer management policy. Each queue has its own static reserved buffer. */
#define	PRIO_QDISC_STATIC_BUFFER 1

/* Disable ECN marking */
#define	PRIO_QDISC_DISABLE_ECN 0
/* Per queue ECN marking */
#define	PRIO_QDISC_QUEUE_ECN 1
/* Per port ECN marking */
#define PRIO_QDISC_PORT_ECN 2
/* Dequeue latency-based ECN marking. This is a general ECN marking approach for any packet scheduler */
#define PRIO_QDISC_DEQUE_ECN 3

/* Debug mode or not */
extern int PRIO_QDISC_DEBUG_MODE;
/* Buffer management mode: shared (0) or static (1)*/
extern int PRIO_QDISC_BUFFER_MODE;
/* Per port shared buffer (bytes) */
extern int PRIO_QDISC_SHARED_BUFFER_BYTES;
/* Bucket size in nanosecond*/
extern int PRIO_QDISC_BUCKET_NS;
/* Per port ECN marking threshold (bytes) */
extern int PRIO_QDISC_PORT_THRESH_BYTES;
/* ECN marking scheme */
extern int PRIO_QDISC_ECN_SCHEME;

/* Per queue ECN marking threshold (bytes) */
extern int PRIO_QDISC_QUEUE_THRESH_BYTES[PRIO_QDISC_MAX_QUEUES];
/* DSCP value for different queues */
extern int PRIO_QDISC_QUEUE_DSCP[PRIO_QDISC_MAX_QUEUES];
/* Per queue static reserved buffer (bytes) */
extern int PRIO_QDISC_QUEUE_BUFFER_BYTES[PRIO_QDISC_MAX_QUEUES];

struct PRIO_QDISC_Param
{
	char name[64];
	int *ptr;
};

extern struct PRIO_QDISC_Param PRIO_QDISC_Params[6 + 3 * PRIO_QDISC_MAX_QUEUES + 1];

/* Intialize parameters and register sysctl */
int prio_qdisc_params_init(void);
/* Unregister sysctl */
void prio_qdisc_params_exit(void);

#endif
