#ifndef __PARAMS_H__
#define __PARAMS_H__

#include <linux/types.h>

/* Our module has 1 high priority queue(s) */
#define PRIO_DWRR_QDISC_MAX_PRIO_QUEUES 1
/* Our module has 7 DWRR queues at the lowest priority */
#define PRIO_DWRR_QDISC_MAX_DWRR_QUEUES 7
/* Our module has 8 queues in total */
#define PRIO_DWRR_QDISC_MAX_QUEUES (PRIO_DWRR_QDISC_MAX_PRIO_QUEUES + PRIO_DWRR_QDISC_MAX_DWRR_QUEUES)
/* MTU(1500B)+Ethernet header(14B)+Frame check sequence (4B)+Frame check sequence(8B)+Interpacket gap(12B) */
#define PRIO_DWRR_QDISC_MTU_BYTES 1538
/* Ethernet packets with less than the minimum 64 bytes (header (14B) + user data + FCS (4B)) are padded to 64 bytes. */
#define PRIO_DWRR_QDISC_MIN_PKT_BYTES 64
/* Maximum (per queue/per port shared) buffer size (2MB)*/
#define PRIO_DWRR_QDISC_MAX_BUFFER_BYTES 2000000

/* Debug mode is off */
#define	PRIO_DWRR_QDISC_DEBUG_OFF 0
/* Debug mode is on */
#define	PRIO_DWRR_QDISC_DEBUG_ON 1

/* Per port shared buffer management policy */
#define	PRIO_DWRR_QDISC_SHARED_BUFFER 0
/* Per port static buffer management policy */
#define	PRIO_DWRR_QDISC_STATIC_BUFFER 1

/* Disable ECN marking */
#define	PRIO_DWRR_QDISC_DISABLE_ECN 0
/* Per queue ECN marking */
#define	PRIO_DWRR_QDISC_QUEUE_ECN 1
/* Per port ECN marking */
#define PRIO_DWRR_QDISC_PORT_ECN 2
/* MQ-ECN for any packet scheduling algorithm */
#define PRIO_DWRR_QDISC_MQ_ECN_GENER 3
/* MQ-ECN for round-robin packet scheduling algorithms */
#define PRIO_DWRR_QDISC_MQ_ECN_RR 4
/* Dequeue latency-based ECN marking. This is a general ECN marking approach for any packet scheduler */
#define PRIO_DWRR_QDISC_DEQUE_ECN 5

#define PRIO_DWRR_QDISC_MAX_ITERATION 10

/* Debug mode or not */
extern int PRIO_DWRR_QDISC_DEBUG_MODE;
/* Buffer management mode: shared (0) or static (1)*/
extern int PRIO_DWRR_QDISC_BUFFER_MODE;
/* Per port shared buffer (bytes) */
extern int PRIO_DWRR_QDISC_SHARED_BUFFER_BYTES;
/* Bucket size in nanosecond*/
extern int PRIO_DWRR_QDISC_BUCKET_NS;
/* Per port ECN marking threshold (bytes) */
extern int PRIO_DWRR_QDISC_PORT_THRESH_BYTES;
/* ECN marking scheme */
extern int PRIO_DWRR_QDISC_ECN_SCHEME;
/* Alpha for quantum sum estimation */
extern int PRIO_DWRR_QDISC_QUANTUM_ALPHA;
/* Alpha for round time estimation */
extern int PRIO_DWRR_QDISC_ROUND_ALPHA;
/* Idle time interval */
extern int PRIO_DWRR_QDISC_IDLE_INTERVAL_NS;

/* Per queue ECN marking threshold (bytes) */
extern int PRIO_DWRR_QDISC_QUEUE_THRESH_BYTES[PRIO_DWRR_QDISC_MAX_QUEUES];
/* Per queue DSCP value */
extern int PRIO_DWRR_QDISC_QUEUE_DSCP[PRIO_DWRR_QDISC_MAX_QUEUES];
/* Per queue static reserved buffer (bytes) */
extern int PRIO_DWRR_QDISC_QUEUE_BUFFER_BYTES[PRIO_DWRR_QDISC_MAX_QUEUES];
/* Quantum for DWRR queues*/
extern int PRIO_DWRR_QDISC_QUEUE_QUANTUM[PRIO_DWRR_QDISC_MAX_DWRR_QUEUES];

struct PRIO_DWRR_QDISC_Param
{
	char name[64];
	int *ptr;
};

extern struct PRIO_DWRR_QDISC_Param PRIO_DWRR_QDISC_Params[9 + 3 * PRIO_DWRR_QDISC_MAX_QUEUES + PRIO_DWRR_QDISC_MAX_DWRR_QUEUES + 1];

/* Intialize parameters and register sysctl */
int prio_dwrr_qdisc_params_init(void);
/* Unregister sysctl */
void prio_dwrr_qdisc_params_exit(void);

#endif
