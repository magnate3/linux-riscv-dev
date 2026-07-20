#include "params.h"
#include <linux/sysctl.h>
#include <linux/string.h>

/* Debug mode or not. By default, we disable debug mode */
int PRIO_DWRR_QDISC_DEBUG_MODE = PRIO_DWRR_QDISC_DEBUG_OFF;
/* Buffer management mode: shared (0) or static (1). By default, we enable shread buffer. */
int PRIO_DWRR_QDISC_BUFFER_MODE = PRIO_DWRR_QDISC_SHARED_BUFFER;
/* Per port shared buffer (bytes) */
int PRIO_DWRR_QDISC_SHARED_BUFFER_BYTES = PRIO_DWRR_QDISC_MAX_BUFFER_BYTES;
/* Bucket size in nanosecond. By default, we use 20us for 1G network. */
int PRIO_DWRR_QDISC_BUCKET_NS = 20000;
/* Per port ECN marking threshold (bytes). By default, we use 30KB for 1G network. */
int PRIO_DWRR_QDISC_PORT_THRESH_BYTES = 30000;
/* ECN marking scheme. By default, we use per queue ECN. */
int PRIO_DWRR_QDISC_ECN_SCHEME = PRIO_DWRR_QDISC_QUEUE_ECN;
/* Alpha for quantum sum estimation. It is 0.75 by default. */
int PRIO_DWRR_QDISC_QUANTUM_ALPHA = 750;
/* Alpha for round time estimation. It is 0.75 by default. */
int PRIO_DWRR_QDISC_ROUND_ALPHA = 750;
/* Idle time slot. It is 12us by default */
int PRIO_DWRR_QDISC_IDLE_INTERVAL_NS = 12000;

int PRIO_DWRR_QDISC_DEBUG_MODE_MIN = PRIO_DWRR_QDISC_DEBUG_OFF;
int PRIO_DWRR_QDISC_DEBUG_MODE_MAX = PRIO_DWRR_QDISC_DEBUG_ON;
int PRIO_DWRR_QDISC_BUFFER_MODE_MIN = PRIO_DWRR_QDISC_SHARED_BUFFER;
int PRIO_DWRR_QDISC_BUFFER_MODE_MAX = PRIO_DWRR_QDISC_STATIC_BUFFER;
int PRIO_DWRR_QDISC_ECN_SCHEME_MIN = PRIO_DWRR_QDISC_DISABLE_ECN;
int PRIO_DWRR_QDISC_ECN_SCHEME_MAX = PRIO_DWRR_QDISC_DEQUE_ECN;
int PRIO_DWRR_QDISC_QUANTUM_ALPHA_MIN = 0;
int PRIO_DWRR_QDISC_QUANTUM_ALPHA_MAX = 1000;
int PRIO_DWRR_QDISC_ROUND_ALPHA_MIN = 0;
int PRIO_DWRR_QDISC_ROUND_ALPHA_MAX = 1000;
int PRIO_DWRR_QDISC_DSCP_MIN = 0;
int PRIO_DWRR_QDISC_DSCP_MAX = 63;
int PRIO_DWRR_QDISC_QUANTUM_MIN = PRIO_DWRR_QDISC_MTU_BYTES;
int PRIO_DWRR_QDISC_QUANTUM_MAX = 200*1024;

/* Per queue ECN marking threshold (bytes) */
int PRIO_DWRR_QDISC_QUEUE_THRESH_BYTES[PRIO_DWRR_QDISC_MAX_QUEUES];
/* DSCP value for different queues*/
int PRIO_DWRR_QDISC_QUEUE_DSCP[PRIO_DWRR_QDISC_MAX_QUEUES];
/* Per queue static reserved buffer (bytes) */
int PRIO_DWRR_QDISC_QUEUE_BUFFER_BYTES[PRIO_DWRR_QDISC_MAX_QUEUES];
/* Quantum for different queues*/
int PRIO_DWRR_QDISC_QUEUE_QUANTUM[PRIO_DWRR_QDISC_MAX_DWRR_QUEUES];

/* All parameters that can be configured through sysctl. We have 9+3*PRIO_DWRR_QDISC_MAX_QUEUES+PRIO_DWRR_QDISC_MAX_DWRR_QUEUESS parameters in total. */
struct PRIO_DWRR_QDISC_Param PRIO_DWRR_QDISC_Params[9 + 3 * PRIO_DWRR_QDISC_MAX_QUEUES + PRIO_DWRR_QDISC_MAX_DWRR_QUEUES + 1] =
{
	{"debug_mode", &PRIO_DWRR_QDISC_DEBUG_MODE},
	{"buffer_mode", &PRIO_DWRR_QDISC_BUFFER_MODE},
	{"shared_buffer_bytes", &PRIO_DWRR_QDISC_SHARED_BUFFER_BYTES},
	{"bucket_ns", &PRIO_DWRR_QDISC_BUCKET_NS},
	{"port_thresh_bytes", &PRIO_DWRR_QDISC_PORT_THRESH_BYTES},
	{"ecn_scheme", &PRIO_DWRR_QDISC_ECN_SCHEME},
	{"quantum_alpha", &PRIO_DWRR_QDISC_QUANTUM_ALPHA},
	{"round_alpha", &PRIO_DWRR_QDISC_ROUND_ALPHA},
	{"idle_interval_ns", &PRIO_DWRR_QDISC_IDLE_INTERVAL_NS},
};

struct ctl_table PRIO_DWRR_QDISC_Params_table[9 + 3 * PRIO_DWRR_QDISC_MAX_QUEUES + PRIO_DWRR_QDISC_MAX_DWRR_QUEUES + 1];

struct ctl_path PRIO_DWRR_QDISC_Params_path[] =
{
	{ .procname = "prio_dwrr" },
	{ },
};

struct ctl_table_header *PRIO_DWRR_QDISC_Sysctl = NULL;

int prio_dwrr_qdisc_params_init()
{
	int i = 0;
	memset(PRIO_DWRR_QDISC_Params_table, 0, sizeof(PRIO_DWRR_QDISC_Params_table));

	/* Initialize per-queue ECN marking thresholds, DSCP values and buffer sizes */
	for (i = 0; i < PRIO_DWRR_QDISC_MAX_QUEUES; i++)
	{
		/* Initialize per-queue ECN marking thresholds */
		snprintf(PRIO_DWRR_QDISC_Params[9 + i].name, 63, "queue_thresh_bytes_%d", i);
		PRIO_DWRR_QDISC_Params[9 + i].ptr = &PRIO_DWRR_QDISC_QUEUE_THRESH_BYTES[i];
		PRIO_DWRR_QDISC_QUEUE_THRESH_BYTES[i] = PRIO_DWRR_QDISC_PORT_THRESH_BYTES;

		/* Initialize per-queue DSCP values */
		snprintf(PRIO_DWRR_QDISC_Params[9 + i + PRIO_DWRR_QDISC_MAX_QUEUES].name, 63, "queue_dscp_%d", i);
		PRIO_DWRR_QDISC_Params[9 + i + PRIO_DWRR_QDISC_MAX_QUEUES].ptr = &PRIO_DWRR_QDISC_QUEUE_DSCP[i];
		PRIO_DWRR_QDISC_QUEUE_DSCP[i] = i;

		/* Initialize per-queue buffer sizes */
		snprintf(PRIO_DWRR_QDISC_Params[9 + i + 2 * PRIO_DWRR_QDISC_MAX_QUEUES].name, 63, "queue_buffer_bytes_%d", i);
		PRIO_DWRR_QDISC_Params[9 + i + 2 * PRIO_DWRR_QDISC_MAX_QUEUES].ptr = &PRIO_DWRR_QDISC_QUEUE_BUFFER_BYTES[i];
		PRIO_DWRR_QDISC_QUEUE_BUFFER_BYTES[i] = PRIO_DWRR_QDISC_MAX_BUFFER_BYTES;
	}

	/* Initialize per-dwrr-queue quantum */
	for (i = 0; i < PRIO_DWRR_QDISC_MAX_DWRR_QUEUES; i++)
	{
		snprintf(PRIO_DWRR_QDISC_Params[9 + i + 3 * PRIO_DWRR_QDISC_MAX_QUEUES].name, 63, "queue_quantum_%d", i + PRIO_DWRR_QDISC_MAX_PRIO_QUEUES);
		PRIO_DWRR_QDISC_Params[9 + i + 3 * PRIO_DWRR_QDISC_MAX_QUEUES].ptr = &PRIO_DWRR_QDISC_QUEUE_QUANTUM[i];
		PRIO_DWRR_QDISC_QUEUE_QUANTUM[i] = PRIO_DWRR_QDISC_MTU_BYTES;
	}

	/* End of the parameters */
	PRIO_DWRR_QDISC_Params[9 + 3 * PRIO_DWRR_QDISC_MAX_QUEUES + PRIO_DWRR_QDISC_MAX_DWRR_QUEUES].ptr = NULL;

	for (i = 0; i < 9 + 3 * PRIO_DWRR_QDISC_MAX_QUEUES + PRIO_DWRR_QDISC_MAX_DWRR_QUEUES + 1; i++)
	{
		struct ctl_table *entry = &PRIO_DWRR_QDISC_Params_table[i];

		/* End */
		if (PRIO_DWRR_QDISC_Params[i].ptr == NULL)
			break;

		/* Initialize entry (ctl_table) */
		entry->procname = PRIO_DWRR_QDISC_Params[i].name;
		entry->data = PRIO_DWRR_QDISC_Params[i].ptr;
		entry->mode = 0644;

		/* debug mode */
		if (i == 0)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_DEBUG_MODE_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_DEBUG_MODE_MAX;
		}
		/* buffer mode */
		else if (i == 1)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_BUFFER_MODE_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_BUFFER_MODE_MAX;
		}
		/* ECN marking scheme */
		else if (i == 5)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_ECN_SCHEME_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_ECN_SCHEME_MAX;
		}
		/* quantum_alpha*/
		else if (i == 6)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_QUANTUM_ALPHA_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_QUANTUM_ALPHA_MAX;
		}
		/* round_alpha */
		else if (i == 7)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_ROUND_ALPHA_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_ROUND_ALPHA_MAX;
		}
		/* per-queue DSCP */
		else if (i >= 9 + PRIO_DWRR_QDISC_MAX_QUEUES && i < 9 + 2 * PRIO_DWRR_QDISC_MAX_QUEUES)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_DSCP_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_DSCP_MAX;
		}
		/* per-dwrr-queue quantums */
		else if (i >= 9 + 3 * PRIO_DWRR_QDISC_MAX_QUEUES)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_DWRR_QDISC_QUANTUM_MIN;
			entry->extra2 = &PRIO_DWRR_QDISC_QUANTUM_MAX;
		}
		/* per-queue marking thresholds and buffer sizes */
		else
		{
			entry->proc_handler = &proc_dointvec;
		}
		entry->maxlen=sizeof(int);
	}

	PRIO_DWRR_QDISC_Sysctl = register_sysctl_paths(PRIO_DWRR_QDISC_Params_path, PRIO_DWRR_QDISC_Params_table);

	if (likely(PRIO_DWRR_QDISC_Sysctl))
		return 0;
	else
		return -1;

}

void prio_dwrr_qdisc_params_exit()
{
	if (likely(PRIO_DWRR_QDISC_Sysctl))
		unregister_sysctl_table(PRIO_DWRR_QDISC_Sysctl);
}
