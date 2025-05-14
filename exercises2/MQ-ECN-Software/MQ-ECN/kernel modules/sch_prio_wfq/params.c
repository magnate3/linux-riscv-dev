#include <linux/sysctl.h>
#include <linux/string.h>

#include "params.h"

/* Debug mode or not. By default, we disable debug mode */
int PRIO_WFQ_QDISC_DEBUG_MODE = PRIO_WFQ_QDISC_DEBUG_OFF;
/* Buffer management mode: shared (0) or static (1). By default, we enable shread buffer. */
int PRIO_WFQ_QDISC_BUFFER_MODE = PRIO_WFQ_QDISC_SHARED_BUFFER;
/* Per port shared buffer (bytes) */
int PRIO_WFQ_QDISC_SHARED_BUFFER_BYTES = PRIO_WFQ_QDISC_MAX_BUFFER_BYTES;
/* Bucket size in nanosecond. By default, we use 25us for 1G network. */
int PRIO_WFQ_QDISC_BUCKET_NS = 25000;
/* Per port ECN marking threshold (bytes). By default, we use 32KB for 1G network. */
int PRIO_WFQ_QDISC_PORT_THRESH_BYTES = 32000;
/* ECN marking scheme. By default, we use per queue ECN. */
int PRIO_WFQ_QDISC_ECN_SCHEME = PRIO_WFQ_QDISC_QUEUE_ECN;

int PRIO_WFQ_QDISC_DEBUG_MODE_MIN = PRIO_WFQ_QDISC_DEBUG_OFF;
int PRIO_WFQ_QDISC_DEBUG_MODE_MAX = PRIO_WFQ_QDISC_DEBUG_ON;
int PRIO_WFQ_QDISC_BUFFER_MODE_MIN = PRIO_WFQ_QDISC_SHARED_BUFFER;
int PRIO_WFQ_QDISC_BUFFER_MODE_MAX = PRIO_WFQ_QDISC_STATIC_BUFFER;
int PRIO_WFQ_QDISC_ECN_SCHEME_MIN = PRIO_WFQ_QDISC_DISABLE_ECN;
int PRIO_WFQ_QDISC_ECN_SCHEME_MAX = PRIO_WFQ_QDISC_DEQUE_ECN;
int PRIO_WFQ_QDISC_DSCP_MIN = 0;
int PRIO_WFQ_QDISC_DSCP_MAX = 63;
int PRIO_WFQ_QDISC_WEIGHT_MIN = 1;
int PRIO_WFQ_QDISC_WEIGHT_MAX = PRIO_WFQ_QDISC_MIN_PKT_BYTES;

/* Per queue ECN marking threshold (bytes) */
int PRIO_WFQ_QDISC_QUEUE_THRESH_BYTES[PRIO_WFQ_QDISC_MAX_QUEUES];
/* DSCP value for different queues */
int PRIO_WFQ_QDISC_QUEUE_DSCP[PRIO_WFQ_QDISC_MAX_QUEUES];
/* Per queue static reserved buffer (bytes) */
int PRIO_WFQ_QDISC_QUEUE_BUFFER_BYTES[PRIO_WFQ_QDISC_MAX_QUEUES];
/* Weights for different WFQ queues*/
int PRIO_WFQ_QDISC_QUEUE_WEIGHT[PRIO_WFQ_QDISC_MAX_WFQ_QUEUES];


/* All parameters that can be configured through sysctl. We have 6 + 3 * PRIO_WFQ_QDISC_MAX_QUEUES + PRIO_WFQ_QDISC_MAX_WFQ_QUEUES in total. */
struct PRIO_WFQ_QDISC_Param PRIO_WFQ_QDISC_Params[6 + 3 * PRIO_WFQ_QDISC_MAX_QUEUES + PRIO_WFQ_QDISC_MAX_WFQ_QUEUES + 1] =
{
	{"debug_mode", &PRIO_WFQ_QDISC_DEBUG_MODE},
	{"buffer_mode",&PRIO_WFQ_QDISC_BUFFER_MODE},
	{"shared_buffer_bytes", &PRIO_WFQ_QDISC_SHARED_BUFFER_BYTES},
	{"bucket_ns", &PRIO_WFQ_QDISC_BUCKET_NS},
	{"port_thresh_bytes", &PRIO_WFQ_QDISC_PORT_THRESH_BYTES},
	{"ecn_scheme", &PRIO_WFQ_QDISC_ECN_SCHEME},
};

struct ctl_table PRIO_WFQ_QDISC_Params_table[6 + 3 * PRIO_WFQ_QDISC_MAX_QUEUES + PRIO_WFQ_QDISC_MAX_WFQ_QUEUES + 1];

struct ctl_path PRIO_WFQ_QDISC_Params_path[] =
{
	{ .procname = "prio_wfq" },
	{ },
};

struct ctl_table_header *PRIO_WFQ_QDISC_Sysctl = NULL;

int prio_wfq_qdisc_params_init()
{
    int i;
	memset(PRIO_WFQ_QDISC_Params_table, 0, sizeof(PRIO_WFQ_QDISC_Params_table));

	/* Initialize per-queue ECN marking thresholds, DSCP values and buffer sizes */
	for (i = 0; i < PRIO_WFQ_QDISC_MAX_QUEUES; i++)
	{
		/* Initialize per-queue ECN marking thresholds */
		snprintf(PRIO_WFQ_QDISC_Params[6 + i].name, 63, "queue_thresh_bytes_%d", i);
		PRIO_WFQ_QDISC_Params[6 + i].ptr = &PRIO_WFQ_QDISC_QUEUE_THRESH_BYTES[i];
		PRIO_WFQ_QDISC_QUEUE_THRESH_BYTES[i] = PRIO_WFQ_QDISC_PORT_THRESH_BYTES;

		/* Initialize per-queue DSCP values */
		snprintf(PRIO_WFQ_QDISC_Params[6 + i + PRIO_WFQ_QDISC_MAX_QUEUES].name, 63, "queue_dscp_%d", i);
		PRIO_WFQ_QDISC_Params[6 + i + PRIO_WFQ_QDISC_MAX_QUEUES].ptr = &PRIO_WFQ_QDISC_QUEUE_DSCP[i];
		PRIO_WFQ_QDISC_QUEUE_DSCP[i] = i;

		/* Initialize per-queue buffer sizes */
		snprintf(PRIO_WFQ_QDISC_Params[6 + i + 2 * PRIO_WFQ_QDISC_MAX_QUEUES].name, 63, "queue_buffer_bytes_%d", i);
		PRIO_WFQ_QDISC_Params[6 + i + 2 * PRIO_WFQ_QDISC_MAX_QUEUES].ptr = &PRIO_WFQ_QDISC_QUEUE_BUFFER_BYTES[i];
		PRIO_WFQ_QDISC_QUEUE_BUFFER_BYTES[i] = PRIO_WFQ_QDISC_MAX_BUFFER_BYTES;
	}

	/* Initialize per-wfq-queue weight */
	for (i = 0; i < PRIO_WFQ_QDISC_MAX_WFQ_QUEUES; i++)
	{
		snprintf(PRIO_WFQ_QDISC_Params[6 + i + 3 * PRIO_WFQ_QDISC_MAX_QUEUES].name, 63, "queue_weight_%d", i + PRIO_WFQ_QDISC_MAX_PRIO_QUEUES);
		PRIO_WFQ_QDISC_Params[6 + i + 3 * PRIO_WFQ_QDISC_MAX_QUEUES].ptr = &PRIO_WFQ_QDISC_QUEUE_WEIGHT[i];
		PRIO_WFQ_QDISC_QUEUE_WEIGHT[i] = 1;
	}
	/* End of the parameters */
	PRIO_WFQ_QDISC_Params[6 + 3 * PRIO_WFQ_QDISC_MAX_QUEUES + PRIO_WFQ_QDISC_MAX_WFQ_QUEUES].ptr = NULL;

    for (i = 0; i < 6 + 3 * PRIO_WFQ_QDISC_MAX_QUEUES + PRIO_WFQ_QDISC_MAX_WFQ_QUEUES; i++)
    {
        struct ctl_table *entry = &PRIO_WFQ_QDISC_Params_table[i];

        /* Initialize entry (ctl_table) */
        entry->procname = PRIO_WFQ_QDISC_Params[i].name;
        entry->data = PRIO_WFQ_QDISC_Params[i].ptr;
        entry->mode = 0644;

        /* debug mode */
        if (i == 0)
        {
            entry->proc_handler = &proc_dointvec_minmax;
            entry->extra1 = &PRIO_WFQ_QDISC_DEBUG_MODE_MIN;
            entry->extra2 = &PRIO_WFQ_QDISC_DEBUG_MODE_MAX;
        }
        /* buffer mode */
        else if (i == 1)
        {
            entry->proc_handler = &proc_dointvec_minmax;
            entry->extra1 = &PRIO_WFQ_QDISC_BUFFER_MODE_MIN;
            entry->extra2 = &PRIO_WFQ_QDISC_BUFFER_MODE_MAX;
        }
        /* ECN marking scheme */
        else if (i == 5)
        {
            entry->proc_handler = &proc_dointvec_minmax;
            entry->extra1 = &PRIO_WFQ_QDISC_ECN_SCHEME_MIN;
            entry->extra2 = &PRIO_WFQ_QDISC_ECN_SCHEME_MAX;
        }
		/* per-queue DSCP */
		else if (i >= 6 + PRIO_WFQ_QDISC_MAX_QUEUES && i < 6 + 2 * PRIO_WFQ_QDISC_MAX_QUEUES)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_WFQ_QDISC_DSCP_MIN;
			entry->extra2 = &PRIO_WFQ_QDISC_DSCP_MAX;
		}
		/* per-wfq-queue weight */
		else if (i >= 6 + 3 * PRIO_WFQ_QDISC_MAX_QUEUES)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_WFQ_QDISC_WEIGHT_MIN;
			entry->extra2 = &PRIO_WFQ_QDISC_WEIGHT_MAX;
		}
		/* per-queue marking thresholds and buffer sizes */
		else
		{
			entry->proc_handler = &proc_dointvec;
		}
        entry->maxlen=sizeof(int);
    }

    PRIO_WFQ_QDISC_Sysctl = register_sysctl_paths(PRIO_WFQ_QDISC_Params_path, PRIO_WFQ_QDISC_Params_table);

    if (likely(PRIO_WFQ_QDISC_Sysctl))
        return 0;
    else
        return -1;
}

void prio_wfq_qdisc_params_exit()
{
	if (likely(PRIO_WFQ_QDISC_Sysctl))
		unregister_sysctl_table(PRIO_WFQ_QDISC_Sysctl);
}
