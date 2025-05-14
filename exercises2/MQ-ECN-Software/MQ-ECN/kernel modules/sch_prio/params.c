#include "params.h"
#include <linux/sysctl.h>
#include <linux/string.h>

/* Debug mode or not. By default, we disable debug mode */
int PRIO_QDISC_DEBUG_MODE = PRIO_QDISC_DEBUG_OFF;
/* Buffer management mode: shared (0) or static (1). By default, we enable shread buffer. */
int PRIO_QDISC_BUFFER_MODE = PRIO_QDISC_SHARED_BUFFER;
/* Per port shared buffer (bytes) */
int PRIO_QDISC_SHARED_BUFFER_BYTES = PRIO_QDISC_MAX_BUFFER_BYTES;
/* Bucket size in nanosecond. By default, we use 20us for 1G network. */
int PRIO_QDISC_BUCKET_NS = 20000;
/* Per port ECN marking threshold (bytes). By default, we use 30KB for 1G network. */
int PRIO_QDISC_PORT_THRESH_BYTES = 30000;
/* ECN marking scheme. By default, we use per queue ECN. */
int PRIO_QDISC_ECN_SCHEME = PRIO_QDISC_QUEUE_ECN;


int PRIO_QDISC_DEBUG_MODE_MIN = PRIO_QDISC_DEBUG_OFF;
int PRIO_QDISC_DEBUG_MODE_MAX = PRIO_QDISC_DEBUG_ON;
int PRIO_QDISC_BUFFER_MODE_MIN = PRIO_QDISC_SHARED_BUFFER;
int PRIO_QDISC_BUFFER_MODE_MAX = PRIO_QDISC_STATIC_BUFFER;
int PRIO_QDISC_ECN_SCHEME_MIN = PRIO_QDISC_DISABLE_ECN;
int PRIO_QDISC_ECN_SCHEME_MAX = PRIO_QDISC_DEQUE_ECN;
int PRIO_QDISC_DSCP_MIN = 0;
int PRIO_QDISC_DSCP_MAX = 63;

/* Per queue ECN marking threshold (bytes) */
int PRIO_QDISC_QUEUE_THRESH_BYTES[PRIO_QDISC_MAX_QUEUES];
/* DSCP value for different queues*/
int PRIO_QDISC_QUEUE_DSCP[PRIO_QDISC_MAX_QUEUES];
/* Per queue minimum guarantee buffer (bytes) */
int PRIO_QDISC_QUEUE_BUFFER_BYTES[PRIO_QDISC_MAX_QUEUES];

/* All parameters that can be configured through sysctl. We have 6 + 3 * PRIO_QDISC_MAX_QUEUES parameters in total. */
struct PRIO_QDISC_Param PRIO_QDISC_Params[6 + 3 * PRIO_QDISC_MAX_QUEUES + 1] =
{
	{"debug_mode", &PRIO_QDISC_DEBUG_MODE},
	{"buffer_mode",&PRIO_QDISC_BUFFER_MODE},
	{"shared_buffer_bytes", &PRIO_QDISC_SHARED_BUFFER_BYTES},
	{"bucket_ns", &PRIO_QDISC_BUCKET_NS},
	{"port_thresh_bytes", &PRIO_QDISC_PORT_THRESH_BYTES},
	{"ecn_scheme",&PRIO_QDISC_ECN_SCHEME},
};

struct ctl_table PRIO_QDISC_Params_table[6 + 3 * PRIO_QDISC_MAX_QUEUES + 1];

struct ctl_path PRIO_QDISC_Params_path[] =
{
	{ .procname = "prio" },
	{ },
};

struct ctl_table_header *PRIO_QDISC_Sysctl = NULL;

int prio_qdisc_params_init()
{
	int i=0;
	memset(PRIO_QDISC_Params_table, 0, sizeof(PRIO_QDISC_Params_table));

	for (i = 0; i < PRIO_QDISC_MAX_QUEUES; i++)
	{
		/* Initialize PRIO_QDISC_QUEUE_THRESH_BYTES[PRIO_QDISC_MAX_QUEUES]*/
		snprintf(PRIO_QDISC_Params[6 + i].name, 63, "queue_thresh_bytes_%d", i);
		PRIO_QDISC_Params[6 + i].ptr = &PRIO_QDISC_QUEUE_THRESH_BYTES[i];
		PRIO_QDISC_QUEUE_THRESH_BYTES[i] = PRIO_QDISC_PORT_THRESH_BYTES;

		/* Initialize PRIO_QDISC_QUEUE_DSCP[PRIO_QDISC_MAX_QUEUES] */
		snprintf(PRIO_QDISC_Params[6 + i + PRIO_QDISC_MAX_QUEUES].name, 63, "queue_dscp_%d", i);
		PRIO_QDISC_Params[6 + i + PRIO_QDISC_MAX_QUEUES].ptr = &PRIO_QDISC_QUEUE_DSCP[i];
		PRIO_QDISC_QUEUE_DSCP[i] = i;

		/* Initialize PRIO_QDISC_QUEUE_BUFFER_BYTES[PRIO_QDISC_MAX_QUEUES] */
		snprintf(PRIO_QDISC_Params[6 + i + 2 * PRIO_QDISC_MAX_QUEUES].name, 63, "queue_buffer_bytes_%d", i);
		PRIO_QDISC_Params[6 + i + 2 * PRIO_QDISC_MAX_QUEUES].ptr = &PRIO_QDISC_QUEUE_BUFFER_BYTES[i];
		PRIO_QDISC_QUEUE_BUFFER_BYTES[i] = PRIO_QDISC_MAX_BUFFER_BYTES;
	}

	/* End of the parameters */
	PRIO_QDISC_Params[6 + 3 * PRIO_QDISC_MAX_QUEUES].ptr = NULL;

	for (i = 0; i < 6 + 3 * PRIO_QDISC_MAX_QUEUES + 1; i++)
	{
		struct ctl_table *entry = &PRIO_QDISC_Params_table[i];

		/* End */
		if (PRIO_QDISC_Params[i].ptr == NULL)
			break;

		/* Initialize entry (ctl_table) */
		entry->procname = PRIO_QDISC_Params[i].name;
		entry->data = PRIO_QDISC_Params[i].ptr;
		entry->mode = 0644;

		/* PRIO_QDISC_DEBUG_MODE */
		if (i == 0)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_QDISC_DEBUG_MODE_MIN;
			entry->extra2 = &PRIO_QDISC_DEBUG_MODE_MAX;
		}
		/* PRIO_QDISC_BUFFER_MODE */
		else if (i == 1)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_QDISC_BUFFER_MODE_MIN;
			entry->extra2 = &PRIO_QDISC_BUFFER_MODE_MAX;
		}
		/* PRIO_QDISC_ECN_SCHEME */
		else if (i == 5)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_QDISC_ECN_SCHEME_MIN;
			entry->extra2 = &PRIO_QDISC_ECN_SCHEME_MAX;
		}
		/* PRIO_QDISC_QUEUE_DSCP[] */
		else if (i >= 6 + PRIO_QDISC_MAX_QUEUES && i < 6 + 2 * PRIO_QDISC_MAX_QUEUES)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &PRIO_QDISC_DSCP_MIN;
			entry->extra2 = &PRIO_QDISC_DSCP_MAX;
		}
		/*PRIO_QDISC_QUEUE_ECN_THRESH[] and PRIO_QDISC_QUEUE_BUFFER_BYTES[] */
		else
		{
			entry->proc_handler = &proc_dointvec;
		}
		entry->maxlen=sizeof(int);
	}

	PRIO_QDISC_Sysctl = register_sysctl_paths(PRIO_QDISC_Params_path, PRIO_QDISC_Params_table);
	if (unlikely(PRIO_QDISC_Sysctl == NULL))
		return -1;
	else
		return 0;
}

void prio_qdisc_params_exit()
{
	if (likely(PRIO_QDISC_Sysctl))
		unregister_sysctl_table(PRIO_QDISC_Sysctl);
}
