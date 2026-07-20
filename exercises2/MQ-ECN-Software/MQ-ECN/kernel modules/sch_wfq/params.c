#include "params.h"
#include <linux/sysctl.h>
#include <linux/string.h>


/* Enable debug mode or not. By default, we disable debug mode. */
int wfq_enable_debug = wfq_disable;
/*
 * Buffer management mode: shared (0) or static (1).
 * By default, we enable shread buffer.
 */
int wfq_buffer_mode = wfq_shared_buffer;
/* Per port shared buffer (bytes) */
int wfq_shared_buffer_bytes = wfq_max_buffer_bytes;
/* Bucket size in bytes. By default, we use 2.5KB for 1G network. */
int wfq_bucket_bytes = 2500;
/*
 * Per port ECN marking threshold (bytes).
 * By default, we use 32KB for 1G network.
 */
int wfq_port_thresh_bytes = 32000;
/* ECN marking scheme. By default, we perform per queue ECN/RED marking. */
int wfq_ecn_scheme = wfq_queue_ecn;
/* By default, we perform enqueue ECN marking. */
int wfq_enable_dequeue_ecn = wfq_disable;
/* TCN threshold (1024 nanoseconds) */
int wfq_tcn_thresh = 250;
/* CoDel target (1024 nanoseconds) */
int wfq_codel_target = 100;
/* CoDel interval (1024 nanoseconds) */
int wfq_codel_interval = 2000;

int wfq_enable_min = wfq_disable;
int wfq_enable_max = wfq_enable;
int wfq_prio_min = 0;
int wfq_prio_max = wfq_max_prio - 1;
int wfq_buffer_mode_min = wfq_shared_buffer;
int wfq_buffer_mode_max = wfq_static_buffer;
int wfq_ecn_scheme_min = wfq_disable_ecn;
int wfq_ecn_scheme_max = wfq_codel;
int wfq_dscp_min = 0;
int wfq_dscp_max = (1 << 6) - 1;
int wfq_weight_min = 1;
int wfq_weight_max = wfq_min_pkt_bytes;

/* Per queue ECN marking threshold (bytes) */
int wfq_queue_thresh_bytes[wfq_max_queues];
/* DSCP value for different queues*/
int wfq_queue_dscp[wfq_max_queues];
/* weight for different queues*/
int wfq_queue_weight[wfq_max_queues];
/* Per queue minimum guarantee buffer (bytes) */
int wfq_queue_buffer_bytes[wfq_max_queues];
/* Per queue priority (0 to wfq_max_prio - 1) */
int wfq_queue_prio[wfq_max_queues];

/*
 * All parameters that can be configured through sysctl.
 * We have wfq_global_params + 4 * wfq_max_queues parameters in total.
 */
struct wfq_param wfq_params[wfq_total_params + 1] =
{
	{"enable_debug",	&wfq_enable_debug},
	{"buffer_mode",		&wfq_buffer_mode},
	{"shared_buffer",	&wfq_shared_buffer_bytes},
	{"bucket", 		&wfq_bucket_bytes},
	{"port_thresh",		&wfq_port_thresh_bytes},
	{"ecn_scheme", 		&wfq_ecn_scheme},
	{"enable_dequeue_ecn",	&wfq_enable_dequeue_ecn},
	{"tcn_thresh",		&wfq_tcn_thresh},
	{"codel_target",	&wfq_codel_target},
	{"codel_interval",	&wfq_codel_interval},
};

struct ctl_table wfq_params_table[wfq_total_params + 1];

struct ctl_path wfq_params_path[] =
{
	{ .procname = "wfq" },
	{ },
};

struct ctl_table_header *wfq_sysctl = NULL;

bool wfq_params_init(void)
{
	int i, index;
	memset(wfq_params_table, 0, sizeof(wfq_params_table));

	for (i = 0; i < wfq_max_queues; i++)
	{
		/* Per queue ECN marking threshold*/
		index = wfq_global_params + i;
		snprintf(wfq_params[index].name, 63, "queue_thresh_%d", i);
		wfq_params[index].ptr = &wfq_queue_thresh_bytes[i];
		wfq_queue_thresh_bytes[i] = wfq_port_thresh_bytes;

		/* Per-queue DSCP */
		index = wfq_global_params + i + wfq_max_queues;
		snprintf(wfq_params[index].name, 63, "queue_dscp_%d", i);
		wfq_params[index].ptr = &wfq_queue_dscp[i];
		wfq_queue_dscp[i] = i;

		/* Per-queue weight */
		index = wfq_global_params + i + 2 * wfq_max_queues;
		snprintf(wfq_params[index].name, 63, "queue_weight_%d", i);
		wfq_params[index].ptr = &wfq_queue_weight[i];
		wfq_queue_weight[i] = wfq_weight_min;

		/* Per-queue buffer size */
		index = wfq_global_params + i + 3 * wfq_max_queues;
		snprintf(wfq_params[index].name, 63, "queue_buffer_%d", i);
		wfq_params[index].ptr = &wfq_queue_buffer_bytes[i];
		wfq_queue_buffer_bytes[i] = wfq_max_buffer_bytes;

		/* Per-queue priority */
		index = wfq_global_params + i + 4 * wfq_max_queues;
		snprintf(wfq_params[index].name, 63, "queue_prio_%d", i);
		wfq_params[index].ptr = &wfq_queue_prio[i];
		wfq_queue_prio[i] = 0;
	}

	/* End of the parameters */
	wfq_params[wfq_total_params].ptr = NULL;

	for (i = 0; i < wfq_total_params; i++)
	{
		struct ctl_table *entry = &wfq_params_table[i];

		/* Initialize entry (ctl_table) */
		entry->procname = wfq_params[i].name;
		entry->data = wfq_params[i].ptr;
		entry->mode = 0644;

		/* enable_debug and enable_dequeue_ecn */
		if (i == 0 || i == 6)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &wfq_enable_min;
			entry->extra2 = &wfq_enable_max;
		}
		/* buffer_mode */
		else if (i == 1)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &wfq_buffer_mode_min;
			entry->extra2 = &wfq_buffer_mode_max;
		}
		/* ecn_scheme */
		else if (i == 5)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &wfq_ecn_scheme_min;
			entry->extra2 = &wfq_ecn_scheme_max;
		}
		/* Per-queue DSCP */
		else if (i >= wfq_global_params + wfq_max_queues &&
			 i < wfq_global_params + 2 * wfq_max_queues)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &wfq_dscp_min;
			entry->extra2 = &wfq_dscp_max;
		}
		/* Per-queue weight */
		else if (i >= wfq_global_params + 2 * wfq_max_queues &&
			 i < wfq_global_params + 3 * wfq_max_queues)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &wfq_weight_min;
			entry->extra2 = &wfq_weight_max;
		}
		/* Per-queue priority */
		else if (i >= wfq_global_params + 4 * wfq_max_queues)
		{
			entry->proc_handler = &proc_dointvec_minmax;
			entry->extra1 = &wfq_prio_min;
			entry->extra2 = &wfq_prio_max;
		}
		else
		{
			entry->proc_handler = &proc_dointvec;
		}
		entry->maxlen=sizeof(int);
	}

	wfq_sysctl = register_sysctl_paths(wfq_params_path, wfq_params_table);

	if (likely(wfq_sysctl))
		return true;
	else
		return false;

}

void wfq_params_exit()
{
	if (likely(wfq_sysctl))
		unregister_sysctl_table(wfq_sysctl);
}
