
# qdisc_watchdog

## qdisc_watchdog触发软中断

```
static enum hrtimer_restart qdisc_watchdog(struct hrtimer *timer)
{
        struct qdisc_watchdog *wd = container_of(timer, struct qdisc_watchdog,
                                                 timer);

        rcu_read_lock();
        __netif_schedule(qdisc_root(wd->qdisc));
        rcu_read_unlock();

        return HRTIMER_NORESTART;
}

```

## clockid

```
	switch (q->clockid) {
	case CLOCK_REALTIME:
		q->get_time = ktime_get_real;
		break;
	case CLOCK_MONOTONIC:
		q->get_time = ktime_get;
		break;
	case CLOCK_BOOTTIME:
		q->get_time = ktime_get_boottime;
		break;
	case CLOCK_TAI:
		q->get_time = ktime_get_clocktai;
		break;
	default:
		NL_SET_ERR_MSG(extack, "Clockid is not supported");
		return -ENOTSUPP;
	}
```


##   qdisc_watchdog_init
```
void qdisc_watchdog_init_clockid(struct qdisc_watchdog *wd, struct Qdisc *qdisc,
                                 clockid_t clockid)
{
        hrtimer_init(&wd->timer, clockid, HRTIMER_MODE_ABS_PINNED);
        wd->timer.function = qdisc_watchdog;
        wd->qdisc = qdisc;
}
EXPORT_SYMBOL(qdisc_watchdog_init_clockid);

void qdisc_watchdog_init(struct qdisc_watchdog *wd, struct Qdisc *qdisc)
{
        qdisc_watchdog_init_clockid(wd, qdisc, CLOCK_MONOTONIC);
}
EXPORT_SYMBOL(qdisc_watchdog_init);

```

##  qdisc_watchdog_schedule_range_ns

```

void qdisc_watchdog_schedule_range_ns(struct qdisc_watchdog *wd, u64 expires,
                                      u64 delta_ns)
{
        if (test_bit(__QDISC_STATE_DEACTIVATED,
                     &qdisc_root_sleeping(wd->qdisc)->state))
                return;

        if (hrtimer_is_queued(&wd->timer)) {
                /* If timer is already set in [expires, expires + delta_ns],
                 * do not reprogram it.
                 */
                if (wd->last_expires - expires <= delta_ns)
                        return;
        }

        wd->last_expires = expires;
        hrtimer_start_range_ns(&wd->timer,
                               ns_to_ktime(expires),
                               delta_ns,
                               HRTIMER_MODE_ABS_PINNED);
}
```