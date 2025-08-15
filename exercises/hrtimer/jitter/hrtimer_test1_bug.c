
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/hrtimer.h>
#include <linux/jiffies.h>
#include <linux/timekeeper_internal.h>
 
 
static struct hrtimer timer;
ktime_t kt;
DEFINE_PER_CPU(struct hrtimer_cpu_base, hrtimer_bases) =
{
        .lock = __RAW_SPIN_LOCK_UNLOCKED(hrtimer_bases.lock),
        .clock_base =
        {
                {
                        .index = HRTIMER_BASE_MONOTONIC,
                        .clockid = CLOCK_MONOTONIC,
                        .get_time = &ktime_get,
                },
                {
                        .index = HRTIMER_BASE_REALTIME,
                        .clockid = CLOCK_REALTIME,
                        .get_time = &ktime_get_real,
                },
                {
                        .index = HRTIMER_BASE_BOOTTIME,
                        .clockid = CLOCK_BOOTTIME,
                        .get_time = &ktime_get_boottime,
                },
                {
                        .index = HRTIMER_BASE_TAI,
                        .clockid = CLOCK_TAI,
                        .get_time = &ktime_get_clocktai,
                },
                {
                        .index = HRTIMER_BASE_MONOTONIC_SOFT,
                        .clockid = CLOCK_MONOTONIC,
                        .get_time = &ktime_get,
                },
                {
                        .index = HRTIMER_BASE_REALTIME_SOFT,
                        .clockid = CLOCK_REALTIME,
                        .get_time = &ktime_get_real,
                },
                {
                        .index = HRTIMER_BASE_BOOTTIME_SOFT,
                        .clockid = CLOCK_BOOTTIME,
                        .get_time = &ktime_get_boottime,
                },
                {
                        .index = HRTIMER_BASE_TAI_SOFT,
                        .clockid = CLOCK_TAI,
                        .get_time = &ktime_get_clocktai,
                },
        }
};
/*
 *  * The most important data for readout fits into a single 64 byte
 *   * cache line.
 *    */
//static struct {
//	        seqcount_raw_spinlock_t seq;
//		        struct timekeeper       timekeeper;
//} tk_core ____cacheline_aligned = {
//	        .seq = SEQCNT_RAW_SPINLOCK_ZERO(tk_core.seq, &timekeeper_lock),
//}
#ifdef CONFIG_CLOCKSOURCE_VALIDATE_LAST_CYCLE
static inline u64 clocksource_delta(u64 now, u64 last, u64 mask)
{
	        u64 ret = (now - last) & mask;

		        /*
			 *          * Prevent time going backwards by checking the MSB of mask in
			 *                   * the result. If set, return 0.
			 *                            */
		        return ret & ~(mask >> 1) ? 0 : ret;
}
#else
static inline u64 clocksource_delta(u64 now, u64 last, u64 mask)
{
	        return (now - last) & mask;
}
#endif
static inline u64 tk_clock_read(const struct tk_read_base *tkr)
{
	struct clocksource *clock = READ_ONCE(tkr->clock);

	return clock->read(clock);
}
#if 0
//#ifdef CONFIG_DEBUG_TIMEKEEPING
#define WARNING_FREQ (HZ*300) /* 5 minute rate-limiting */
static inline u64 timekeeping_get_delta(const struct tk_read_base *tkr)
{
	struct timekeeper *tk = &tk_core.timekeeper;
	u64 now, last, mask, max, delta;
	unsigned int seq;

	/*
	 * Since we're called holding a seqcount, the data may shift
	 * under us while we're doing the calculation. This can cause
	 * false positives, since we'd note a problem but throw the
	 * results away. So nest another seqcount here to atomically
	 * grab the points we are checking with.
	 */
	do {
		seq = read_seqcount_begin(&tk_core.seq);
		now = tk_clock_read(tkr);
		last = tkr->cycle_last;
		mask = tkr->mask;
		max = tkr->clock->max_cycles;
	} while (read_seqcount_retry(&tk_core.seq, seq));

	delta = clocksource_delta(now, last, mask);

	/*
	 * Try to catch underflows by checking if we are seeing small
	 * mask-relative negative values.
	 */
	if (unlikely((~delta & mask) < (mask >> 3))) {
		tk->underflow_seen = 1;
		delta = 0;
	}

	/* Cap delta value to the max_cycles values to avoid mult overflows */
	if (unlikely(delta > max)) {
		tk->overflow_seen = 1;
		delta = tkr->clock->max_cycles;
	}

	return delta;
}
#else
static inline void timekeeping_check_update(struct timekeeper *tk, u64 offset)
{
}
static inline u64 timekeeping_get_delta(const struct tk_read_base *tkr)
{
	u64 cycle_now, delta;

	/* read clocksource */
	cycle_now = tk_clock_read(tkr);

	/* calculate the delta since the last update_wall_time */
	delta = clocksource_delta(cycle_now, tkr->cycle_last, tkr->mask);

	return delta;
}
#endif
/* Timekeeper helper functions. */

static inline u64 timekeeping_delta_to_ns(const struct tk_read_base *tkr, u64 delta)
{
	u64 nsec;

	nsec = delta * tkr->mult + tkr->xtime_nsec;
	nsec >>= tkr->shift;

	return nsec;
}

static inline u64 timekeeping_get_ns(const struct tk_read_base *tkr)
{
	u64 delta;

	delta = timekeeping_get_delta(tkr);
	return timekeeping_delta_to_ns(tkr, delta);
}
ktime_t ktime_get_update_offsets_now(unsigned int *cwsseq, ktime_t *offs_real,
				     ktime_t *offs_boot, ktime_t *offs_tai)
{
	struct timekeeper *tk = &tk_core.timekeeper;
	unsigned int seq;
	ktime_t base;
	u64 nsecs;

	do {
		seq = read_seqcount_begin(&tk_core.seq);

		base = tk->tkr_mono.base;
		nsecs = timekeeping_get_ns(&tk->tkr_mono);
		base = ktime_add_ns(base, nsecs);

		if (*cwsseq != tk->clock_was_set_seq) {
			*cwsseq = tk->clock_was_set_seq;
			*offs_real = tk->offs_real;
			*offs_boot = tk->offs_boot;
			*offs_tai = tk->offs_tai;
		}

		/* Handle leapsecond insertion adjustments */
		if (unlikely(base >= tk->next_leap_ktime))
			*offs_real = ktime_sub(tk->offs_real, ktime_set(1, 0));

	} while (read_seqcount_retry(&tk_core.seq, seq));

	return base;
}
#if 0
static inline ktime_t hrtimer_update_base(struct hrtimer_cpu_base *base)
{
        ktime_t *offs_real = &base->clock_base[HRTIMER_BASE_REALTIME].offset;
        ktime_t *offs_boot = &base->clock_base[HRTIMER_BASE_BOOTTIME].offset;
        ktime_t *offs_tai = &base->clock_base[HRTIMER_BASE_TAI].offset;
#if 0
        ktime_t now = ktime_get_update_offsets_now(&base->clock_was_set_seq,
                                            offs_real, offs_boot, offs_tai);
#endif
#if 0
        base->clock_base[HRTIMER_BASE_REALTIME_SOFT].offset = *offs_real;
        base->clock_base[HRTIMER_BASE_BOOTTIME_SOFT].offset = *offs_boot;
        base->clock_base[HRTIMER_BASE_TAI_SOFT].offset = *offs_tai;
#endif
        return now;
}
#endif
static enum hrtimer_restart hrtimer_handler(struct hrtimer *timer)
{

 //ktime_t basenow;
 //ktime_t now;
 //struct hrtimer_cpu_base *cpu_base = this_cpu_ptr(&hrtimer_bases);
 //now = hrtimer_update_base(cpu_base);
 //basenow = ktime_add(now, timer->base->offset);
 u64 now = ktime_to_ns(ktime_get());
 u64 soft = ktime_to_ns(timer->_softexpires);
 u64 expires = ktime_to_ns(timer->node.expires);
 printk("softexpires %llu, expires %llu\n, now %llu , now - expires=  %llu", soft, expires, now, now - expires);
 if(soft != expires)
 {
      pr_info("softexpires %llu, expires %llu\n not equals \n", soft, expires);  
 }
 hrtimer_forward(timer, timer->base->get_time(), kt);
 return HRTIMER_RESTART;
 }
 
static int __init test_init(void)
{
 
 pr_info("timer resolution: %lu\n", TICK_NSEC);
 //kt = ktime_set(1, 10); /* 1 sec, 10 nsec */
 kt = ktime_set(0, 5000000); /* 1 sec, 10 nsec */
 hrtimer_init(&timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
 //hrtimer_set_expires(&timer, kt);
 hrtimer_start(&timer, kt, HRTIMER_MODE_REL);//中断触发周期为:1sec + 10 nsec
 timer.function = hrtimer_handler;
 
 printk("\n-------- test start ---------\n");
 return 0;
}
 
static void __exit test_exit(void)
{
 hrtimer_cancel(&timer);
 printk("-------- test over ----------\n");
 return;
}
 
MODULE_LICENSE("GPL");
module_init(test_init);
module_exit(test_exit);
