/* https://github.com/torvalds/linux/blob/master/Documentation/RCU/whatisRCU.txt
 *
 * TOY RCU implementation: Locking
 *
 * This code presents a "toy" RCU implementation
 * that is based on familiar locking primitives.
 * Its overhead makes it a non-starter for real-life
 * use, as does its lack of scalability. It is also
 * unsuitable for realtime use, since it allows
 * scheduling latency to "bleed" from one read-side
 * critical section to another. It also assumes
 * recursive reader-writer locks: If you try this with
 * non-recursive locks, and you allow nested rcu_read_lock()
 * call, you can deadlock.
 */
#include <linux/module.h>

static DEFINE_RWLOCK(rcu_gp_mutex);

void toy_rcu_read_lock(void)
{
	read_lock(&rcu_gp_mutex);
}

void toy_rcu_read_unlock(void)
{
	read_unlock(&rcu_gp_mutex);
}

void toy_synchronize_rcu(void)
{
	write_lock(&rcu_gp_mutex);
	/* smp_mb__after_spinlock():
	 * It provides full ordering after lock acquisition.
	 * The ordering guarantees of smp_mb_after_spinlock()
	 * are a strict superset of those smp_mb__after_unlock_lock()
	 *
	 * smp_mb__after_unlock_lock():
	 * It provies a full memory barrier after the immediately
	 * preceding lock operation, but only when paired with
	 * a preceding unlock operation by this same thread or a
	 * preceding unlock operation on the same lock variable.
	 */
	smp_mb__after_spinlock();
	write_unlock(&rcu_gp_mutex);
}

/* QUIZ 1
 * How could a deadlock occur when using this algorithm in
 * a real world Linux kernel?
 *
 * CPU 0:
 * acquires some unrelated lock,
 * call it "problematic_lock",
 * disabling irq via
 * spin lock_irqsave().
 *                                CPU1:
 *				  Enters synchronize_rcu(),
 *				  write-acquiring rcu_gp_mutex.
 * CPU 0:
 * Enters rcu_read_lock(), but
 * must wait.
 *				  CPU1:
 *				  CPU1 is interrupted, and the irq handler
 *				  attemps to acquire problematic_lock.
 *
 * Now the system is deadlocked.
 */
