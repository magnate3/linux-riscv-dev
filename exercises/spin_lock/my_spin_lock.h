#ifndef __MY_LINUX_SPINLOCK_TYPES_H
#define __MY_LINUX_SPINLOCK_TYPES_H

/*
 * include/linux/spinlock_types.h - generic spinlock type definitions
 *                                  and initializers
 *
 * portions Copyright 2005, Red Hat, Inc., Ingo Molnar
 * Released under the General Public License (GPL).
 */

#include <linux/spinlock_types_raw.h>


/* Non PREEMPT_RT kernels map spinlock to raw_spinlock */
typedef struct my_spinlock {
	union {
		struct raw_spinlock rlock;

#ifdef CONFIG_DEBUG_LOCK_ALLOC
# define LOCK_PADSIZE (offsetof(struct raw_spinlock, dep_map))
		struct {
			u8 __padding[LOCK_PADSIZE];
			struct lockdep_map dep_map;
		};
#endif
	};
} my_spinlock_t;

#define ___MY_SPIN_LOCK_INITIALIZER(lockname)	\
	{					\
	.raw_lock = __ARCH_SPIN_LOCK_UNLOCKED,	\
	SPIN_DEBUG_INIT(lockname)		\
	SPIN_DEP_MAP_INIT(lockname) }

#define __MY_SPIN_LOCK_INITIALIZER(lockname) \
	{ { .rlock = ___MY_SPIN_LOCK_INITIALIZER(lockname) } }

#define __MY_SPIN_LOCK_UNLOCKED(lockname) \
	(my_spinlock_t) __MY_SPIN_LOCK_INITIALIZER(lockname)

#define DEFINE_MY_SPINLOCK(x)	my_spinlock_t x = __MY_SPIN_LOCK_UNLOCKED(x)

#include <linux/rwlock_types.h>
#endif

