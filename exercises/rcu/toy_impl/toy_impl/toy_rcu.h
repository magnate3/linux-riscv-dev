#ifndef __TOY_RCU_H_
#define __TOY_RCU_H_

void toy_rcu_read_lock(void);
void toy_rcu_read_unlock(void);
void toy_synchronize_rcu(void);
/*
 * rcu_assign_pointer() is implemented as a macro,
 * though it would be cool to be able to declare
 * a function in this manner
 */
#define toy_rcu_assign_pointer(p, v) \
	({ \
		smp_store_release(&(p), (v)); \
	})

#define toy_rcu_dereference(p) \
	({ \
	 typeof(p) __p1 = READ_ONCE(p); \
	 (__p1); \
	})


#endif
