#include "common.h"

typedef struct{
	//1:locked 0:unlocked
	volatile int locked;
}cus_spinlock_t;

static inline void
cus_spinlock_unlock(cus_spinlock_t *lock)
{
	__sync_lock_release(&lock->locked);
}

static inline void
cus_spinlock_lock(cus_spinlock_t *lock)
{
	while( __sync_lock_test_and_set(&lock->locked, 1) == 1){
		while(lock->locked == 1){
			cus_pause();	
		}	
	}
}

static inline void
cus_spinlock_init(cus_spinlock_t *lock)
{
	lock->locked = 0;
}
