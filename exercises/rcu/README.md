 
 
 # rcu_read_lock
 
 ```
 static __always_inline void rcu_read_lock(void)
{
	__rcu_read_lock();
	__acquire(RCU);
	rcu_lock_acquire(&rcu_lock_map);
	RCU_LOCKDEP_WARN(!rcu_is_watching(),
			 "rcu_read_lock() used illegally while idle");
}

#ifdef CONFIG_PREEMPT_RCU
void __rcu_read_lock(void);
void __rcu_read_unlock(void);
void rcu_read_unlock_special(struct task_struct *t);
void synchronize_rcu(void);
/*
 * Defined as a macro as it is a very low level header included from
 * areas that don't even know about current.  This gives the rcu_read_lock()
 * nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
 * types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
 */
#define rcu_preempt_depth() (current->rcu_read_lock_nesting)
#else /* #ifdef CONFIG_PREEMPT_RCU */
static inline void __rcu_read_lock(void)
{
	preempt_disable();
}
static inline void __rcu_read_unlock(void)
{
	preempt_enable();
}
static inline void synchronize_rcu(void)
{
	synchronize_sched();
}
static inline int rcu_preempt_depth(void)
{
	return 0;
}
#endif /* #else #ifdef CONFIG_PREEMPT_RCU */
 ```
 ## synchronize_rcu
 ```
 static void delete_book(int id, int async) {
        struct book *b;

        spin_lock(&books_lock);
        list_for_each_entry(b, &books, node) {
                if(b->id == id) {
                        /**
                         * list_del
                         *
                         * del_node(writer - delete) require locking mechanism.
                         * we can choose 3 ways to lock. Use 'a' here.
                         *
                         *      a.      locking,
                         *      b.      atomic operations, or
                         *      c.      restricting updates to a single task.
                        */
                        list_del_rcu(&b->node);
                        spin_unlock(&books_lock);

                        if(async) {
                                call_rcu(&b->rcu, book_reclaim_callback);
                        }else {
                                synchronize_rcu();
                                kfree(b);
                        }
                        return;
                }
        }
        spin_unlock(&books_lock);

        pr_err("not exist book\n");
}
  void synchronize_sched(void)
 {
     rcu_lockdep_assert(!lock_is_held(&rcu_bh_lock_map) &&
                !lock_is_held(&rcu_lock_map) &&
                !lock_is_held(&rcu_sched_lock_map),
                "Illegal synchronize_sched() in RCU-sched read-side critical section");
     if (rcu_blocking_is_gp())
         return;
     if (rcu_expedited)
         synchronize_sched_expedited();
     else
         wait_rcu_gp(call_rcu_sched);
 }
 EXPORT_SYMBOL_GPL(synchronize_sched);

 ```
 
 
 # insmod rcu.ko 
  ```
 [ 3544.302482] Module starting ... ... ...
[ 3544.500563] gbl_foo->a: 10 
[ 3544.620561] kfree(old_fp) 
[ 3544.623256] gbl_foo->a: 11 
[ 3575.220635] kfree(gbl_foo) 
[ 3575.223420] Module terminating ... ... ...
  ```
  
# rcu nest

```
static int foo_get_a(void)
{
    int retval;

    rcu_read_lock();  // 1
    rcu_read_lock(); // 1
    retval = rcu_dereference(gbl_foo)->a;
    rcu_read_unlock();
    rcu_read_unlock();
    return retval;
}



static int __init kernel_init(void)
{
    printk("Module starting ... ... ...\n");

    foo_update_a(10, 11, 12);
    printk("gbl_foo->a: %d \n", foo_get_a());

    foo_update_a(11, 12, 13);
    printk("gbl_foo->a: %d \n", foo_get_a());

    return 0;
}

````

```
[ 1499.348693] Module starting ... ... ...
[ 1499.442565] gbl_foo->a: 10 
[ 1499.562553] kfree(old_fp) 
[ 1499.565248] gbl_foo->a: 11 
```

# insmod  rcu_test2.ko

![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/task2.png)

***其中,rcu_read_lock()和rcu_read_unlock()用来保持一个读者的RCU临界区.在该临界区内不允许发生上下文切换.***

为什么不能发生切换呢？因为内核要根据“是否发生过切换”来判断读者是否已结束读操作

  对于reader，RCU的操作包括：
（1）rcu_read_lock，用来标识RCU read side临界区的开始。

（2）rcu_dereference，该接口用来获取RCU protected pointer。reader要访问RCU保护的共享数据，当然要获取RCU protected pointer，然后通过该指针进行dereference的操作。

（3）rcu_read_unlock，用来标识reader离开RCU read side临界区

对于writer，RCU的操作包括：

（1）rcu_assign_pointer。该接口被writer用来进行removal的操作，在witer完成新版本数据分配和更新之后，调用这个接口可以让RCU protected pointer指向RCU protected data。

（2）synchronize_rcu。writer端的操作可以是同步的，也就是说，完成更新操作之后，可以调用该接口函数等待所有在旧版本数据上的reader线程离开临界区，一旦从该函数返回，说明旧的共享数据没有任何引用了，可以直接进行reclaimation的操作。

（3）call_rcu。当然，某些情况下（例如在softirq context中），writer无法阻塞，这时候可以调用call_rcu接口函数，该函数仅仅是注册了callback就直接返回了，在适当的时机会调用callback函数，完成reclaimation的操作。这样的场景其实是分开removal和reclaimation的操作在两个不同的线程中：updater和reclaimer。

 ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/synchronize_rcu.png)
 
 

```

[15961.179144] 1664348380 352341:create task 1 success
[15961.184048] 1664348380 357247:create task 2 success
[15961.188909] task1 enter rcu 
[15961.191785] task1 enter rcu read lock: ffffa05fc9c97d80  ///ptest不是null
[15962.188943] task2 enter rcu 
[15962.191815] task2 assign ptest null 
[15971.197396] ptest is null  ///ptest为什么是null,因为task2 enter rcu ，rcu_assign_pointer
[15971.200095] ptest2 is ffffa05fc9c97d80 ///ptest2为什么不是null,因为执行reference的时候ptest不是NULL
[15981.204242] after rcu unlock,  ptest is null 
[15981.279532] rcu_demo_free 1664348400 453065:free ptest
```

# insmod  rcu_test5.ko 


 ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/task5.png)

```
[16290.537132] 1664348709 715809:create task 1 success
[16290.542025] 1664348709 720705:create task 2 success
[16290.546905] task2 enter rcu 
[16290.549776] task2 assign ptest null 
[16291.546936] task1 enter rcu 
[16301.550128] ptest is null 
[16301.552825] ptest2 is null  ///ptest也是null
[16311.555936] after rcu unlock,  ptest is null 
[16311.574043] rcu_demo_free 1664348730 753072:free ptest  //task1
```
  
  
   ![image](https://github.com/magnate3/linux-riscv-dev/blob/main/exercises/rcu/task51.png)
  
```
[17474.793095] 1664349893 991479:create task 1 success
[17474.797999] 1664349893 996385:create task 2 success
[17474.802861] task2 enter rcu 
[17474.805736] task2 assign ptest null 
[17474.944676] rcu_demo_free 1664349894 143064:free ptest //马上free
```

# list_add_rcu

```
int gen_pool_add_virt(struct gen_pool *pool, unsigned long virt, phys_addr_t phys,
                 size_t size, int nid)
{
        struct gen_pool_chunk *chunk;
        int nbits = size >> pool->min_alloc_order;
        int nbytes = sizeof(struct gen_pool_chunk) +
                                BITS_TO_LONGS(nbits) * sizeof(long);

        chunk = kzalloc_node(nbytes, GFP_KERNEL, nid);
        if (unlikely(chunk == NULL))
                return -ENOMEM;

        chunk->phys_addr = phys;
        chunk->start_addr = virt;
        chunk->end_addr = virt + size - 1;
        atomic_long_set(&chunk->avail, size);

        spin_lock(&pool->lock);
        list_add_rcu(&chunk->next_chunk, &pool->chunks);
        spin_unlock(&pool->lock);

        return 0;
}
```

# RCU机制的函数接口

关于写者函数，主要就是call_rcu和call_rcu_bh两个函数。其中call_rcu能实现的功能是它不会使写者阻塞，因而它可在中断上下文及软中断使用，该函数将函数func挂接到RCU的回调函数链表上，然后立即返回。而call_rcu_bh函数实现的功能几乎与call_rcu完全相同，唯一的差别是它将软中断的完成当作经历一个quiescent state（静默状态，本节一开始有提及这个概念）。  

 因此若写者使用了该函数，那么读者需对应的使用rcu_read_lock_bh() 和rcu_read_unlock_bh()。  

为什么这么说呢，这里笔者结合call_rcu_bh的源码实现给出自己的看法：一个静默状态表示一次的进程上下文切换（上述提及），就是当前进程执行完毕并顺利切换到下一个进程。将软中断的完成当作经历一个静默状态是确保此时系统的软中断能够顺利的执行完毕，**因为call_rcu_bh可在中断上下文使用，而中断上下文能打断软中断的运行**，故而当call_rcu_bh在中断上下文中使用的时候，需确保软中断的能够顺利执行完毕。  

对应于此时读者需使用rcu_read_lock_bh() 和rcu_read_unlock_bh()函数的原因是由于call_rcu_bh函数不会使写者阻塞，可在中断上下文及软中断使用。这表明此时系统中的中断和软中断并没有被关闭。那么**写者**在调用call_rcu_bh函数访问临界区时，RCU机制下的读者也能访问临界区。此时对于读者而言，它若是需要读取临界区的内容，它必须把软中断关闭，以免读者在当前的进程上下文过程中被软中断打断（上述内容提过软中断可以打断当前的进程上下文）。而rcu_read_lock_bh() 和rcu_read_unlock_bh()函数的实质是调用local_bh_disable()和local_bh_enable()函数，显然这是实现了禁止软中断和使能软中断的功能。

另外在Linux源码中关于call_rcu_bh函数的注释中还明确说明了如果当前的进程是在中断上下文中，则需要执行rcu_read_lock()和rcu_read_unlock()，结合这两个函数的实现实质表明它实际上禁止或使能内核的抢占调度，原因不言而喻，**避免当前进程在执行读写过程中被其它进程抢占*。同时内核注释还表明call_rcu_bh这个接口函数的使用条件是在大部分的读临界区操作发生在软中断上下文中，原因还是需从它实现的功能出发，相信很容易理解，主要是要从执行效率方面考虑。

关于RCU的回调函数实现本质是：它主要是由两个数据结构维护，包括rcu_data和rcu_bh_data数据结构，实现了挂接回调函数，从而使回调函数组成链表。回调函数的原则先注册到链表的先执行。
 
  
 
   
   
   
  