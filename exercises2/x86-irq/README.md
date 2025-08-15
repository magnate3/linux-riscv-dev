


# linux-6.3
arch/x86/kernel/irq.c    
```
/*
 * common_interrupt() handles all normal device IRQ's (the special SMP
 * cross-CPU interrupts have their own entry points).
 */
DEFINE_IDTENTRY_IRQ(common_interrupt)
{
        struct pt_regs *old_regs = set_irq_regs(regs);
        struct irq_desc *desc;

        /* entry code tells RCU that we're not quiescent.  Check it. */
        RCU_LOCKDEP_WARN(!rcu_is_watching(), "IRQ failed to wake up RCU");

        desc = __this_cpu_read(vector_irq[vector]);
        if (likely(!IS_ERR_OR_NULL(desc))) {
                handle_irq(desc, regs);
        } else {
                ack_APIC_irq();

                if (desc == VECTOR_UNUSED) {
                        pr_emerg_ratelimited("%s: %d.%u No irq handler for vector\n",
                                             __func__, smp_processor_id(),
                                             vector);
                } else {
                        __this_cpu_write(vector_irq[vector], VECTOR_UNUSED);
                }
        }

        set_irq_regs(old_regs);
}

#ifdef CONFIG_X86_LOCAL_APIC
/* Function pointer for generic interrupt vector handling */
void (*x86_platform_ipi_callback)(void) = NULL;
/*
 * Handler for X86_PLATFORM_IPI_VECTOR.
 */
DEFINE_IDTENTRY_SYSVEC(sysvec_x86_platform_ipi)
{
        struct pt_regs *old_regs = set_irq_regs(regs);

        ack_APIC_irq();
        trace_x86_platform_ipi_entry(X86_PLATFORM_IPI_VECTOR);
        inc_irq_stat(x86_platform_ipis);
        if (x86_platform_ipi_callback)
                x86_platform_ipi_callback();
        trace_x86_platform_ipi_exit(X86_PLATFORM_IPI_VECTOR);
        set_irq_regs(old_regs);
}
#endif
```

## 软中断   

```
static inline void __irq_exit_rcu(void)
{
#ifndef __ARCH_IRQ_EXIT_IRQS_DISABLED
        local_irq_disable();
#else
        lockdep_assert_irqs_disabled();
#endif
        account_hardirq_exit(current);
        preempt_count_sub(HARDIRQ_OFFSET);
        if (!in_interrupt() && local_softirq_pending())
                invoke_softirq();

        tick_irq_exit();
}
```