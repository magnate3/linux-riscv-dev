// SPDX-License-Identifier: GPL-2.0-or-later
/*
 *  linux/drivers/clocksource/timer-sp.c
 *
 *  Copyright (C) 1999 - 2003 ARM Limited
 *  Copyright (C) 2000 Deep Blue Solutions Ltd
 */

#define pr_fmt(fmt)    KBUILD_MODNAME ": " fmt

#include <linux/clk.h>
#include <linux/clocksource.h>
#include <linux/clockchips.h>
#include <linux/clkdev.h>
#include <linux/err.h>
#include <linux/interrupt.h>
#include <linux/irq.h>
#include <linux/io.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_clk.h>
#include <linux/of_irq.h>
#include <linux/sched_clock.h>

#include "timer-sp.h"

#define SP804_CLKSRC	"sp804 source"
#define SP804_CLKEVT	"sp804 event"
#define SP804_TEST  1
#define SP804_TEST_IRQ 1
/* Hisilicon 64-bit timer(a variant of ARM SP804) */
#define HISI_TIMER_1_BASE	0x00
#define HISI_TIMER_2_BASE	0x40
#define HISI_TIMER_LOAD		0x00
#define HISI_TIMER_LOAD_H	0x04
#define HISI_TIMER_VALUE	0x08
#define HISI_TIMER_VALUE_H	0x0c
#define HISI_TIMER_CTRL		0x10
#define HISI_TIMER_INTCLR	0x14
#define HISI_TIMER_RIS		0x18
#define HISI_TIMER_MIS		0x1c
#define HISI_TIMER_BGLOAD	0x20
#define HISI_TIMER_BGLOAD_H	0x24

#define PHY_ADDR_OF_RESET_CTL    0x43060000
#define PHY_ADDR_OFFSET_OF_TIMER0    0xC
#define EN_VALUE_OF_TIMER0  1 << 12
static struct sp804_timer arm_sp804_timer __initdata = {
	.load		= TIMER_LOAD,
	.value		= TIMER_VALUE,
	.ctrl		= TIMER_CTRL,
	.intclr		= TIMER_INTCLR,
	.timer_base	= {TIMER_1_BASE, TIMER_2_BASE},
	.width		= 32,
};

static struct sp804_timer hisi_sp804_timer __initdata = {
	.load		= HISI_TIMER_LOAD,
	.load_h		= HISI_TIMER_LOAD_H,
	.value		= HISI_TIMER_VALUE,
	.value_h	= HISI_TIMER_VALUE_H,
	.ctrl		= HISI_TIMER_CTRL,
	.intclr		= HISI_TIMER_INTCLR,
	.timer_base	= {HISI_TIMER_1_BASE, HISI_TIMER_2_BASE},
	.width		= 64,
};

static struct sp804_clkevt sp804_clkevt[NR_TIMERS];
#if SP804_TEST
static u64  sp804_read_all(struct clocksource  *cs);
#endif
static struct clk __init *sp804_dt_init_clk(struct device_node *np, int i,
					    const char *name)
{
	struct clk_lookup *lookup = NULL;
	struct clk *clk;

	clk = of_clk_get(np, i);
	pr_err("tracing %s IS_ERR: %d\n", name, IS_ERR(clk));
	if (!IS_ERR(clk))
		return clk;
         
	lookup = clkdev_create(clk, name, "sp804");
	pr_err("tracing %s clkdev_create: %d\n", name, (NULL == lookup));
	if (!lookup) {
		clk_put(clk);
		return ERR_PTR(-EINVAL);
	}
	clkdev_add(lookup);
	return clk;
}

static long __init sp804_get_clock_rate(struct clk *clk, const char *name)
{
	int err;

	if (!clk)
		clk = clk_get_sys("sp804", name);
	if (IS_ERR(clk)) {
		pr_err("%s clock not found: %ld\n", name, PTR_ERR(clk));
		return PTR_ERR(clk);
	}

	err = clk_prepare_enable(clk);
	if (err) {
		pr_err("clock failed to enable: %d\n", err);
		clk_put(clk);
		return err;
	}

	return clk_get_rate(clk);
}

static struct sp804_clkevt * __init sp804_clkevt_get(void __iomem *base)
{
	int i;

	for (i = 0; i < NR_TIMERS; i++) {
		if (sp804_clkevt[i].base == base)
			return &sp804_clkevt[i];
	}

	/* It's impossible to reach here */
	WARN_ON(1);

	return NULL;
}

static struct sp804_clkevt *sched_clkevt;

static u64 notrace sp804_read(void)
{
	return ~readl_relaxed(sched_clkevt->value);
}

static int __init sp804_clocksource_and_sched_clock_init(void __iomem *base,
							 const char *name,
							 struct clk *clk,
							 int use_sched_clock)
{
	long rate;
	struct sp804_clkevt *clkevt;

	rate = sp804_get_clock_rate(clk, name);
	pr_err("tracing clock rate: %ld\n", rate);
	if (rate < 0)
		return -EINVAL;

	clkevt = sp804_clkevt_get(base);

	writel(0, clkevt->ctrl);
	writel(0xffffffff, clkevt->load);
	writel(0xffffffff, clkevt->value);
	if (clkevt->width == 64) {
		writel(0xffffffff, clkevt->load_h);
		writel(0xffffffff, clkevt->value_h);
	}
#if SP804_TEST
	writel(TIMER_CTRL_32BIT | TIMER_CTRL_ENABLE | TIMER_CTRL_PERIODIC ,
		clkevt->ctrl);
#else
	writel(TIMER_CTRL_32BIT | TIMER_CTRL_ENABLE | TIMER_CTRL_PERIODIC,
		clkevt->ctrl);
#endif
#if SP804_TEST
	clocksource_mmio_init(clkevt->value, name,
		rate, 200, 32, sp804_read_all);
#else
	clocksource_mmio_init(clkevt->value, name,
		rate, 200, 32, clocksource_mmio_readl_down);
#endif
	if (use_sched_clock) {
		sched_clkevt = clkevt;
		sched_clock_register(sp804_read, 32, rate);
	}

	return 0;
}


static struct sp804_clkevt *common_clkevt;

static unsigned long  sp804_intc_count = 0;
static int sp804_sys_init(void);
/*
 * IRQ handler for the timer
 */
static irqreturn_t sp804_timer_interrupt(int irq, void *dev_id)
{
#if SP804_TEST_IRQ
	pr_err("sp804 timers interrupt happens ");
	if(0 == ++sp804_intc_count % 10){
	    pr_err("sp804 10 timers of interrupt happens again");
	}
	/* clear the interrupt */
	writel(1, common_clkevt->intclr);
	return IRQ_HANDLED;
#endif
	struct clock_event_device *evt = dev_id;

	/* clear the interrupt */
	writel(1, common_clkevt->intclr);

	evt->event_handler(evt);

	return IRQ_HANDLED;
}

static inline void timer_shutdown(struct clock_event_device *evt)
{
#if SP804_TEST_IRQ
	    pr_err("sp804 timers shutdown \n");
	    // to test interrupt
	    return ;
#endif
	writel(0, common_clkevt->ctrl);
}

static int sp804_shutdown(struct clock_event_device *evt)
{
	timer_shutdown(evt);
	return 0;
}

static int sp804_set_periodic(struct clock_event_device *evt)
{
#if SP804_TEST
	pr_err("sp804 set periodic\n");
	unsigned long ctrl = TIMER_CTRL_32BIT | TIMER_CTRL_IE |
			     TIMER_CTRL_PERIODIC | TIMER_CTRL_ENABLE ;
#else
	unsigned long ctrl = TIMER_CTRL_32BIT | TIMER_CTRL_IE |
			     TIMER_CTRL_PERIODIC | TIMER_CTRL_ENABLE;
#endif

	timer_shutdown(evt);
	writel(common_clkevt->reload, common_clkevt->load);
	writel(ctrl, common_clkevt->ctrl);
	return 0;
}

static int sp804_set_next_event(unsigned long next,
	struct clock_event_device *evt)
{
#if SP804_TEST
	pr_err("sp804 set next event\n");
	unsigned long ctrl = TIMER_CTRL_32BIT | TIMER_CTRL_IE |
			     TIMER_CTRL_ONESHOT | TIMER_CTRL_ENABLE ;
#else
	unsigned long ctrl = TIMER_CTRL_32BIT | TIMER_CTRL_IE |
			     TIMER_CTRL_ONESHOT | TIMER_CTRL_ENABLE;
#endif

	writel(next, common_clkevt->load);
	writel(ctrl, common_clkevt->ctrl);

	return 0;
}

static struct clock_event_device sp804_clockevent = {
	.features		= CLOCK_EVT_FEAT_PERIODIC |
				  CLOCK_EVT_FEAT_ONESHOT |
				  CLOCK_EVT_FEAT_DYNIRQ,
	.set_state_shutdown	= sp804_shutdown,
	.set_state_periodic	= sp804_set_periodic,
	.set_state_oneshot	= sp804_shutdown,
	.tick_resume		= sp804_shutdown,
	.set_next_event		= sp804_set_next_event,
	.rating			= 300,
};

static int __init sp804_clockevents_init(void __iomem *base, unsigned int irq,
					 struct clk *clk, const char *name)
{
	struct clock_event_device *evt = &sp804_clockevent;
	long rate;

	rate = sp804_get_clock_rate(clk, name);
	if (rate < 0)
		return -EINVAL;

	common_clkevt = sp804_clkevt_get(base);
	common_clkevt->reload = DIV_ROUND_CLOSEST(rate, HZ);
	evt->name = name;
	evt->irq = irq;
	evt->cpumask = cpu_possible_mask;


#if SP804_TEST_IRQ
	writel(0, common_clkevt->ctrl);
	writel(0xffffffff, common_clkevt->load);
	writel(0xffffffff, common_clkevt->value);
	writel(TIMER_CTRL_32BIT | TIMER_CTRL_ENABLE | TIMER_CTRL_PERIODIC | TIMER_CTRL_IE,
		common_clkevt->ctrl);
#else
	writel(0, common_clkevt->ctrl);
#endif
	pr_err("sp804 request_irq num %u and rate %ld \n", irq, rate);
	if (request_irq(irq, sp804_timer_interrupt, IRQF_TIMER | IRQF_IRQPOLL,
			"timer", &sp804_clockevent))
		pr_err("sp804 request_irq() failed\n");
	clockevents_config_and_register(evt, rate, 0xf, 0xffffffff);

	return 0;
}

static void __init sp804_clkevt_init(struct sp804_timer *timer, void __iomem *base)
{
	int i;

	for (i = 0; i < NR_TIMERS; i++) {
		void __iomem *timer_base;
		struct sp804_clkevt *clkevt;

		timer_base = base + timer->timer_base[i];
		clkevt = &sp804_clkevt[i];
		clkevt->base	= timer_base;
		clkevt->load	= timer_base + timer->load;
		clkevt->load_h	= timer_base + timer->load_h;
		clkevt->value	= timer_base + timer->value;
		clkevt->value_h	= timer_base + timer->value_h;
		clkevt->ctrl	= timer_base + timer->ctrl;
		clkevt->intclr	= timer_base + timer->intclr;
		clkevt->width	= timer->width;
	}
}

static int __init sp804_of_init(struct device_node *np, struct sp804_timer *timer)
{
	static bool initialized = false;
	void __iomem *base;
	void __iomem *timer1_base;
	void __iomem *timer2_base;
	int irq, ret = -EINVAL;
	u32 irq_num = 0;
	struct clk *clk1, *clk2;
        struct clk *clksrc = NULL;
	const char *name = of_get_property(np, "compatible", NULL);
	//unsigned long *ptr = phys_to_virt(PHY_ADDR_OF_RESET_CTL + PHY_ADDR_OFFSET_OF_TIMER0);
	//*ptr = EN_VALUE_OF_TIMER0; 
	base = of_iomap(np, 0);
	if (!base) {
		pr_err("Failed to iomap\n");
		return -ENXIO;
	}
	timer1_base = base + timer->timer_base[0];
	timer2_base = base + timer->timer_base[1];

	/* Ensure timers are disabled */
	writel(0, timer1_base + timer->ctrl);
	writel(0, timer2_base + timer->ctrl);

	if (initialized || !of_device_is_available(np)) {
		ret = -EINVAL;
		goto err;
	}
        pr_info("tracing Using SP804 '%s' as a clock & events source",np->full_name);
        WARN_ON(clk_register_clkdev(of_clk_get_by_name(np,
			"timer1"), "timer1", "sp804"));
	WARN_ON(clk_register_clkdev(of_clk_get_by_name(np,
				"timer2"), "timer2", "sp804"));
	WARN_ON(clk_register_clkdev(of_clk_get_by_name(np,
				"pclk"), "pclk", "sp804"));
        
        pr_info("tracing register successfully SP804  clock & events source\n");
	clk1 = of_clk_get(np, 0);
	if (IS_ERR(clk1))
		clk1 = NULL;

	/* Get the 2nd clock if the timer has 3 timer clocks */
	if (of_clk_get_parent_count(np) == 3) {
		clk2 = of_clk_get(np, 1);
		if (IS_ERR(clk2)) {
			pr_err("%pOFn clock not found: %d\n", np,
				(int)PTR_ERR(clk2));
			clk2 = NULL;
		}
	} else
		clk2 = clk1;

	irq = irq_of_parse_and_map(np, 0);
	if (irq <= 0)
		goto err;

	sp804_clkevt_init(timer, base);

	of_property_read_u32(np, "arm,sp804-has-irq", &irq_num);
        pr_err("tracing sp804 irq , irq of index_0 %d, and irq_num %d \n", irq, irq_num);
#if !SP804_TEST_IRQ
       	if (1) {
#else
       	if (irq_num == 2) {
#endif
                //clksrc = sp804_dt_init_clk(np, 1, SP804_CLKSRC);
		//if (IS_ERR(clksrc))
		//    goto err;
                pr_info("tracing sp804_clockevents_init clk2\n");
		ret = sp804_clockevents_init(timer2_base, irq, clk2, name);
		if (ret)
			goto err;

		ret = sp804_clocksource_and_sched_clock_init(timer1_base,
							     name, clk1, 1);
		if (ret)
			goto err;
	} else {

                //clksrc = sp804_dt_init_clk(np, 0, SP804_CLKSRC);
		//if (IS_ERR(clksrc))
		//    goto err;
                pr_info("tracing sp804_clockevents_init clk1\n");
		ret = sp804_clockevents_init(timer1_base, irq, clk1, name);
		if (ret)
			goto err;

		ret = sp804_clocksource_and_sched_clock_init(timer2_base,
							     name, clk2, 1);
		if (ret)
			goto err;
	}
	initialized = true;

        pr_info("tracing sp804_of_init sucessfully\n");
        //sp804_sys_init();
	return 0;
err:
	iounmap(base);
	return ret;
}

static int __init arm_sp804_of_init(struct device_node *np)
{
	return sp804_of_init(np, &arm_sp804_timer);
}
TIMER_OF_DECLARE(sp804, "arm,sp804", arm_sp804_of_init);

static int __init hisi_sp804_of_init(struct device_node *np)
{
	return sp804_of_init(np, &hisi_sp804_timer);
}
TIMER_OF_DECLARE(hisi_sp804, "hisilicon,sp804", hisi_sp804_of_init);

static int __init integrator_cp_of_init(struct device_node *np)
{
	static int init_count = 0;
	void __iomem *base;
	int irq, ret = -EINVAL;
	const char *name = of_get_property(np, "compatible", NULL);
	struct clk *clk;

	base = of_iomap(np, 0);
	if (!base) {
		pr_err("Failed to iomap\n");
		return -ENXIO;
	}

	clk = of_clk_get(np, 0);
	if (IS_ERR(clk)) {
		pr_err("Failed to get clock\n");
		return PTR_ERR(clk);
	}

	/* Ensure timer is disabled */
	writel(0, base + arm_sp804_timer.ctrl);

	if (init_count == 2 || !of_device_is_available(np))
		goto err;

	sp804_clkevt_init(&arm_sp804_timer, base);

	if (!init_count) {
		ret = sp804_clocksource_and_sched_clock_init(base,
							     name, clk, 0);
		if (ret)
			goto err;
	} else {
		irq = irq_of_parse_and_map(np, 0);
		if (irq <= 0)
			goto err;

		ret = sp804_clockevents_init(base, irq, clk, name);
		if (ret)
			goto err;
	}

	init_count++;
	return 0;
err:
	iounmap(base);
	return ret;
}
TIMER_OF_DECLARE(intcp, "arm,integrator-cp-timer", integrator_cp_of_init);
static int sp804_value;
static ssize_t sp804_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf);
static struct kobj_attribute sp804_attribute = __ATTR(sp804_value, 0664, sp804_show, NULL);
static struct kobject *sp804_kobj = NULL;
static int sp804_sys_init(void)
{
    int retval;
    sp804_kobj = kobject_create_and_add("sp804", NULL);
    if (!sp804_kobj)
        return -ENOMEM;
    retval = sysfs_create_file(sp804_kobj, &sp804_attribute.attr);
    if (retval)
        kobject_put(sp804_kobj);
    return retval;
}
static ssize_t sp804_show(struct kobject *kobj, struct kobj_attribute *attr, char *buf)
{
	//pr_err("jiffies_read: %x ", jiffies_read());
	pr_err("sp804_read: %x ", sp804_read());
        pr_err("sp804_read: %x ",  ~readl_relaxed(common_clkevt->value));
	return 0;
}
static u64  sp804_read_all(struct clocksource  *cs)
{
        pr_err("sp804_read common_clkevt : %x ",  ~readl_relaxed(common_clkevt->value));
        pr_err("sp804_read common_clkevt ctrl : %x ",  readl_relaxed(common_clkevt->ctrl));
        pr_err("sp804_read sched_clkevt ctrl : %x ",  readl_relaxed(sched_clkevt->ctrl));
	pr_err("sp804_read sched_clkevt : %x ", sp804_read());
	return ~readl_relaxed(sched_clkevt->value);
}
