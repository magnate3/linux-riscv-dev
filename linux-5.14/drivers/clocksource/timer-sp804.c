
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
#include <linux/device.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/of_clk.h>
#include <linux/of_irq.h>
#include <linux/sched_clock.h>
#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/amba/bus.h>

#include "timer-sp.h"

#define SP804_TEST  1
#define SP804_TEST_IRQ 1
#define SP804_TEST_DATE 1

#define PHY_ADDR_OF_RESET_CTL    0x43060000
#define PHY_ADDR_OFFSET_OF_TIMER0    0xC
#define EN_VALUE_OF_TIMER0  1 << 12
static bool initialized = false;
struct sp804_private {
	struct sp804_clkevt sp804_clkevt;
	struct clock_event_device clkevt;
	//struct device *dev;
};

static struct sp804_timer arm_sp804_timer = {
	.load		= TIMER_LOAD,
	.value		= TIMER_VALUE,
	.ctrl		= TIMER_CTRL,
	.intclr		= TIMER_INTCLR,
	.timer_base	= {TIMER_1_BASE, TIMER_2_BASE},
	.width		= 32,
};
#if SP804_TEST
static u64  sp804_read_all(struct clocksource  *cs);
static struct sp804_clkevt *g_common_clkevt;
#endif
static struct sp804_clkevt * sp804_clkevt_get(struct sp804_private *priv, void __iomem *base)
{
	int i;

	for (i = 0; i < NR_TIMERS; i++) {
		if (priv[i].sp804_clkevt.base == base)
			return &(priv[i].sp804_clkevt);
	}

	/* It's impossible to reach here */
	WARN_ON(1);

	return NULL;
}
static struct clock_event_device* sp804_clkevt_dev_get(struct sp804_private *priv, void __iomem *base)
{
	int i;

	for (i = 0; i < NR_TIMERS; i++) {
		if (priv[i].sp804_clkevt.base == base)
			return &(priv[i].clkevt);
	}

	/* It's impossible to reach here */
	WARN_ON(1);

	return NULL;
}
static struct sp804_private *
to_priv(struct clock_event_device *clkevt)
{
	return container_of(clkevt, struct sp804_private, clkevt);
}


#if SP804_TEST_IRQ
static unsigned long  sp804_intc_count = 0;
#endif
static irqreturn_t sp804_timer_interrupt(int irq, void *dev_id)
{
        struct sp804_clkevt *common_clkevt;
	struct clock_event_device *evt = (struct clock_event_device *)dev_id;
	struct sp804_private *priv = to_priv(evt);
	common_clkevt = &(priv->sp804_clkevt);
#if SP804_TEST_IRQ
	//pr_err("sp804 timers interrupt happens ");
	if(0 == ++sp804_intc_count % 10){
	    //pr_err("sp804 10 timers of interrupt happens again");
	}
	/* clear the interrupt */
	writel(1, common_clkevt->intclr);
	return IRQ_HANDLED;
#endif

	/* clear the interrupt */
	writel(1, common_clkevt->intclr);

	evt->event_handler(evt);

	return IRQ_HANDLED;
}

static inline void timer_shutdown(struct clock_event_device *evt)
{
        struct sp804_clkevt *common_clkevt;
	struct sp804_private *priv = to_priv(evt);
	common_clkevt = &(priv->sp804_clkevt);
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
        struct sp804_clkevt *common_clkevt;
	struct sp804_private *priv = to_priv(evt);
	common_clkevt = &(priv->sp804_clkevt);
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
        struct sp804_clkevt *common_clkevt;
	struct sp804_private *priv = to_priv(evt);
	common_clkevt = &(priv->sp804_clkevt);
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

static void sp804_clockevents_dev_init(struct clock_event_device *evt) 
{
	evt->features	= CLOCK_EVT_FEAT_PERIODIC |
	       		  CLOCK_EVT_FEAT_ONESHOT |
	       		  CLOCK_EVT_FEAT_DYNIRQ,
	evt->set_state_shutdown	= sp804_shutdown;
	evt->set_state_periodic	= sp804_set_periodic;
	evt->set_state_oneshot	= sp804_shutdown;
	evt->tick_resume		= sp804_shutdown;
	evt->set_next_event		= sp804_set_next_event;
	evt->rating			= 300;
};

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
static int __init sp804_clockevents_init(struct sp804_private * priv, void __iomem *base, unsigned int irq,
					 struct clk *clk, const char *name)
{
	struct clock_event_device *evt = sp804_clkevt_dev_get(priv, base);
        static struct sp804_clkevt *common_clkevt;
	long rate;
        sp804_clockevents_dev_init(evt);
	rate = sp804_get_clock_rate(clk, name);
	if (rate < 0)
		return -EINVAL;

	common_clkevt = sp804_clkevt_get(priv,base);
	common_clkevt->reload = DIV_ROUND_CLOSEST(rate, HZ);
	evt->name = name;
	evt->irq = irq;
	evt->cpumask = cpu_possible_mask;


#if SP804_TEST_IRQ
	writel(0, common_clkevt->ctrl);
	writel(0x0ffffff, common_clkevt->load);
	writel(0x0ffffff, common_clkevt->value);
	//writel(0xffffffff, common_clkevt->load);
	//writel(0xffffffff, common_clkevt->value);
	if (common_clkevt->width == 64) {
		writel(0xffffffff, common_clkevt->load_h);
		writel(0xffffffff, common_clkevt->value_h);
	}
	writel(TIMER_CTRL_32BIT | TIMER_CTRL_ENABLE | TIMER_CTRL_PERIODIC | TIMER_CTRL_IE,
		common_clkevt->ctrl);
#else
	writel(0, common_clkevt->ctrl);
#endif
	pr_err("sp804 request_irq num %u and rate %ld \n", irq, rate);
	if (request_irq(irq, sp804_timer_interrupt, IRQF_TIMER | IRQF_IRQPOLL,
			"timer", evt))
		pr_err("sp804 request_irq() failed\n");
	clockevents_config_and_register(evt, rate, 0xf, 0xffffffff);
#if SP804_TEST
    if(!initialized)
    {
        g_common_clkevt = common_clkevt;
    }
#endif
	return 0;
}
static struct sp804_clkevt *g_sched_clkevt;

static u64 notrace sp804_read(void)
{
	return ~readl_relaxed(g_sched_clkevt->value);
}
static int __init sp804_clocksource_and_sched_clock_init(struct sp804_private * priv, void __iomem *base,
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

	clkevt = sp804_clkevt_get(priv,base);

	writel(0, clkevt->ctrl);
#if SP804_TEST_DATE
	writel(0xffffff, clkevt->load);
	writel(0xffffff, clkevt->value);
#else
	writel(0xffffffff, clkevt->load);
	writel(0xffffffff, clkevt->value);
#endif
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
#if 0
//#if SP804_TEST
	clocksource_mmio_init(clkevt->value, name,
		rate/4, 200, 32, sp804_read_all);
#else
	clocksource_mmio_init(clkevt->value, name,
		rate/4, 200, 32, clocksource_mmio_readl_down);
#endif
	if (use_sched_clock) {
		g_sched_clkevt = clkevt;
		sched_clock_register(sp804_read, 32, rate);
	}

	return 0;
}
static void sp804_clkevt_init(struct sp804_private *priv, struct sp804_timer * timer, void __iomem *base)
{
	int i;
	for (i = 0; i < NR_TIMERS; i++) {
		void __iomem *timer_base;
		struct sp804_clkevt *clkevt;

		timer_base = base + timer->timer_base[i];
		clkevt = &(priv[i].sp804_clkevt);
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
static int sp804_probe(struct platform_device *pdev)
{
        struct device_node *np = pdev->dev.of_node;
	void __iomem *base;
	void __iomem *timer1_base;
	void __iomem *timer2_base;
	int irq, ret = -EINVAL;
	u32 irq_num = 0;
	struct clk *clk1, *clk2;
	struct sp804_private *priv;
	//const char *name = pdev->name;
	const char *name = np->full_name;
        pr_info("tracing Using SP804 %s as a clock & events source",np->full_name);
	base = of_iomap(np, 0);
	if (!base) {
		pr_err("Failed to iomap\n");
		return -ENXIO;
	}
	priv = devm_kzalloc(&pdev->dev, sizeof(struct  sp804_private)*NR_TIMERS, GFP_KERNEL);
	if (!priv)
		return -ENOMEM;
	//sp804_init_priv(priv);
	struct sp804_timer *timer = &arm_sp804_timer;
	timer1_base = base + timer->timer_base[0];
	timer2_base = base + timer->timer_base[1];

	/* Ensure timers are disabled */
	writel(0, timer1_base + timer->ctrl);
	writel(0, timer2_base + timer->ctrl);

	if (!of_device_is_available(np)) {
		ret = -EINVAL;
		goto err;
	}
        WARN_ON(clk_register_clkdev(of_clk_get_by_name(np,
			"timer1"), "timer1", "sp804"));
	WARN_ON(clk_register_clkdev(of_clk_get_by_name(np,
				"timer2"), "timer2", "sp804"));
	WARN_ON(clk_register_clkdev(of_clk_get_by_name(np,
				"pclk"), "pclk", "sp804"));
        
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

	sp804_clkevt_init(priv, timer, base);

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
		ret = sp804_clockevents_init(priv,timer2_base, irq, clk2, name);
		if (ret)
			goto err;

		ret = sp804_clocksource_and_sched_clock_init(priv,timer1_base,
							     name, clk1, initialized ? 0 : 1);
		if (ret)
			goto err;
	} else {

                //clksrc = sp804_dt_init_clk(np, 0, SP804_CLKSRC);
		//if (IS_ERR(clksrc))
		//    goto err;
		ret = sp804_clockevents_init(priv, timer1_base, irq, clk1, name);
		if (ret)
			goto err;

		ret = sp804_clocksource_and_sched_clock_init(priv, timer2_base,
							     name, clk2, initialized ? 0 : 1);
		if (ret)
			goto err;
	}

        pr_info("tracing sp804_probe sucessfully\n");
	//priv->dev = &pdev->dev;
        //sp804_sys_init();
	if (!initialized)
	{
	    initialized = true;
	}
	return 0;
err:
	iounmap(base);
	devm_kfree(&pdev->dev, priv);
	return ret;
}

static u64  sp804_read_all(struct clocksource  *cs)
{
        pr_err("sp804_read common_clkevt : %x ",  ~readl_relaxed(g_common_clkevt->value));
        pr_err("sp804_read common_clkevt ctrl : %x ",  readl_relaxed(g_common_clkevt->ctrl));
        pr_err("sp804_read sched_clkevt ctrl : %x ",  readl_relaxed(g_sched_clkevt->ctrl));
	pr_err("sp804_read sched_clkevt : %x ", sp804_read());
	return ~readl_relaxed(g_sched_clkevt->value);
}
static int sp804_remove(struct platform_device *pdev)
{
	return -EBUSY; /* cannot unregister clockevent */
}

static const struct of_device_id sp804_of_match[] = {
	{ .compatible = "arm,sp804", },
	{},
};
MODULE_DEVICE_TABLE(of, sp804_of_match);

static struct platform_driver sp804_driver = {
	.probe	= sp804_probe,
	.remove = sp804_remove,
	.driver	= {
		.name = "sp804-timer",
		.of_match_table = of_match_ptr(sp804_of_match),
	},
};
module_platform_driver(sp804_driver);

//module_amba_driver(sp804_driver);
//static int __init sp804_timer_init(void)
//{
// printk(KERN_ALERT "tracing amba sp804 driver_register\n");
// return amba_driver_register(&sp804_driver);
//}
//module_init(sp804_timer_init);
//
//static void __exit sp804_timer_exit(void)
//{
// amba_driver_unregister(&sp804_driver);
//}
//module_exit(sp804_timer_exit);
MODULE_ALIAS("platform:sp804-timer");
MODULE_DESCRIPTION("sp804 timer driver");
MODULE_LICENSE("GPL v2");
