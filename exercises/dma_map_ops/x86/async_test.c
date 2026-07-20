#include <linux/module.h>
#include <linux/platform_device.h>
#include <linux/of.h>
#include <linux/delay.h>

//#define USE_ASYNC

static int async_demo1_probe(struct platform_device *pdev)
{
	if (pdev->dev.parent)
		printk("%s: parent %s\n", __func__, dev_name(pdev->dev.parent));
	printk("%s enter.\n", __func__);
	msleep(4000);
	printk("%s exit.\n", __func__);
	return 0;
}

static int async_demo1_remove(struct platform_device *pdev)
{
	printk("%s enter.\n", __func__);
	return 0;
}

static const struct of_device_id async_demo1_dt_ids[] = {
	{ .compatible = "async_demo1", },
	{  }
};
MODULE_DEVICE_TABLE(of, async_demo1_dt_ids);

static struct platform_driver async_demo1_driver = {
	.probe		= async_demo1_probe,
	.remove		= async_demo1_remove,
	.driver		= {
		.name	= "async_demo1",
		.of_match_table = of_match_ptr(async_demo1_dt_ids),
#ifdef USE_ASYNC
		.probe_type = PROBE_PREFER_ASYNCHRONOUS,
#else
		.probe_type = PROBE_FORCE_SYNCHRONOUS,
#endif
	},
};

static int async_demo2_probe(struct platform_device *pdev)
{
	if (pdev->dev.parent)
		printk("%s: parent %s\n", __func__, dev_name(pdev->dev.parent));
	printk("%s enter.\n", __func__);
	msleep(3000);
	printk("%s exit.\n", __func__);
	return 0;
}

static int async_demo2_remove(struct platform_device *pdev)
{
	printk("%s enter.\n", __func__);
	return 0;
}

static const struct of_device_id async_demo2_dt_ids[] = {
	{ .compatible = "async_demo2", },
	{  }
};
MODULE_DEVICE_TABLE(of, async_demo2_dt_ids);

static struct platform_driver async_demo2_driver = {
	.probe		= async_demo2_probe,
	.remove		= async_demo2_remove,
	.driver		= {
		.name	= "async_demo2",
		.of_match_table = of_match_ptr(async_demo2_dt_ids),
#ifdef USE_ASYNC
		.probe_type = PROBE_PREFER_ASYNCHRONOUS,
#else
		.probe_type = PROBE_FORCE_SYNCHRONOUS,
#endif
	},
};

static int async_demo3_probe(struct platform_device *pdev)
{
	printk("%s enter.\n", __func__);
	msleep(2000);
	printk("%s exit.\n", __func__);
	return 0;
}

static int async_demo3_remove(struct platform_device *pdev)
{
	printk("%s enter.\n", __func__);
	return 0;
}

static const struct of_device_id async_demo3_dt_ids[] = {
	{ .compatible = "async_demo3", },
	{  }
};
MODULE_DEVICE_TABLE(of, async_demo3_dt_ids);

static struct platform_driver async_demo3_driver = {
	.probe		= async_demo3_probe,
	.remove		= async_demo3_remove,
	.driver		= {
		.name	= "async_demo3",
		.of_match_table = of_match_ptr(async_demo3_dt_ids),
#ifdef USE_ASYNC
		.probe_type = PROBE_PREFER_ASYNCHRONOUS,
#else
		.probe_type = PROBE_FORCE_SYNCHRONOUS,
#endif
	},
};

static int async_demo4_probe(struct platform_device *pdev)
{
	printk("%s enter.\n", __func__);
	msleep(1000);
	printk("%s exit.\n", __func__);
	return 0;
}

static int async_demo4_remove(struct platform_device *pdev)
{
	printk("%s enter.\n", __func__);
	return 0;
}

static const struct of_device_id async_demo4_dt_ids[] = {
	{ .compatible = "async_demo4", },
	{  }
};
MODULE_DEVICE_TABLE(of, async_demo4_dt_ids);

static struct platform_driver async_demo4_driver = {
	.probe		= async_demo4_probe,
	.remove		= async_demo4_remove,
	.driver		= {
		.name	= "async_demo4",
		.of_match_table = of_match_ptr(async_demo4_dt_ids),
#ifdef USE_ASYNC
		.probe_type = PROBE_PREFER_ASYNCHRONOUS,
#else
		.probe_type = PROBE_FORCE_SYNCHRONOUS,
#endif
	},
};
static struct platform_device *pdev[4];
static __init int async_demo_init(void)
{
	printk("%s enter.\n", __func__);

	printk("before async_demo1\n");
	platform_driver_register(&async_demo1_driver);
	printk("before async_demo2\n");
	platform_driver_register(&async_demo2_driver);
	printk("before async_demo3\n");
	platform_driver_register(&async_demo3_driver);
	printk("before async_demo4\n");
	platform_driver_register(&async_demo4_driver);
	printk("\n\n Register Platform Device\n");
	pdev[0] = platform_device_register_simple("async_demo1", 0, NULL, 0);
	pdev[1] = platform_device_register_simple("async_demo2", 0, NULL, 0);
	pdev[2] = platform_device_register_simple("async_demo3", 0, NULL, 0);
	pdev[3] = platform_device_register_simple("async_demo4", 0, NULL, 0);

	return 0;
}

static __exit void async_demo_exit(void)
{
	printk("%s enter.\n", __func__);
	platform_device_unregister(pdev[0]);
	platform_device_unregister(pdev[1]);
	platform_device_unregister(pdev[2]);
	platform_device_unregister(pdev[3]);

	platform_driver_unregister(&async_demo1_driver);
	platform_driver_unregister(&async_demo2_driver);
	platform_driver_unregister(&async_demo3_driver);
	platform_driver_unregister(&async_demo4_driver);
}

module_init(async_demo_init);
module_exit(async_demo_exit);

MODULE_LICENSE("GPL");
