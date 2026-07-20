#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt
#include <linux/clk.h>
#include <linux/clk-provider.h>
#include <linux/crc32.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/types.h>
#include <linux/circ_buf.h>
#include <linux/slab.h>
#include <linux/init.h>
#include <linux/io.h>
#include <linux/gpio.h>
#include <linux/gpio/consumer.h>
#include <linux/interrupt.h>
#include <linux/netdevice.h>
#include <linux/etherdevice.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include <linux/phylink.h>
#include <linux/of.h>
#include <linux/of_device.h>
#include <linux/of_gpio.h>
#include <linux/of_mdio.h>
#include <linux/of_net.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <linux/iopoll.h>
#include <linux/pm_runtime.h>
#include<linux/of_irq.h>
///////////////////////////////////////////
#include <linux/sysfs.h>
#include "macb.h"
#include "common.h"
#define ETH_NAME "eth2"
char g_gmac_name[128];
struct macb *g_mac_bp = NULL;
static struct device *dev;
static struct kobject *root, *gmac;
static ssize_t gmac_show(struct device *dev, struct device_attribute *attr, char *buf) {
    return sprintf(buf, "gmac name : %s\n", g_gmac_name);
}

static struct macb * get_gmac(const char * eth_name)
{
    struct net_device * dst_dev;
    struct macb *bp = NULL ;
    dst_dev = dev_get_by_name(&init_net,eth_name);
    pr_err("******** get  nic %s    *** \n", eth_name);
    if(!dst_dev)
    {
	   	 pr_err("******** nic %s not exist *** \n", eth_name);
    }
    else
    {
        bp = netdev_priv(dst_dev);
	dev_put(dst_dev);
    }
    return bp;
 }
static void macb_reset_hw(struct macb *bp)
{
	struct macb_queue *queue;
	unsigned int q;
	u32 ctrl = macb_readl(bp, NCR);

	/* Disable RX and TX (XXX: Should we halt the transmission
	 * more gracefully?)
	 */
	ctrl &= ~(MACB_BIT(RE) | MACB_BIT(TE));

	/* Clear the stats registers (XXX: Update stats first?) */
	ctrl |= MACB_BIT(CLRSTAT);

	macb_writel(bp, NCR, ctrl);

	/* Clear all status flags */
	macb_writel(bp, TSR, -1);
	macb_writel(bp, RSR, -1);

	/* Disable all interrupts */
	for (q = 0, queue = bp->queues; q < bp->num_queues; ++q, ++queue) {
		queue_writel(queue, IDR, -1);
		queue_readl(queue, ISR);
		if (bp->caps & MACB_CAPS_ISR_CLEAR_ON_WRITE)
			queue_writel(queue, ISR, -1);
	}
}


static ssize_t gmac_store(struct device *dev, struct device_attribute *attr, const char *buf, size_t count) {
    u32 read, reg_addr, page, reg_value;
    sscanf(buf, "%x,%x,%x,%x",&read, &reg_addr,&page, &reg_value);
    pr_err("%s read %d, reg add %x ,page %x, reg value %x\n",__func__,read, reg_addr,page, reg_value);
    if (SET_GMAC_OP !=read && (NULL == g_mac_bp))
    {
	 pr_err("gmac not set  or not match driver and return \n ");
	 return count;
    }
    if (WRITE_GMAC_OP == read)
    {
             pr_err("write gmac name %s\n", g_gmac_name);
    }
    else if(READ_GMAC_OP == read)
    {
             pr_err("read gmac name %s\n", g_gmac_name);
    }
    else if (RESET_GMAC_OP == read)
    {
             pr_err("reset gmac name %s\n", g_gmac_name);
             macb_reset_hw(g_mac_bp); 
    }
    else if (CLOSE_GMAC_OP == read)
    {
             pr_err("close gmac name %s\n", g_gmac_name);
             g_mac_bp->dev->netdev_ops->ndo_stop(g_mac_bp->dev); 
    }
    else if (SET_GMAC_OP == read)
    {
       snprintf(g_gmac_name,sizeof(g_gmac_name), "eth%d", reg_value);
       pr_err("set gmac name %s\n", g_gmac_name);
       g_mac_bp = get_gmac(g_gmac_name);
    }
    return count;

}


static struct device_attribute gmac_dev_attr = __ATTR(dev1, 0660, gmac_show, gmac_store);

static int sysfs_macb_init(void) {
    snprintf(g_gmac_name,sizeof(g_gmac_name), "%s", ETH_NAME);
    g_mac_bp = get_gmac(g_gmac_name);
    dev = root_device_register("sysfs_macb");
    root = &dev->kobj;
    gmac = kobject_create_and_add("nicDbg", root);
    sysfs_create_file(gmac, &gmac_dev_attr.attr);
    return 0;
}

static void sysfs_macb_exit(void) {
    printk(KERN_INFO "sysfs macb exit\n");
    sysfs_remove_file(root, &gmac_dev_attr.attr);
    kobject_put(gmac);
    root_device_unregister(dev);
    g_mac_bp = NULL;
}
module_init(sysfs_macb_init);
module_exit(sysfs_macb_exit);
MODULE_LICENSE("GPL");
