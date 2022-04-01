#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/init.h>
#include <linux/kernel.h> 
 
#define SIZE 8

MODULE_LICENSE("Dual BSD/GPL");
MODULE_AUTHOR("Liran B.H");


static int addr[SIZE];
static int count;
module_param_array(addr, int, &count,  0660);

enum irq_type {
	IRQ_TYPE_LEVEL,
	IRQ_TYPE_EDGE,
	IRQ_TYPE_POLLING
};


static int irq_type = IRQ_TYPE_LEVEL; // default

static int irqtype_op_write_handler(const char *val,
				  const struct kernel_param *kp)
{
	char valcp[16];
	char *s;

	strncpy(valcp, val, 16);
	valcp[15] = '\0';

	s = strstrip(valcp);

	if (strcmp(s, "level") == 0)
		irq_type = IRQ_TYPE_LEVEL;
	else if (strcmp(s, "edge") == 0)
		irq_type = IRQ_TYPE_EDGE;
	else if (strcmp(s, "polling") == 0)
		irq_type = IRQ_TYPE_POLLING;
	else
		return -EINVAL;

	return 0;
}

static int irqtype_op_read_handler(char *buffer, const struct kernel_param *kp)
{
	switch (irq_type) {
	case IRQ_TYPE_LEVEL:
		strcpy(buffer, "level");
		break;

	case IRQ_TYPE_EDGE:
		strcpy(buffer, "edge");
		break;

	case IRQ_TYPE_POLLING:
		strcpy(buffer, "polling");
		break;

	default:
		strcpy(buffer, "error");
		break;
	}

	return strlen(buffer);
}

static const struct kernel_param_ops irqtype_op_ops = {
	.set = irqtype_op_write_handler,
	.get = irqtype_op_read_handler
};

module_param_cb(irqtype, &irqtype_op_ops, NULL, 0600);

static int my_set(const char *val, const struct kernel_param *kp)
{
	int n = 0, ret;

	ret = kstrtoint(val, 10, &n);
	if (ret != 0 || n < 1 || n > 32)
		return -EINVAL;

	return param_set_int(val, kp);
}

static const struct kernel_param_ops param_ops = {
	.set	= my_set,
	.get	= param_get_int,
};

static int num;
module_param_cb(simpcb, &param_ops,&num, 0664);

static int debug=0;
module_param(debug,int,0660);

static int irq=10;
module_param(irq,int,0660);

static char *devname = "simpdev";
module_param(devname,charp,0660);


static int simple_init(void)
{
	printk(KERN_WARNING "hello... irq=%d name=%s debug=%d\n",irq,devname,debug);
	return 0;
}

static void simple_cleanup(void)
{
	printk(KERN_WARNING "bye ...irq=%d name=%s debug=%d\n",irq,devname,debug);
}

module_init(simple_init);
module_exit(simple_cleanup);
