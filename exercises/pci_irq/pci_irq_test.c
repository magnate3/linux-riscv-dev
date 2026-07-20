
#include <linux/init.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/interrupt.h>
#ifdef SCHEDULE_IN_INTR
#include <linux/sched.h>
#endif

#include <asm/io.h>
#include <asm/irq.h>
#include <asm/uaccess.h>

#define TEST_VID 0x19e5
#define TEST_DID 0x0200

#define DRV_NAME "Test Driver"

//static DEFINE_PCI_DEVICE_TABLE(test_pci_table) = {
//	{PCI_DEVICE(TEST_VID, TEST_DID), },
//	{ },
//};

static const struct pci_device_id test_pci_table[] = {
	{ PCI_DEVICE(0x19e5, 0x0200), },
        {},
};
void dummy_tasklet_fun(unsigned long);
MODULE_DEVICE_TABLE(pci, test_pci_table);
DECLARE_TASKLET(dummy_tasklet, dummy_tasklet_fun, 0);


void dummy_tasklet_fun(unsigned long data)
{

	printk(KERN_ALERT "Test dummy tasklet invoked\n");
}

static irqreturn_t
test_isr(int irq, void *dev_id)
{
	printk(KERN_ALERT "Test ISR invoked\n");
	tasklet_schedule(&dummy_tasklet);
#ifdef SCHEDULE_IN_INTR
	printk(KERN_ALERT "Death wish, scheduling out\n");
	schedule();
#endif
	return IRQ_HANDLED;
}

static int
test_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	uint8_t irq;
	int ret;

	printk(KERN_ALERT "\nProbed");
	pci_enable_device(pdev);

	pci_read_config_byte(pdev, PCI_INTERRUPT_LINE, &irq);

	printk(KERN_ALERT "irq %u", irq);

	if ((ret =request_irq(irq, test_isr, IRQF_SHARED, "test_isr", pdev)) != 0) {
		printk(KERN_ALERT "\n ISR registration failure, %u", ret);
		return -1;
	}

	printk(KERN_ALERT "ISR registered");
	return 0;
}

static void
test_remove(struct pci_dev *pdev)
{
	uint8_t irq;
	
	pci_read_config_byte(pdev, PCI_INTERRUPT_LINE, &irq);
	free_irq(irq, pdev);
}

static struct pci_driver test_pci_driver = {
	.name 		= DRV_NAME,
	.id_table 	= test_pci_table,
	.probe		= test_probe,
	.remove		= test_remove,
};

static int
test_init(void)
{
	printk(KERN_ALERT "\nHello, world");
	return pci_register_driver(&test_pci_driver);
}

static void
test_exit(void)
{
	pci_unregister_driver(&test_pci_driver);
}


module_init(test_init);
module_exit(test_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("xyz");
MODULE_DESCRIPTION("Test Driver");
