
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");
DEFINE_PER_CPU(int, counter);
static int test_percpu_init(void)
{
	int num_cpus = num_online_cpus();
	int i = 0;
	int val;

	pr_info("Number of cpus available:%d\n", num_cpus);
	for (i = 0; i < num_cpus; i++) {
		int value = per_cpu(counter, i);
		pr_info("Value of counter is %d at Processor:%d\n", value, i);
	}

	val = get_cpu_var(counter);
	get_cpu_var(counter) = 10;
	put_cpu_var(counter);
	
	pr_info("Printing counter value of all processor after updating current processor:%d\n",
			smp_processor_id());

	for (i = 0; i < num_cpus; i++) {
		int value = per_cpu(counter, i);
		pr_info("Value of counter is %d at Processor:%d\n", value, i);
	}


    return 0;
}

static void test_percpu_exit(void)
{
}

module_init(test_percpu_init);
module_exit(test_percpu_exit);