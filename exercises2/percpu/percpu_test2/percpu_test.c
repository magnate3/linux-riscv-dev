
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/delay.h>

MODULE_LICENSE("GPL");
struct my_struct {
    unsigned int *values;
};
static struct my_struct my;
static int test_percpu_init(void)
{
	int num_cpus = num_online_cpus();
	int i = 0;
        unsigned int *value;
        int cpu;
        my.values= __alloc_percpu(sizeof(unsigned int), 8);
	pr_info("Number of cpus available:%d\n", num_cpus);
	for (i = 0; i < num_cpus; i++) {
                value = per_cpu_ptr(my.values, cpu);
		pr_info("Value of counter is %d at Processor:%d\n", *value, i);
	}
        cpu = get_cpu();
        value = per_cpu_ptr(my.values, cpu);
        *value = 99;
        put_cpu();	
	pr_info("Printing counter value of all processor after updating current processor:%d\n", smp_processor_id());

 
	for (i = 0; i < num_cpus; i++) {
                value = per_cpu_ptr(my.values, i);
		pr_info("Value of counter is %d at Processor:%d\n", *value, i);
	}


    return 0;
}

static void test_percpu_exit(void)
{
    free_percpu(my.values);
}

module_init(test_percpu_init);
module_exit(test_percpu_exit);
