#include <linux/module.h>   
   
static int module_test_init(void)   
{   
    printk("%s: call %s\n", KBUILD_MODNAME, __FUNCTION__);   
    return 0;   
}   
   
static void module_test_exit(void)   
{   
    printk("%s: call %s\n", KBUILD_MODNAME, __FUNCTION__);   
}   
   
module_init(module_test_init);   
module_exit(module_test_exit);   
   
MODULE_AUTHOR("<kuriking@gmail.com>");   
MODULE_DESCRIPTION("Test module");   
MODULE_LICENSE("GPL");   
