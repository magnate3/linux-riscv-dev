#include <linux/init.h>
#include <linux/module.h>
MODULE_LICENSE("Dual BSD/GPL");
static int disableCache_init(void)
{
        printk(KERN_ALERT "Disabling L1 and L2 caches.\n");
        __asm__(".intel_syntax noprefix\n\t"
                "mov    rax,cr0\n\t"
                "or     rax,(1 << 30)\n\t"
                "mov    cr0,rax\n\t"
                "wbinvd\n\t"
                ".att_syntax noprefix\n\t"
        : : : "rax" );
        return 0;
}
static void disableCache_exit(void)
{
        printk(KERN_ALERT "Enabling L1 and L2 caches.\n");
        __asm__(".intel_syntax noprefix\n\t"
                "mov    rax,cr0\n\t"
                "and     rax,~(1 << 30)\n\t"
                "mov    cr0,rax\n\t"
                "wbinvd\n\t"
                ".att_syntax noprefix\n\t"
        : : : "rax" );
}
module_init(disableCache_init);
module_exit(disableCache_exit);
