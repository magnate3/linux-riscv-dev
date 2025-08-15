/*************************************************************************
    > File Name: mod1.c
    > Author: Shawn Guo
    > Mail: iseanxp@gmail.com 
    > Created Time: 2013年12月02日 星期一 14时25分52秒
    > Last Changed: 
    > Notes:	EXPORT_SYMBOL宏-使用练习 
*************************************************************************/
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>

static int func1(void)
{
	    printk("In Func: %s...\n",__func__);
		return 0;
}

EXPORT_SYMBOL(func1);	//导出fun1函数符号,函数地址和名称
//导出的符号表可以在中间生成文件Module.symvers中查看.
//如果别的模块需要该符号, 需要在Makefile中制定该Module.symvers的路径
//eg.	KBUILD_EXTRA_SYMBOLS=/mod_a/Module.symvers


static int __init hello_init(void)
{
	    printk("Module 1, Init!\n");
		func1();
	    return 0;
}


static void __exit hello_exit(void)
{
	    printk("Module 1, Exit!\n");
}


module_init(hello_init);
module_exit(hello_exit);

MODULE_LICENSE("GPL");
