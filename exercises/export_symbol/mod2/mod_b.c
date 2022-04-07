/*************************************************************************
  > File Name: mod_b.c
  > Author: Shawn Guo
  > Mail: iseanxp@gmail.com 
  > Created Time: 2013年12月02日 星期一 14时38分12秒
  > Last Changed: 
  > Notes:		EXPORT_SYMBOL的使用练习 
  >				make以后生成mod2.ko
  >				加载mod2前，需要先加载mod1.
 *************************************************************************/

#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/module.h>

static int func2(void)
{
	extern int func1(void);	//在这个模块中声明了其他的函数符号
	func1();		
	printk("In Func: %s...\n",__func__);
	return 0;
}

static int __init hello_init(void)
{
	printk("Module 2, Init!\n");
	func2();
	return 0;
}

static void __exit hello_exit(void)
{
	printk("Module 2, Exit!\n");
}

module_init(hello_init);
module_exit(hello_exit);

MODULE_LICENSE("GPL");
