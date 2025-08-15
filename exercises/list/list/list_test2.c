/*************************************************************************
    > File Name: list_test.c
    > Author: baiy
    >
    > Created Time: 2021-04-06-14:34:52
    > Func:  测试Linux 内核链表, 参考文档：
    > https://www.yuque.com/docs/share/779a01b2-8660-40ed-8289-3feaf1b60f53?#
 ************************************************************************/
#define pr_fmt(fmt) "[%s:%d]: " fmt, __func__, __LINE__
#include <linux/uaccess.h>
#include <linux/types.h>
#include <linux/kernel.h>
#include <linux/sched.h>
#include <linux/notifier.h>
#include <linux/init.h>
#include <linux/types.h>
#include <linux/module.h>
#include <linux/kthread.h>
#include <linux/version.h>
#include <linux/slab.h>
#if LINUX_VERSION_CODE > KERNEL_VERSION(3, 3, 0)
    #include <asm/switch_to.h>
#else
    #include <asm/system.h>
#endif

#include <linux/proc_fs.h>
#include <linux/mutex.h>
#include <linux/list.h>


struct list_test_info {
	struct list_head head;  // 链表头部结构体
};

struct list_test_node {
	unsigned int num;		// 数据域
	struct list_head node;	// 链表域
};

static void list_test(void)
{
	int i = 0;
	struct list_test_info head1, head2;

	pr_info("Test List [E]\n");

	// 初始化链表头
	INIT_LIST_HEAD(&head1.head);
	INIT_LIST_HEAD(&head2.head);

	// 申请并添加链表
	pr_info("create list\n");
	for(i=0; i<5; ++i){
		struct list_test_node *n1 = kzalloc(sizeof(struct list_test_node), GFP_KERNEL);
		struct list_test_node *n2 = kzalloc(sizeof(struct list_test_node), GFP_KERNEL);
		INIT_LIST_HEAD(&n1->node);
		INIT_LIST_HEAD(&n2->node);
		n1->num = i;
		n2->num = i + 5;
		list_add_tail(&n1->node,&head1.head); // list_add 把链表添加在链表首部, list_add_tail把数据添加到链表尾部
		list_add_tail(&n2->node,&head2.head); // 注：因为是双向链表，所以不用考虑遍历到尾部的性能问题，只需要在prev添加即可
	}


	// 链表的遍历
	pr_info("traversal list\n");
	{
		struct list_head *p;
		struct list_test_node *n1, *n2;
		pr_info("Dump List n1 used list_for_each\n");
		// 方式一
		list_for_each(p, &head1.head){
			n1 = list_entry(p, struct list_test_node, node);
			pr_info("n1 is %d\n",n1->num);
		}

		// 方式二
		pr_info("Dump List n2 used list_for_each_entry\n");
		list_for_each_entry(n2, &head2.head, node){ // list_for_each_entry_reverse 反向遍历
			pr_info("n2 is %d\n",n2->num);
		}
	}

	// 链表的添加
	// list_add / list_add_tail	添加到链表头部或尾部

	// 链表的删除
	// list_del / list_del_init  删除并对其初始化

	// 链表的判空
	// list_empty 直接判断 READ_ONCE(head->next) == head; 即可

	// 获取链表首元素 和 尾部元素
	// list_first_entry / list_last_entry


	// 链表的移动 把一个链表节点从当前链表删除并添加到另一个链表中
	// 就是 list_del 和 list_add的组合
	// list_move(&n1, &head1.head);
    // list_move_tail(&n2, &head1.head);


	// 链表的合并
	pr_info("splice list\n"); // splice： 黏接
	{
		struct list_test_node *n1, *n2;

		list_splice_init(&head1.head,&head2.head);
		pr_info("Dump List n1 used list_for_each_entry\n");
		list_for_each_entry(n1, &head1.head, node){
			pr_info("n1 is %d\n",n1->num);
		}

		pr_info("Dump List n2 used list_for_each_entry\n");
		list_for_each_entry(n2, &head2.head, node){ // list_for_each_entry_reverse 反向遍历
			pr_info("n2 is %d\n",n2->num);
		}
	}

	// 链表的左移
	pr_info("rotate_left list\n"); // rotate ：旋转
	{
		struct list_test_node *n2;
		list_rotate_left(&head2.head);		// 左移一个元素：函数用于将链表第一节点移动到链表的末尾。
		pr_info("Dump List n2 used list_for_each_entry\n");
		list_for_each_entry(n2, &head2.head, node){ // list_for_each_entry_reverse 反向遍历
			pr_info("n2 is %d\n",n2->num);
		}
	}

	// 释放链表
	pr_info("delete list\n");
	{
		struct list_test_node *n1, *tmpn1;
		struct list_test_node *n2, *tmpn2;
		pr_info("Delete List n1 used list_for_each_entry_safe\n");
		list_for_each_entry_safe(n1, tmpn1, &head1.head, node){
			list_del(&n1->node);
			kfree(n1);
		}
		pr_info("Delete List n2 used list_for_each_entry_safe\n");
		list_for_each_entry_safe(n2, tmpn2, &head2.head, node){
			list_del(&n2->node);
			kfree(n2);
		}
	}

	pr_info("Test List [X]\n");
}

static int __init list_dev_init(void)
{
	pr_info("list init[E]");
	list_test();
	pr_info("list init[X]");
    return 0;
}
module_init(list_dev_init);

static void __exit list_dev_exit(void)
{
	pr_info("list exit");
}
module_exit(list_dev_exit);

MODULE_LICENSE("GPL v2");
MODULE_INFO(supported, "Test driver that simulate serial port over PCI");