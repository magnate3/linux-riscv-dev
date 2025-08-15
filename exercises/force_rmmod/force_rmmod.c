#include <linux/module.h>
#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/cpumask.h>
#include <linux/list.h>
#include <asm-generic/local.h>
#include <linux/platform_device.h>
#include <linux/kallsyms.h>
#include <linux/sched.h>


/*
 *  ����ģ���ʱ��, �����ַ�����ģ���һ��ȫ���ַ���������
 *
 *  module_param_string(name, string, len, perm)
 *
 *  @name   �ڼ���ģ��ʱ������������
 *  @string ģ���ڲ����ַ����������
 *  @len    ģ���ڲ����ַ�����Ĵ�С
 *  #perm   ����Ȩ��
 *
 * */
static char *modname = NULL;
module_param(modname, charp, 0644);
MODULE_PARM_DESC(modname, "The name of module you want do clean or delete...\n");


//#define CONFIG_REPLACE_EXIT_FUNCTION

#ifdef CONFIG_REPLACE_EXIT_FUNCTION
//  �˴�Ϊ�ⲿע��Ĵ�ж��ģ���exit����
//  �������ģ��ԭ����exit����
//  ע��--�˺���������Ҫ����ɾ��ģ������, ��˲�������Ϊstatic
/* static */ void force_replace_exit_module_function(void)
{
    /////////////////////
    //  �˴����ƴ�ж�������� exit/cleanup ����
    /////////////////////

    printk("module %s exit SUCCESS...\n", modname);
//    platform_device_unregister((struct platform_device*)(*(int*)symbol_addr));
}
#endif  //  CONFIG_REPLACE_EXIT_FUNCTION


static int force_cleanup_module(char *del_mod_name)
{
    struct module   *mod = NULL, *relate = NULL;
    int              cpu;
#ifdef CONFIG_REPLACE_EXIT_FUNCTION
    void            *origin_exit_addr = NULL;
#endif

    /////////////////////
    //  �ҵ���ɾ��ģ����ں�module��Ϣ
    /////////////////////
#if 0
    //  ����һ, �����ں�ģ����list_mod��ѯ
    struct module *list_mod = NULL;
    /*  ����ģ���б�, ���� del_mod_name ģ��  */
    list_for_each_entry(list_mod, THIS_MODULE->list.prev, list)
    {
        if (strcmp(list_mod->name, del_mod_name) == 0)
        {
            mod = list_mod;
        }
    }
    /*  ���δ�ҵ� del_mod_name ��ֱ���˳�  */
    if(mod == NULL)
    {
        printk("[%s] module %s not found\n", THIS_MODULE->name, modname);
        return -1;
    }
#endif

    //  ������, ͨ��find_mod��������
    if((mod = find_module(del_mod_name)) == NULL)
    {
        printk("[%s] module %s not found\n", THIS_MODULE->name, del_mod_name);
        return -1;
    }
    else
    {
        printk("[before] name:%s, state:%d, refcnt:%u\n",
                mod->name ,mod->state, module_refcount(mod));
    }

    /////////////////////
    //  ������������������ڵ�ǰ����, ����ǿ��ж��, �����˳�
    /////////////////////
    /*  ���������ģ�������� del_mod  */
    if (!list_empty(&mod->source_list))
    {
        /*  ��ӡ����������target��ģ����  */
        list_for_each_entry(relate, &mod->source_list, source_list)
        {
            printk("[relate]:%s\n", relate->name);
        }
    }
    else
    {
        printk("No modules depond on %s...\n", del_mod_name);
    }

    /////////////////////
    //  ���������״̬�����ü���
    /////////////////////
    //  ����������״̬ΪLIVE
    mod->state = MODULE_STATE_LIVE;

    //  ������������ü���
    for_each_possible_cpu(cpu)
    {
        local_set((local_t*)per_cpu_ptr(&(mod->refcnt), cpu), 0);
        //local_set(__module_ref_addr(mod, cpu), 0);
        //per_cpu_ptr(mod->refptr, cpu)->decs;
        //module_put(mod);
    }
    atomic_set(&mod->refcnt, 1);

#ifdef CONFIG_REPLACE_EXIT_FUNCTION
    /////////////////////
    //  ����ע��������exit����
    /////////////////////
    origin_exit_addr = mod->exit;
    if (origin_exit_addr == NULL)
    {
        printk("module %s don't have exit function...\n", mod->name);
    }
    else
    {
        printk("module %s exit function address %p\n", mod->name, origin_exit_addr);
    }

    mod->exit = force_replace_exit_module_function;
    printk("replace module %s exit function address (%p -=> %p)\n", mod->name, origin_exit_addr, mod->exit);
#endif

    printk("[after] name:%s, state:%d, refcnt:%u\n",
            mod->name, mod->state, module_refcount(mod));

    return 0;
}


static int __init force_rmmod_init(void)
{
    return force_cleanup_module(modname);
}


static void __exit force_rmmod_exit(void)
{
    printk("=======name : %s, state : %d EXIT=======\n", THIS_MODULE->name, THIS_MODULE->state);
}

module_init(force_rmmod_init);
module_exit(force_rmmod_exit);

MODULE_LICENSE("GPL");